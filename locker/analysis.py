from collections import OrderedDict
from itertools import count
from warnings import warn

from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm
from pycircstat import event_series as es
import pycircstat as circ
from .utils.data import peakdet
from .utils.locking import find_significant_peaks, find_best_locking, vector_strength_at
from . import colordict
import numpy as np
import pandas as pd
import seaborn as sns
import sympy
from scipy import optimize, stats, signal

import datajoint as dj

from .data import Runs, Cells, BaseEOD, Baseline

schema = dj.schema('efish_analysis', locals())


class PlotableSpectrum:
    def plot(self, ax, restrictions, f_max=2000, ncol=None):
        sns.set_context('paper')
        # colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
        # colors = ['deeppink', 'dodgerblue', sns.xkcd_rgb['mustard'], sns.xkcd_rgb['steel grey']]

        markers = [(4, 0, 90), '^', 'D', 's', 'o']
        stim, eod, baseline, beat = sympy.symbols('f_s, EODf, f_b, \Delta')

        for fos in ((self * Runs()).proj() & restrictions).fetch(as_dict=True):
            if isinstance(self, FirstOrderSpikeSpectra):
                peaks = (FirstOrderSignificantPeaks() * restrictions & fos)
            elif isinstance(self, SecondOrderSpikeSpectra):
                if isinstance(restrictions, dict):
                    peaks = (SecondOrderSignificantPeaks() & restrictions & fos)
                else:
                    peaks = (SecondOrderSignificantPeaks() * restrictions & fos)
            else:
                raise Exception("Mother class unknown!")

            f, v, alpha, cell, run = (self & fos).fetch1('frequencies', 'vector_strengths', 'critical_value',
                                                         'cell_id', 'run_id')

            # insert refined vector strengths
            peak_f, peak_v = peaks.fetch('frequency', 'vector_strength')
            f = np.hstack((f, peak_f))
            v = np.hstack((v, peak_v))
            idx = np.argsort(f)
            f, v = f[idx], v[idx]

            # only take frequencies within defined ange
            idx = (f >= 0) & (f <= f_max) & ~np.isnan(v)
            ax.fill_between(f[idx], 0 * f[idx], 0 * f[idx] + alpha, lw=0, color='silver')
            ax.fill_between(f[idx], 0 * f[idx], v[idx], lw=0, color='darkslategray')

            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('vector strength')
            ax.set_ylim((0, 1.))
            ax.set_xlim((0, f_max))
            ax.set_yticks([0, .25, .5, .75, 1.0])

            df = pd.DataFrame(peaks.fetch())
            df['on'] = np.abs(df.ix[:, :3]).sum(axis=1)
            df = df[df.frequency > 0]

            for freq, freq_group in df.groupby('frequency'):  # get all combinations that have the same frequency

                freq_group = freq_group[
                    freq_group.on == freq_group.on.min()]  # take the ones that have the lowest factors

                def label_order(x):
                    if 'EODf' in x[0]:
                        return 0
                    elif 'stimulus' in x[0]:
                        return 1
                    elif 'Delta' in x[0]:
                        return 2
                    else:
                        return 3

                for i, (cs, ce, cb, freq, vs) in freq_group[
                    ['stimulus_coeff', 'eod_coeff', 'baseline_coeff', 'frequency', 'vector_strength']].iterrows():
                    terms = []
                    if 0 <= freq <= f_max:
                        term = cs * stim + ce * eod + cb * baseline
                        if (cs < 0 and ce > 0) or (cs > 0 and ce < 0):
                            coeff = np.sign(ce) * min(abs(cs), abs(ce))
                            term = term + coeff * (stim - eod) - coeff * beat
                        terms.append(sympy.latex(term.simplify()))
                    term = ' = '.join(terms)
                    fontsize = ax.xaxis.get_ticklabels()[0].get_fontsize()
                    # use different colors and labels depending on the frequency
                    if cs != 0 and ce == 0 and cb == 0:
                        ax.plot(freq, vs, 'k', mfc=colordict['stimulus'],
                                label='$f_s$={:.0f} Hz'.format(freq, ) if cs == 1 else None, marker=markers[0],
                                linestyle='None')

                    elif cs == 0 and ce != 0 and cb == 0:
                        ax.plot(freq, vs, 'k', mfc=colordict['eod'],
                                label='EODf = {:.0f} Hz'.format(freq) if ce == 1 else None,
                                marker=markers[1],
                                linestyle='None')
                    elif cs == 0 and ce == 0 and cb != 0:
                        ax.plot(freq, vs, 'k', mfc=colordict['baseline'],
                                label='baseline firing = {:.0f} Hz'.format(freq) if cb == 1 else None,
                                marker=markers[2],
                                linestyle='None')
                    elif cs == 1 and ce == -1 and cb == 0:
                        ax.plot(freq, vs, 'k', mfc=colordict['delta_f'],
                                label=r'$\Delta f$=%.0f Hz' % freq,
                                marker=markers[3],
                                linestyle='None')
                    else:
                        ax.plot(freq, vs, 'k', mfc=colordict['combinations'], label='combinations', marker=markers[4],
                                linestyle='None')
                    term = term.replace('1.0 ', ' ')
                    term = term.replace('.0 ', ' ')
                    term = term.replace('EODf', '\\mathdefault{EODf}')
                    term = term.replace('\\Delta', '\\Delta f')

                    ax.text(freq - 20, vs + 0.05, r'${}$'.format(term),
                            fontsize=fontsize, rotation=90, ha='left', va='bottom')
            handles, labels = ax.get_legend_handles_labels()

            by_label = OrderedDict(sorted(zip(labels, handles), key=label_order))
            ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1.3),
                      ncol=len(by_label) if ncol is None else ncol)


@schema
class CoincidenceTolerance(dj.Lookup):
    definition = """
    # Coincidence tolerance of EOD and stimulus phase in s

    coincidence_idx         : int
    ---
    tol                     : double
    """

    contents = [(0, 0.0001), ]


@schema
class SpectraParameters(dj.Lookup):
    definition = """
    spectra_setting     : tinyint   # index of the setting
    ---
    f_max               : float     # maximal frequency considered
    """

    contents = [(0, 2000)]


@schema
class TrialAlign(dj.Computed):
    definition = """
    # computes a time point where the EOD and the stimulus coincide

    -> Runs                         # each trial has an alignmnt point
    -> CoincidenceTolerance         # tolerance of alignment
    ---
    """

    class Alignment(dj.Part):
        definition = """
        -> master
        -> Runs.SpikeTimes                   
        -> Runs.GlobalEODPeaksTroughs
        -> Runs.GlobalEFieldPeaksTroughs
        --- 
        t0                          : double   # time where the trial will be aligned to
        """

    @property
    def key_source(self):
        return Runs() * CoincidenceTolerance() & dict(am=0, n_harmonics=0) # TODO: is n_harmonics=0 necessary, what's with the newly recorded cells?

    def _make_tuples(self, key):
        print('Populating', key)
        tol = CoincidenceTolerance().fetch1('tol')
        samplingrate = (Runs() & key).fetch1('samplingrate')
        trials = Runs.GlobalEODPeaksTroughs() * \
                 Runs.GlobalEFieldPeaksTroughs().proj(stim_peaks='peaks') * \
                 Runs.SpikeTimes() & key

        self.insert1(key)
        for trial_key in tqdm(trials.fetch.keys()):
            ep, sp = (trials & trial_key).fetch1('peaks', 'stim_peaks')
            p0 = ep[np.abs(sp[:, None] - ep[None, :]).min(axis=0) <= tol * samplingrate] / samplingrate
            if len(p0) == 0:
                warn('Could not find an alignment within given tolerance of {}s. Skipping!'.format(tol))
                continue
            else:
                self.Alignment().insert1(dict(trial_key, **key, t0=p0.min()), ignore_extra_fields=True)

    def load_trials(self, restriction):
        """
        Loads aligned trials.

        :param restriction: restriction on Runs.SpikeTimes() * TrialAlign()
        :returns: aligned trials; spike times are in seconds
        """

        trials = Runs.SpikeTimes() * TrialAlign.Alignment() & restriction
        return [s / 1000 - t0 for s, t0 in zip(*trials.fetch('times', 't0'))]

    def plot(self, ax, restriction):
        trials = self.load_trials(restriction)
        for i, trial in enumerate(trials):
            ax.plot(trial, 0 * trial + i, '.k', ms=1)

        ax.set_ylabel('trial no')
        ax.set_xlabel('time [s]')

    def plot_traces(self, ax, restriction):
        sampling_rate = (Runs() & restriction).fetch('samplingrate')
        sampling_rate = np.unique(sampling_rate)

        assert len(sampling_rate) == 1, 'Sampling rate must be unique by restriction'
        sampling_rate = sampling_rate[0]

        trials = Runs.GlobalEOD() * Runs.GlobalEField() * TrialAlign.Alignment() & restriction

        t = np.arange(0, 0.01, 1 / sampling_rate)
        n = len(t)
        for geod, gef, t0 in zip(*trials.fetch('global_efield', 'global_voltage', 't0')):
            ax.plot(t - t0, geod[:n], '-', color='dodgerblue', lw=.1)
            ax.plot(t - t0, gef[:n], '-', color='k', lw=.1)


@schema
class FirstOrderSpikeSpectra(dj.Computed, PlotableSpectrum):
    definition = """
    # table that holds 1st order vector strength spectra

    -> Runs                         # each run has a spectrum
    -> SpectraParameters
    ---

    frequencies             : longblob # frequencies at which the spectra are computed
    vector_strengths        : longblob # vector strengths at those frequencies
    critical_value          : float    # critical value for significance with alpha=0.001
    """

    @property
    def key_source(self):
        return Runs() * SpectraParameters() & TrialAlign() & dict(am=0)

    @staticmethod
    def compute_1st_order_spectrum(aggregated_spikes, sampling_rate, duration, alpha=0.001, f_max=2000):
        """
        Computes the 1st order amplitue spectrum of the spike train (i.e. the vector strength spectrum
        of the aggregated spikes).

        :param aggregated_spikes: all spike times over all trials
        :param sampling_rate: sampling rate of the spikes
        :param alpha: significance level for the boundary against non-locking
        :returns: the frequencies for the vector strength spectrum, the spectrum, and the threshold against non-locking

        """
        if len(aggregated_spikes) < 2:
            return np.array([0]), np.array([0]), 0,
        f = np.fft.fftfreq(int(duration * sampling_rate), 1 / sampling_rate)
        f = f[(f >= -f_max) & (f <= f_max)]
        v = es.direct_vector_strength_spectrum(aggregated_spikes, f)
        threshold = np.sqrt(- np.log(alpha) / len(aggregated_spikes))
        return f, v, threshold

    def _make_tuples(self, key):
        print('Processing', key['cell_id'], 'run', key['run_id'], )
        samplingrate, duration = (Runs() & key).fetch1('samplingrate', 'duration')
        f_max = (SpectraParameters() & key).fetch1('f_max')

        aggregated_spikes = TrialAlign().load_trials(key)
        if len(aggregated_spikes) > 0:
            aggregated_spikes = np.hstack(aggregated_spikes)
        else:
            warn("Trial align returned no spikes! Continuing")
            return

        key['frequencies'], key['vector_strengths'], key['critical_value'] = \
            self.compute_1st_order_spectrum(aggregated_spikes, samplingrate, duration, alpha=0.001, f_max=f_max)
        vs = key['vector_strengths']
        vs[np.isnan(vs)] = 0
        self.insert1(key)


@schema
class FirstOrderSignificantPeaks(dj.Computed):
    definition = """
    # hold significant peaks in spektra

    stimulus_coeff          : int   # how many multiples of the stimulus
    eod_coeff               : int   # how many multiples of the eod
    baseline_coeff          : int   # how many multiples of the baseline firing rate
    refined                 : int   # whether the search was refined or not
    ->FirstOrderSpikeSpectra

    ---

    frequency               : double # frequency at which there is significant locking
    vector_strength         : double # vector strength at that frequency
    tolerance               : double # tolerance within which a peak was accepted
    """

    def _make_tuples(self, key):
        double_peaks = -1
        data = (FirstOrderSpikeSpectra() & key).fetch1()
        run = (Runs() & key).fetch1()
        cell = (Cells() & key).fetch1()

        spikes = np.hstack(TrialAlign().load_trials(key))
        interesting_frequencies = {'stimulus_coeff': run['eod'] + run['delta_f'], 'eod_coeff': run['eod'],
                                   'baseline_coeff': cell['baseline']}

        f_max = (SpectraParameters() & key).fetch1('f_max')
        sas = find_significant_peaks(spikes, data['frequencies'], data['vector_strengths'],
                                     interesting_frequencies, data['critical_value'], upper_cutoff=f_max)
        for s in sas:
            s.update(key)
            try:
                self.insert1(s)
            except dj.DataJointError:  # sometimes one peak has two peaks nearby
                print("Found double peak")
                s['refined'] = double_peaks
                self.insert1(s)
                double_peaks -= 1


@schema
class SecondOrderSpikeSpectra(dj.Computed, PlotableSpectrum):
    definition = """
    # table that holds 2nd order vector strength spectra
    -> Runs                  # each run has a spectrum
    -> SpectraParameters
    ---

    frequencies             : longblob # frequencies at which the spectra are computed
    vector_strengths        : longblob # vector strengths at those frequencies
    critical_value          : float    # critical value for significance with alpha=0.001
    """

    @property
    def key_source(self):
        return Runs() * SpectraParameters() & dict(am=0)

    @staticmethod
    def compute_2nd_order_spectrum(spikes, t, sampling_rate, alpha=0.001, method='poisson', f_max=2000):
        """
        Computes the 1st order amplitue spectrum of the spike train (i.e. the vector strength spectrum
        of the aggregated spikes).


        :param spikes: list of spike trains from the single trials
        :param t: numpy.array of time points
        :param sampling_rate: sampling rate of the spikes
        :param alpha: significance level for the boundary against non-locking
        :param method: method to compute the confidence interval (poisson or gauss)
        :returns: the frequencies for the vector strength spectrum, the spectrum, and the threshold against non-locking

        """

        # compute 99% confidence interval for Null distribution of 2nd order spectra (no locking)
        spikes_per_trial = list(map(len, spikes))
        freqs, vs_spectra = zip(*[es.vector_strength_spectrum(sp, sampling_rate, time=t) for sp in spikes])
        freqs = freqs[0]
        m_ampl = np.mean(vs_spectra, axis=0)

        if method == 'poisson':
            poiss_rate = np.mean(spikes_per_trial)
            r = np.linspace(0, 2, 10000)
            dr = r[1] - r[0]
            mu = np.sum(2 * poiss_rate * r ** 2 * np.exp(poiss_rate * np.exp(-r ** 2) - poiss_rate - r ** 2) / (
                    1 - np.exp(-poiss_rate))) * dr
            s = np.sum(2 * poiss_rate * r ** 3 * np.exp(poiss_rate * np.exp(-r ** 2) - poiss_rate - r ** 2) / (
                    1 - np.exp(-poiss_rate))) * dr
            s2 = np.sqrt(s - mu ** 2.)
            y = stats.norm.ppf(1 - alpha, loc=mu,
                               scale=s2 / np.sqrt(len(spikes_per_trial)))  # use central limit theorem

        elif method == 'gauss':
            n = np.asarray(spikes_per_trial)
            mu = np.sqrt(np.pi) / 2. * np.mean(1. / np.sqrt(n))
            N = len(spikes_per_trial)
            s = np.sqrt(np.mean(1. / n - np.pi / 4. / n) / N)
            y = stats.norm.ppf(1 - alpha, loc=mu, scale=s)
        else:
            raise ValueError("Method %s not known" % (method,))
        idx = (freqs >= -f_max) & (freqs <= f_max)
        return freqs[idx], m_ampl[idx], y

    def _make_tuples(self, key):
        print('Processing {cell_id} run {run_id}'.format(**key))
        dat = (Runs() & key).fetch1(as_dict=True)
        dt = 1 / dat['samplingrate']
        t = np.arange(0, dat['duration'], dt)
        st = (Runs.SpikeTimes() & key).fetch(as_dict=True)
        st = [s['times'] / 1000 for s in st if len(s) > 0]  # convert to s and drop empty trials
        f_max = (SpectraParameters() & key).fetch1('f_max')

        key['frequencies'], key['vector_strengths'], key['critical_value'] = \
            SecondOrderSpikeSpectra.compute_2nd_order_spectrum(st, t, 1 / dt, alpha=0.001, method='poisson',
                                                               f_max=f_max)
        self.insert1(key)


@schema
class SecondOrderSignificantPeaks(dj.Computed):
    definition = """
    # hold significant peaks in spektra

    stimulus_coeff          : int   # how many multiples of the stimulus
    eod_coeff               : int   # how many multiples of the eod
    baseline_coeff          : int   # how many multiples of the baseline firing rate
    refined                 : int   # whether the search was refined or not
    ->SecondOrderSpikeSpectra

    ---

    frequency               : double # frequency at which there is significant locking
    vector_strength         : double # vector strength at that frequency
    tolerance               : double # tolerance within which a peak was accepted
    """

    def _make_tuples(self, key):
        double_peaks = -1
        data = (SecondOrderSpikeSpectra() & key).fetch1()
        run = (Runs() & key).fetch1()
        cell = (Cells() & key).fetch1()

        st = (Runs.SpikeTimes() & key).fetch(as_dict=True)
        spikes = [s['times'] / 1000 for s in st]  # convert to s

        interesting_frequencies = {'stimulus_coeff': run['eod'] + run['delta_f'], 'eod_coeff': run['eod'],
                                   'baseline_coeff': cell['baseline']}
        f_max = (SpectraParameters() & key).fetch1('f_max')
        sas = find_significant_peaks(spikes, data['frequencies'], data['vector_strengths'],
                                     interesting_frequencies, data['critical_value'], upper_cutoff=f_max)
        for s in sas:
            s.update(key)

            try:
                self.insert1(s)
            except dj.DataJointError:  # sometimes one peak has two peaks nearby
                print("Found double peak")
                s['refined'] = double_peaks
                self.insert1(s)
                double_peaks -= 1


@schema
class SamplingPointsPerBin(dj.Lookup):
    definition = """
    # sampling points per bin

    n           : int # sampling points per bin
    ---

    """

    contents = [(2,), (4,), (8,)]


@schema
class PhaseLockingHistogram(dj.Computed):
    definition = """
    # phase locking histogram at significant peaks

    -> FirstOrderSignificantPeaks
    ---
    locking_frequency       : double   # frequency for which the locking is computed
    peak_frequency          : double   # frequency as determined by the peaks of the electric field
    spikes                  : longblob # union of spike times over trials relative to period of locking frequency
    vector_strength         : double   # vector strength computed from the spikes for sanity checking
    """

    class Histograms(dj.Part):
        definition = """
        ->PhaseLockingHistogram
        ->SamplingPointsPerBin
        ---
        bin_width_radians       : double   # bin width in radians
        bin_width_time          : double   # bin width in time
        histogram               : longblob # vector of counts
        """

    @property
    def key_source(self):
        return FirstOrderSignificantPeaks() \
               & 'baseline_coeff=0' \
               & '((stimulus_coeff=1 and eod_coeff=0) or (stimulus_coeff=0 and eod_coeff=1))' \
               & 'refined=1'

    def _make_tuples(self, key):
        key_sub = dict(key)
        delta_f, eod, samplingrate = (Runs() & key).fetch1('delta_f', 'eod', 'samplingrate')
        locking_frequency = (FirstOrderSignificantPeaks() & key).fetch1('frequency')

        if key['eod_coeff'] > 0:
            # convert spikes to s and center on first peak of eod
            # times, peaks = (Runs.SpikeTimes() * LocalEODPeaksTroughs() & key).fetch('times', 'peaks')
            peaks = (Runs.GlobalEODPeaksTroughs() & key).fetch('peaks')
        #
        #     spikes = np.hstack([s / 1000 - p[0] / samplingrate for s, p in zip(times, peaks)])
        else:
            #     # convert spikes to s and center on first peak of stimulus
            #     times, peaks = (Runs.SpikeTimes() * GlobalEFieldPeaksTroughs() & key).fetch('times', 'peaks')
            peaks = (Runs.GlobalEFieldPeaksTroughs() & key).fetch('peaks')
        # spikes = np.hstack([s / 1000 - p[0] / samplingrate for s, p in zip(times, peaks)])

        spikes = np.hstack(TrialAlign().load_trials(key))
        key['peak_frequency'] = samplingrate / np.mean([np.diff(p).mean() for p in peaks])
        key['locking_frequency'] = locking_frequency

        cycle = 1 / locking_frequency
        spikes %= cycle

        key['spikes'] = spikes / cycle * 2 * np.pi
        key['vector_strength'] = 1 - circ.var(key['spikes'])

        self.insert1(key)

        histograms = self.Histograms()
        for n in SamplingPointsPerBin().fetch:
            n = int(n[0])
            bin_width_time = n / samplingrate
            bin_width_radians = bin_width_time / cycle * np.pi * 2
            bins = np.arange(0, cycle + bin_width_time, bin_width_time)
            key_sub['n'] = n
            key_sub['histogram'], _ = np.histogram(spikes, bins=bins)
            key_sub['bin_width_time'] = bin_width_time
            key_sub['bin_width_radians'] = bin_width_radians

            histograms.insert1(key_sub)

    def violin_plot(self, ax, restrictions, palette):
        runs = Runs() * self & restrictions
        if len(runs) == 0:
            return

        df = pd.concat([pd.DataFrame(item) for item in runs.fetch(as_dict=True)])
        df.ix[df.stimulus_coeff == 1, 'type'] = 'stimulus'
        df.ix[df.eod_coeff == 1, 'type'] = 'EOD'
        delta_fs = np.unique(runs.fetch('delta_f'))
        delta_fs = delta_fs[np.argsort(-delta_fs)]

        sns.violinplot(data=df, y='delta_f', x='spikes', hue='type', split=True, ax=ax, hue_order=['EOD', 'stimulus'],
                       order=delta_fs, palette=palette, cut=0, inner=None, linewidth=.5,
                       orient='h', bw=.05)


@schema
class EODStimulusPSTSpikes(dj.Computed):
    definition = """
    # PSTH of Stimulus and EOD at the difference frequency of both

    -> FirstOrderSignificantPeaks
    -> CoincidenceTolerance
    cycle_idx                : int  # index of the cycle
    ---
    stimulus_frequency       : double
    eod_frequency            : double
    window_half_size         : double # spikes will be extracted around +- this size around in phase points of stimulus and eod
    vector_strength_eod      : double   # vector strength of EOD
    vector_strength_stimulus : double   # vector strength of stimulus
    spikes                   : longblob # spikes in that window
    efield                   : longblob # stimulus + eod
    """

    @property
    def key_source(self):
        constr = dict(stimulus_coeff=1, baseline_coeff=0, eod_coeff=0, refined=1)
        cell_type = Cells() & dict(cell_type='p-unit')
        return FirstOrderSignificantPeaks() * CoincidenceTolerance() & cell_type & constr

    def _make_tuples(self, key):
        # key_sub = dict(key)
        print('Populating', key, flush=True)
        delta_f, eod, samplingrate, duration = (Runs() & key).fetch1('delta_f', 'eod', 'samplingrate', 'duration')
        runs_stim = Runs() * FirstOrderSignificantPeaks() & key
        runs_eod = Runs() * FirstOrderSignificantPeaks() & dict(key, stimulus_coeff=0, eod_coeff=1)

        if len(runs_eod) > 0:
            # duration = runs_eod.fetch1('duration')
            tol = (CoincidenceTolerance() & key).fetch1('tol')
            eod_period = 1 / runs_eod.fetch1('frequency')

            whs = 10 * eod_period

            times, peaks, epeaks, global_eod = \
                (Runs.SpikeTimes() * Runs.GlobalEODPeaksTroughs() * Runs.LocalEOD() \
                 * Runs.GlobalEFieldPeaksTroughs().proj(epeaks='peaks') \
                 & key).fetch('times', 'peaks', 'epeaks', 'local_efield')

            p0 = [peaks[i][
                      np.abs(epeaks[i][:, None] - peaks[i][None, :]).min(axis=0) <= tol * samplingrate] / samplingrate
                  for i in range(len(peaks))]

            spikes, eod, field = [], [], []
            t = np.linspace(0, duration, duration * samplingrate, endpoint=False)
            sampl_times = np.linspace(-whs, whs, 1000)

            for train, eftrain, in_phase in zip(times, global_eod, p0):
                train = np.asarray(train) / 1000  # convert to seconds
                for phase in in_phase:
                    chunk = train[(train >= phase - whs) & (train <= phase + whs)] - phase
                    if len(chunk) > 0:
                        spikes.append(chunk)
                        field.append(np.interp(sampl_times + phase, t, eftrain))

            key['eod_frequency'] = runs_eod.fetch1('frequency')
            key['vector_strength_eod'] = runs_eod.fetch1('vector_strength')
            key['stimulus_frequency'] = runs_stim.fetch1('frequency')
            key['vector_strength_stimulus'] = runs_stim.fetch1('vector_strength')
            key['window_half_size'] = whs

            for cycle_idx, train, ef in zip(count(), spikes, field):
                key['spikes'] = train
                key['cycle_idx'] = cycle_idx
                key['efield'] = ef
                self.insert1(key)

    def plot(self, ax, restrictions, coincidence=0.0001, repeats=200):
        rel = self * CoincidenceTolerance() * Runs().proj('delta_f') & restrictions & dict(tol=coincidence)
        df = pd.DataFrame(rel.fetch())
        df['adelta_f'] = np.abs(df.delta_f)
        df['sdelta_f'] = np.sign(df.delta_f)
        df.sort_values(['adelta_f', 'sdelta_f'], inplace=True)
        eod = (Runs() & restrictions).fetch('eod').mean()
        if len(df) > 0:
            whs = df.window_half_size.mean()
            cycles = int(whs * eod) * 2
            db = 2 * whs / 400
            bins = np.arange(-whs, whs + db, db)
            g = np.exp(-np.linspace(-whs, whs, len(bins) - 1) ** 2 / 2 / (whs / 25) ** 2)
            print('Low pass kernel sigma=', whs / 25)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            y = [0]
            yticks = []
            i = 0
            for (adf, sdf), dgr in df.groupby(['adelta_f', 'sdelta_f'], sort=True):
                delta_f = adf * sdf
                yticks.append(delta_f)
                n_trials = min(repeats, len(dgr.spikes))
                dgr = dgr[:n_trials]
                h, _ = np.histogram(np.hstack(dgr.spikes), bins=bins)

                for sp in dgr.spikes:
                    ax.plot(sp, 0 * sp + i, '.k', mfc='k', ms=1, zorder=-10, rasterized=False)
                    i += 1
                y.append(i)
                h = np.convolve(h, g, mode='same')
                h *= (y[-1] - y[-2]) / h.max()
                ax.fill_between(bin_centers, 0 * h + y[-2], h + y[-2], color='silver', zorder=-20)

            if BaseEOD() & restrictions:
                t, e, pe = (BaseEOD() & restrictions).fetch1('time', 'eod_ampl', 'max_idx')
                t = t / 1000
                pe_t = t[pe]
                t = t - pe_t[cycles // 2]
                fr, to = pe[0], pe[cycles]
                t, e = t[fr:to], e[fr:to]
                e = Baseline.clean_signal(e, eod, t[1] - t[0])
                dy = 0.15 * (y[-1] - y[0])
                e = (e - e.min()) / (e.max() - e.min()) * dy
                ax.plot(t, e + y[-1], lw=2, color='steelblue', zorder=-15, label='EOD')
                y.append(y[-1] + 1.2 * dy)

            y = np.asarray(y)
            ax.set_xlim((-whs, whs))
            ax.set_xticks([-whs, -whs / 2, 0, whs / 2, whs])

            ax.set_xticklabels([-10, -5, 0, 5, 10])
            ax.set_ylabel(r'$\Delta f$ [Hz]')
            ax.tick_params(axis='y', length=3, width=1, which='major')

            ax.set_ylim(y[[0, -1]])
            y = y[:-1]
            ax.set_yticks(0.5 * (y[1:] + y[:-1]))
            ax.set_yticklabels(['%.0f' % yt for yt in yticks])

    def plot_single(self, ax, restrictions, coincidence=0.0001, repeats=20):

        rel = self * CoincidenceTolerance() * Runs().proj('delta_f', 'contrast') & restrictions & dict(tol=coincidence)
        df = pd.DataFrame(rel.fetch())
        samplingrate, eod = (Runs() & restrictions).fetch1('samplingrate', 'eod')

        if len(df) > 0:
            whs = df.window_half_size.mean()
            db = 1 / eod
            bins = np.arange(-whs, whs + db, db)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            # --- histogram
            h, _ = np.histogram(np.hstack(df.spikes), bins=bins)
            f_max = h.max() / db / len(df.spikes)

            h = h.astype(np.float64)
            h *= repeats / h.max() / 2
            ax.bar(bin_centers, h, align='center', width=db, color='lightgray', zorder=-20, lw=0, label='PSTH')
            ax.plot(bin_centers[0] * np.ones(2), [repeats // 8, h.max() * 450 / f_max + repeats // 8], '-',
                    color='darkslategray',
                    lw=3, solid_capstyle='butt')
            ax.text(bin_centers[0] + db / 4, repeats / 5, '450 Hz')

            # y = np.asarray(y)
            if len(df) > repeats:
                df = df[:repeats]

            for offset, sp in zip(count(start=repeats // 2 + 1), df.spikes):
                # ax.plot(sp, 0 * sp + offset, '.k', mfc='k', ms=2, zorder=-10, rasterized=False,
                #         label='spikes' if offset == repeats // 2 + 1 else None)
                ax.vlines(sp, 0 * sp + offset, 0 * sp + offset + 1, 'k', zorder=-10, rasterized=False,
                          label='spikes' if offset == repeats // 2 + 1 else None)
                offset += 1
            norm = lambda x: (x - x.min()) / (x.max() - x.min())

            avg_efield = norm(np.mean(df.efield, axis=0)) * repeats / 2
            t = np.linspace(-whs, whs, len(avg_efield), endpoint=False)
            high, hidx, low, lidx = peakdet(avg_efield, delta=0.01)
            fh = InterpolatedUnivariateSpline(t[hidx], high, k=3)
            fl = InterpolatedUnivariateSpline(t[lidx], low, k=3)
            ax.plot(t, avg_efield + offset, lw=2, color=colordict['stimulus'], zorder=-15,
                    label='stimulus + EOD')
            ax.plot(t, fh(t) + offset, lw=2, color=colordict['delta_f'], zorder=-15, label='AM')
            ax.plot(t, fl(t) + offset, lw=2, color=colordict['delta_f'], zorder=-15)

            ax.set_xlim((-whs, whs))
            ax.set_xticks([-whs, -whs / 2, 0, whs / 2, whs])

            ax.set_xticklabels([-10, -5, 0, 5, 10])
            ax.tick_params(axis='y', length=0, width=0, which='major')
            ax.set_yticks([])
            # ax.set_yticklabels(['%.0f' % yt for yt in yticks])
            ax.set_ylim((0, 2.4 * repeats))


@schema
class Decoding(dj.Computed):
    definition = """
    # locking by decoding time

    -> Runs
    ---
    beat                    : float    # refined beat frequency
    stimulus                : float    # refined stimulus frequency
    """

    class Beat(dj.Part):
        definition = """
        -> Decoding
        -> Runs.SpikeTimes
        ---
        crit_beat=null               : float # critical value for beat locking
        vs_beat=null                 : float # vector strength for full trial
        """

    class Stimulus(dj.Part):
        definition = """
        -> Decoding
        -> Runs.SpikeTimes
        ---
        crit_stimulus=null           : float # critical value for stimulus locking
        vs_stimulus=null             : float # vector strength for full trial
        """

    @property
    def key_source(self):
        return Runs() * Cells() & dict(cell_type='p-unit')

    def _make_tuples(self, key):
        print('Processing', key['cell_id'], 'run', key['run_id'], )
        dat = (Runs() & key).fetch(as_dict=True)[0]

        spike_times, trial_ids = (Runs.SpikeTimes() & key).fetch('times', 'trial_id')
        spike_times = [s / 1000 for s in spike_times]  # convert to s

        # refine delta f locking on all spikes
        delta_f = find_best_locking(spike_times, [dat['delta_f']], tol=3)[0][0]
        stimulus_frequency = find_best_locking(spike_times, [dat['delta_f'] + dat['eod']], tol=3)[0][0]

        self.insert1(dict(key, beat=delta_f, stimulus=stimulus_frequency))
        stim = self.Stimulus()
        beat = self.Beat()
        for key['trial_id'], trial in zip(trial_ids, spike_times):
            v, c = vector_strength_at(stimulus_frequency, trial, alpha=0.001)
            if np.isinf(c):
                c = np.NaN
            stim.insert1(dict(key, vs_stimulus=v, crit_stimulus=c))
            v, c = vector_strength_at(delta_f, trial, alpha=0.001)
            if np.isinf(c):
                c = np.NaN
            beat.insert1(dict(key, vs_beat=v, crit_beat=c))


@schema
class BaselineSpikeJitter(dj.Computed):
    definition = """
    # circular variance and mean of spike times within an EOD period

    -> Baseline

    ---

    base_var         : double # circular variance
    base_std         : double # circular std
    base_mean        : double # circular mean
    """

    @property
    def key_source(self):
        return Baseline() & Baseline.LocalEODPeaksTroughs() & dict(cell_type='p-unit')

    def _make_tuples(self, key):
        print('Processing', key['cell_id'])
        sampling_rate, eod = (Baseline() & key).fetch1('samplingrate', 'eod')
        dt = 1. / sampling_rate

        trials = Baseline.LocalEODPeaksTroughs() * Baseline.SpikeTimes() & key

        aggregated_spikes = np.hstack([s / 1000 - p[0] * dt for s, p in zip(*trials.fetch('times', 'peaks'))])

        aggregated_spikes %= 1 / eod

        aggregated_spikes *= eod * 2 * np.pi  # normalize to 2*pi
        key['base_var'], key['base_mean'], key['base_std'] = \
            circ.var(aggregated_spikes), circ.mean(aggregated_spikes), circ.std(aggregated_spikes)
        self.insert1(key)


@schema
class StimulusSpikeJitter(dj.Computed):
    definition = """
    # circular variance and std of spike times within an EOD period during stimulation

    -> Runs

    ---
    stim_var         : double # circular variance
    stim_std         : double # circular std
    stim_mean        : double # circular mean
    """

    @property
    def key_source(self):
        return Runs() & TrialAlign()

    def _make_tuples(self, key):
        print('Processing', key['cell_id'], 'run', key['run_id'])
        if SecondOrderSignificantPeaks() & dict(key, eod_coeff=1, stimulus_coeff=0, baseline_coeff=0, refined=1):
            eod, vs = (SecondOrderSignificantPeaks() & dict(key, eod_coeff=1, stimulus_coeff=0, baseline_coeff=0,
                                                            refined=1)).fetch1('frequency', 'vector_strength')
        elif SecondOrderSignificantPeaks() & dict(key, eod_coeff=1, stimulus_coeff=0, baseline_coeff=0, refined=0):
            eod, vs = (SecondOrderSignificantPeaks() & dict(key, eod_coeff=1, stimulus_coeff=0, baseline_coeff=0,
                                                            refined=0)).fetch1('frequency', 'vector_strength')
        else:
            eod = (Runs() & key).fetch1('eod')

        aggregated_spikes = TrialAlign().load_trials(key)
        if len(aggregated_spikes) == 0:
            warn('TrialAlign returned no spikes. Skipping')
            return
        else:
            aggregated_spikes = np.hstack(aggregated_spikes)
        aggregated_spikes %= 1 / eod

        aggregated_spikes *= eod * 2 * np.pi  # normalize to 2*pi
        if len(aggregated_spikes) > 1:
            key['stim_var'], key['stim_mean'], key['stim_std'] = \
                circ.var(aggregated_spikes), circ.mean(aggregated_spikes), circ.std(aggregated_spikes)
            self.insert1(key)
