import warnings
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.integrate import odeint
from scipy.signal import butter, filtfilt

import datajoint as dj
import pycircstat as circ
import datajoint as dj

from .utils.modeling import estimate_fundamental, get_best_time_window, get_harm_coeff
from .analysis import TrialAlign
from . import mkdir
from .utils.data import peakdet
from .data import Runs, Cells, EFishes, PaperCells
from scipy import interp
from . import colordict, markerdict

schema = dj.schema('efish_modeling', locals())


# =======================================================================================

@schema
class NoHarmonics(dj.Lookup):
    definition = """
    no_harmonics        : int  # number of harmonics that are fitted
    """

    contents = [(8,)]


@schema
class EODFit(dj.Computed):
    definition = """
    ->EFishes
    ->NoHarmonics
    ---
    fundamental     : double    # fundamental frequency
    """

    @property
    def key_source(self):
        return EFishes() * NoHarmonics() & (PaperCells() & dict(locking_experiment=1))

    def _make_tuples(self, key):
        print('Populating', pformat(key))
        if not Runs() * Runs.GlobalEOD() & key:
            print('Found no entry in Runs() * Runs.GlobalEOD() for key', key)
            return
        dat = (Runs() * Runs.GlobalEOD() & key).fetch(limit=1, as_dict=True)[0]  # get some EOD trace

        w0 = estimate_fundamental(dat['global_voltage'], dat['samplingrate'], highcut=3000, normalize=.5)
        t, win = get_best_time_window(dat['global_voltage'], dat['samplingrate'], w0, eod_cycles=10)

        fundamental = estimate_fundamental(win, dat['samplingrate'], highcut=3000)
        assert abs(fundamental - dat['eod']) < 2, \
            "EOD and fundamental estimation are more than 2Hz apart: %.2fHz, %.2fHz" % (fundamental, dat['eod'])
        harm_coeff = get_harm_coeff(t, win, fundamental, key['no_harmonics'])

        self.insert1(dict(key, fundamental=fundamental))
        for key['harmonic'], (key['sin'], key['cos']) in enumerate(harm_coeff):
            EODFit.Harmonic().insert1(key)

    class Harmonic(dj.Part):
        definition = """
        # sin and cos coefficient for harmonics
        -> EODFit
        harmonic        : int   # 0th is fundamental, 1st is frist harmonic and so on
        ---
        sin             : double # coefficient for sin
        cos             : double # coefficient for cos
        """

    def generate_eod(self, t, key):
        w = (self & key).fetch1('fundamental')

        ret = 0 * t
        VR = w * 2. * np.pi
        for i, coeff_sin, coeff_cos in zip(*(self.Harmonic() & key).fetch('harmonic', 'sin', 'cos')):
            V1 = np.sin(t * (i + 1) * VR)
            V2 = np.cos(t * (i + 1) * VR)
            V1 = V1 / np.sqrt(sum(V1 ** 2.))
            V2 = V2 / np.sqrt(sum(V2 ** 2.))

            VS = coeff_sin * V1
            VC = coeff_cos * V2
            ret = ret + VS + VC
        return ret

    def eod_func(self, key, fundamental=None, harmonics=None):
        harm_coeff = np.vstack((EODFit.Harmonic() & key).fetch('sin', 'cos', order_by='harmonic')).T
        if harmonics is not None:
            harm_coeff = harm_coeff[:harmonics + 1, :]
        if fundamental is None:
            fundamental = (self & key).fetch1('fundamental')

        A = np.sqrt(np.sum(harm_coeff[0, :] ** 2))

        def ret_func(t):
            vr = fundamental * 2 * np.pi
            ret = 0 * t
            for i, (coeff_sin, coeff_cos) in enumerate(harm_coeff / A):
                ret += coeff_sin * np.sin(t * (i + 1) * vr)
                ret += coeff_cos * np.cos(t * (i + 1) * vr)
            return ret

        return ret_func

    def plot_eods(self, fundamental=800, outdir='./'):
        t = np.linspace(0, 10 / 800, 200)
        for key in self.fetch.keys():
            print('Plotting', key)
            f = self.eod_func(key, fundamental=fundamental)
            fig, ax = plt.subplots()
            ax.plot(t, f(t), '-k')
            ax.set_xlabel('time [s]')
            ax.set_ylabel('EOD')
            fig.savefig(outdir + '/{fish_id}.png'.format(**key))
            plt.close(fig)

#
# @schema
# class LIFPUnit(dj.Computed):
#     definition = """
#     # parameters for a LIF P-Unit simulation
#
#     id           : varchar(100) # non-double unique identifier
#     ->EODFit
#     ---
#     zeta            : double
#     resonant_freq   : double    # resonant frequency of the osciallator in Hz
#     tau             : double
#     gain            : double
#     offset          : double
#     noise_sd        : double
#     threshold       : double
#     reset           : double
#     lif_tau         : double
#     """
#
#     def _make_tuples(self, key):
#         eod = (EODFit() & key).fetch1['fundamental']
#         self.insert1(dict(key, id='nwgimproved',
#                           zeta=0.2,
#                           tau=0.002,
#                           resonant_freq=eod,
#                           gain=70,
#                           offset=9,
#                           noise_sd=30,
#                           threshold=14.,
#                           reset=0.,
#                           lif_tau=0.001
#                           ))
#
#     def simulate(self, key, n, t, stimulus, y0=None):
#         """
#         Samples spikes from leaky integrate and fire neuron with id==settings_name and time t.
#         Returns n trials
#
#         :param key: key that uniquely identifies a setting
#         :param n: number of trials
#         :param t: time array
#         :param stimulus: stimulus as a function of time (function handle)
#         :return: spike times
#         """
#
#         # --- get parameters from database
#         zeta, tau, gain, wr, lif_tau, offset, threshold, reset, noisesd = (self & key).fetch1[
#             'zeta', 'tau', 'gain', 'resonant_freq', 'lif_tau', 'offset', 'threshold', 'reset', 'noise_sd']
#         wr *= 2 * np.pi
#         w0 = wr / np.sqrt(1 - 2 * zeta ** 2)
#         Zm = np.sqrt((2 * w0 * zeta) ** 2 + (wr ** 2 - w0 ** 2) ** 2 / wr ** 2)
#         alpha = wr * Zm
#
#         # --- set initial values if not given
#         if y0 is None:
#             y0 = np.zeros(3)
#
#         # --- differential equations for resonantor
#         def _d(y, t):
#             return np.array([
#                 y[1],
#                 stimulus(t) - 2 * zeta * w0 * y[1] - w0 ** 2 * y[0],
#                 (-y[2] + gain * alpha * max(y[0], 0)) / tau
#             ])
#
#         # --- simulate LIF
#         dt = t[1] - t[0]
#
#         Vin = odeint(lambda y, tt: _d(y, tt), y0, t).T[2]
#         Vin -= offset
#
#         Vout = np.zeros(n)
#
#         ret = [list() for _ in range(n)]
#
#         sdB = np.sqrt(dt) * noisesd
#
#         for i, T in enumerate(t):
#             Vout += (-Vout + Vin[i]) * dt / lif_tau + np.random.randn(n) * sdB
#             idx = Vout > threshold
#             for j in np.where(idx)[0]:
#                 ret[j].append(T)
#             Vout[idx] = reset
#
#         return tuple(np.asarray(e) for e in ret), Vin
#
# @schema
# class HarmonicStimulation(dj.Lookup):
#     definition = """
#     harmonic_stimulation  : smallint # 0 if only fundamental was used for stimulation and 1 if all harmonics were used
#     ---
#     """
#
#     contents = [(0,),(1,)]
#
# @schema
# class PUnitSimulations(dj.Computed):
#     definition = """
#     # LIF simulations
#
#     ->LIFPUnit
#     ->Runs
#     -> HarmonicStimulation
#     ---
#     dt                    : double # time resolution for differential equation
#     duration              : double # duration of trial in seconds
#     """
#
#     @property
#     def key_source(self):
#         return LIFPUnit() * Runs() * Cells()*HarmonicStimulation() & dict(am=0, n_harmonics=0, cell_type='p-unit', contrast=20)
#
#     def _make_tuples(self, key):
#         print('Populating',dict(key))
#         dt, duration = 0.000005, 1
#         trials = 50
#         ikey = dict(key)
#         ikey['dt'] = dt
#         ikey['duration'] = duration
#
#         eod = (EODFit() & key).fetch1['fundamental']
#
#         delta_f = (Runs() & key).fetch1['delta_f']
#         other_eod = eod + delta_f
#
#         t = np.arange(0, duration, dt)
#
#         baseline = EODFit().eod_func(key)
#         if key['harmonic_stimulation'] == 1:
#             foreign_eod = EODFit().eod_func(dict(fish_id='2014lepto0021'), fundamental=other_eod)
#         else:
#             foreign_eod = EODFit().eod_func(dict(fish_id='2014lepto0021'), fundamental=other_eod, harmonics=0)
#
#         bl = baseline(t)
#         foreign = foreign_eod(t)
#         fac = (bl.max() - bl.min()) * 0.2 / (foreign.max() - foreign.min())
#         stimulus = lambda tt: baseline(tt) + fac * foreign_eod(tt)
#
#         spikes_base, membran_base = LIFPUnit().simulate(key, trials, t, baseline)
#         spikes_stim, membran_stim = LIFPUnit().simulate(key, trials, t, stimulus)
#
#         n = int(duration / dt)
#         w = np.fft.fftfreq(n, d=dt)
#         w = w[(w >= 0) & (w <= 4000)]
#         vs = np.mean([circ.event_series.direct_vector_strength_spectrum(sp, w) for sp in spikes_stim], axis=0)
#         ci = second_order_critical_vector_strength(spikes_stim)
#
#         self.insert1(ikey)
#
#         for i, (bsp, ssp) in enumerate(zip(spikes_base, spikes_stim)):
#             PUnitSimulations.BaselineSpikes().insert1(dict(key, trial_idx=i, times=bsp))
#             PUnitSimulations.StimulusSpikes().insert1(dict(key, trial_idx=i, times=ssp))
#
#         PUnitSimulations.BaselineMembranePotential().insert1(dict(key, potential=membran_base))
#         PUnitSimulations.StimulusMembranePotential().insert1(dict(key, potential=membran_stim))
#         PUnitSimulations.Baseline().insert1(dict(key, signal=bl))
#         PUnitSimulations.Stimulus().insert1(dict(key, signal=stimulus(t)))
#         PUnitSimulations.StimulusSecondOrderSpectrum().insert1(dict(key, spectrum=vs, ci=ci, freq=w))
#
#     class BaselineSpikes(dj.Part):
#         definition = """
#         # holds the simulated spiketimes
#
#         ->PUnitSimulations
#         trial_idx       : int # index of trial
#         ---
#         times           : longblob # spike times
#         """
#
#     class StimulusSpikes(dj.Part):
#         definition = """
#         # holds the simulated spiketimes
#
#         ->PUnitSimulations
#         trial_idx       : int # index of trial
#         ---
#         times           : longblob # spike times
#         """
#
#     class StimulusSecondOrderSpectrum(dj.Part):
#         definition = """
#         # holds the vector strength spectrum of simulated spiketimes
#
#         ->PUnitSimulations
#         ---
#         freq               : longblob # frequencies at which the vector strengths are computed
#         spectrum           : longblob # spike times
#         ci                 : double   # (1-0.001) confidence interval
#         """
#
#     class BaselineMembranePotential(dj.Part):
#         definition = """
#         # holds the simulated membrane potential
#
#         ->PUnitSimulations
#         ---
#         potential       : longblob # membrane potential
#         """
#
#     class StimulusMembranePotential(dj.Part):
#         definition = """
#         # holds the simulated membrane potential
#
#         ->PUnitSimulations
#         ---
#         potential       : longblob # membrane potential
#         """
#
#     class Baseline(dj.Part):
#         definition = """
#         # holds the simulated membrane potential
#
#         ->PUnitSimulations
#         ---
#         signal       : longblob # membrane potential
#         """
#
#     class Stimulus(dj.Part):
#         definition = """
#         # holds the simulated membrane potential
#
#         ->PUnitSimulations
#         ---
#         signal       : longblob # membrane potential
#         """
#
#     def plot_stimulus_spectrum(self, key, ax, f_max=2000):
#         dt = (self & key).fetch1['dt']
#         eod = (EODFit() & key).fetch1['fundamental']
#         fstim = eod + (Runs() & key).fetch1['delta_f']
#
#         stimulus_signal = (PUnitSimulations.Stimulus() & key).fetch1['signal']
#         w = np.fft.fftfreq(len(stimulus_signal), d=dt)
#         idx = (w > 0) & (w < f_max)
#
#         S = np.abs(np.fft.fft(stimulus_signal))
#         S /= S.max()
#
#         ax.fill_between(w[idx], 0 * w[idx], S[idx], color='darkslategray')
#
#         # --- get parameters from database
#         zeta, tau, gain, wr, lif_tau, offset, threshold, reset, noisesd = (LIFPUnit() & key).fetch1[
#             'zeta', 'tau', 'gain', 'resonant_freq', 'lif_tau', 'offset', 'threshold', 'reset', 'noise_sd']
#         wr *= 2 * np.pi
#         w0 = wr / np.sqrt(1 - 2 * zeta ** 2)
#
#         w2 = w * 2 * np.pi
#         Zm = np.sqrt((2 * w0 * zeta) ** 2 + (w2 ** 2 - w0 ** 2) ** 2 / w2 ** 2)
#         dho = 1. / (w2[idx] * Zm[idx])
#
#         ax.plot(w[idx], dho / np.nanmax(dho), '--', dashes=(2, 2), color='gray', label='harmonic oscillator', lw=1,
#                 zorder=-10)
#         lp = 1. / np.sqrt(w2[idx] ** 2 * tau ** 2 + 1)
#         ax.plot(w[idx], lp / lp.max(), '--', color='gray', label='low pass filter', lw=1, zorder=-10)
#
#         # tmp = lp * dho
#         # tmp /=tmp.max()
#         # ax.plot(w[idx], tmp, '-', color='gray', label=' filter product', lw=1, zorder=-10)
#
#         fonsize=ax.xaxis.get_ticklabels()[0].get_fontsize()
#         ax.text(eod, 1.1, r'EODf', rotation=0, horizontalalignment='center',
#                 verticalalignment='bottom', fontsize=fonsize)
#         ax.plot(eod, val_at(w[idx], S[idx], eod), marker=markerdict['eod'], color=colordict['eod'])
#         ax.text(eod * 2, 0.35, r'2 EODf', rotation=0, horizontalalignment='center',
#                 verticalalignment='bottom', fontsize=fonsize)
#         ax.plot(eod * 2, val_at(w[idx], S[idx],2* eod), marker=markerdict['eod'], color=colordict['eod'])
#
#         ax.text(fstim, 0.27, r'$f_s$', rotation=0, horizontalalignment='center',
#                 verticalalignment='bottom', fontsize=fonsize)
#         ax.plot(fstim, val_at(w[idx], S[idx],fstim), marker=markerdict['stimulus'], color=colordict['stimulus'])
#         ax.set_ylim((0, 2))
#         ax.set_yticks([])
#         ax.set_ylabel('amplitude spectrum of\nstimulus s(t)')
#         ax.legend(loc='upper right', ncol=2)
#
#         ax.set_xlim((0, f_max))
#
#     def plot_membrane_potential_spectrum(self, key, ax, f_max=2000):
#         dt = (self & key).fetch1['dt']
#         eod = (EODFit() & key).fetch1['fundamental']
#         fstim = eod + (Runs() & key).fetch1['delta_f']
#
#         membrane_potential = (PUnitSimulations.StimulusMembranePotential() & key).fetch1['potential']
#         w = np.fft.fftfreq(len(membrane_potential), d=dt)
#         idx = (w > 0) & (w < f_max)
#
#         # fig, ax = plt.subplots()
#         # stimulus_signal = (PUnitSimulations.Stimulus() & key).fetch1['signal']
#         # print(stimulus_signal.shape, membrane_potential.shape)
#         # ax.plot(w[idx],np.abs(np.fft.fft(membrane_potential))[idx], '--b')
#         # ax.plot(w[idx],np.abs(np.fft.fft(stimulus_signal))[idx], '-r')
#         # # ax.twinx().plot(stimulus_signal[5000:8000],'-r')
#         # plt.show()
#
#         M = np.abs(np.fft.fft(membrane_potential))
#         M /= M[idx].max()
#         ax.fill_between(w[idx], 0 * w[idx], M[idx], color='darkslategray')
#         ax.set_ylim((0, 1.5))
#         fonsize = ax.xaxis.get_ticklabels()[0].get_fontsize()
#         ax.text(fstim - eod, 0.47, r'$\Delta f$', rotation=0, horizontalalignment='center',
#                 verticalalignment='bottom', fontsize=fonsize)
#
#         ax.plot(fstim - eod, val_at(w[idx], M[idx], np.abs(fstim - eod)),
#                 marker=markerdict['delta_f'], color=colordict['delta_f'])
#
#         ax.text(eod + fstim, 0.07, r'EODf + $f_s$' % (eod + fstim), rotation=0, horizontalalignment='center',
#                 verticalalignment='bottom', fontsize=fonsize)
#         ax.set_yticks([])
#         ax.set_ylabel('amplitude spectrum of\nLIF input z(t)')
#
#     def plot_spike_spectrum(self, key, ax, f_max=2000):
#         df = (Runs() & key).fetch1['delta_f']
#         eod = (EODFit() & key).fetch1['fundamental']
#         eod2 = eod + df
#         eod3 = eod - df
#
#         w, vs, ci = (PUnitSimulations.StimulusSecondOrderSpectrum() & key).fetch1['freq', 'spectrum', 'ci']
#         stimulus_spikes = (PUnitSimulations.StimulusSpikes() & key).fetch['times']
#         idx = (w > 0) & (w < f_max)
#
#         ax.set_ylim((0, .8))
#         ax.set_yticks(np.arange(0, 1, .4))
#         fonsize = ax.xaxis.get_ticklabels()[0].get_fontsize()
#         ax.fill_between(w[idx], 0 * w[idx], vs[idx], color='darkslategray')
#         ci = second_order_critical_vector_strength(stimulus_spikes)
#         ax.fill_between(w[idx], 0 * w[idx], 0 * w[idx] + ci, color='silver', alpha=.5)
#         ax.text(eod3, 0.27, r'EODf - $\Delta f$' % eod3, rotation=0, horizontalalignment='center',
#                 verticalalignment='bottom', fontsize=fonsize)
#         ax.plot(eod3, val_at(w[idx], vs[idx], eod3),
#                 marker=markerdict['combinations'], color=colordict['combinations'])
#         ax.set_ylabel('vector strength')
#
#     def plot_isi(self, key, ax):
#         eod = (EODFit() & key).fetch1['fundamental']
#         period = 1 / eod
#         baseline_spikes = (PUnitSimulations.BaselineSpikes() & key).fetch['times']
#         isi = np.hstack((np.diff(r) for r in baseline_spikes))
#         ax.hist(isi, bins=320, lw=0, color=sns.xkcd_rgb['charcoal grey'])
#         ax.set_xlim((0, 15 * period))
#         ax.set_xticks(np.arange(0, 20, 5) * period)
#         ax.set_xticklabels(np.arange(0, 20, 5))
#         ax.set_label('time [EOD cycles]')
#
#
# @schema
# class RandomTrials(dj.Lookup):
#     definition = """
#     n_total                 : int # total number of trials
#     repeat_id               : int # repeat number
#     ---
#
#     """
#
#     class TrialSet(dj.Part):
#         definition = """
#         ->RandomTrials
#         new_trial_id            : int # index of the particular trial
#         ->Runs.SpikeTimes
#         ---
#         """
#
#     class PhaseSet(dj.Part):
#         definition = """
#         ->RandomTrials
#         new_trial_id       : int   # index of the phase sample
#         ---
#         ->EFishes
#         """
#
#     def _prepare(self):
#         lens = [len(self & dict(n_total=ntot)) == 10 for ntot in (100,)]
#         n_total = 100
#         if not np.all(lens):
#             # data = (Runs() * Runs.SpikeTimes() & dict(contrast=20, cell_id="2014-12-03-ad",
#             #                                           delta_f=-300)).project().fetch.as_dict()
#             data = (Runs() * Runs.SpikeTimes() & dict(contrast=20, cell_id="2014-06-06-ak",
#                                                       delta_f=300)).project().fetch.as_dict()
#             print('Using ', len(data), 'trials')
#             data = list(sorted(data, key=lambda x: x['trial_id']))
#             n = len(data)
#
#             df = pd.DataFrame(CenteredPUnitPhases().fetch[dj.key])
#             ts = self.TrialSet()
#             ps = self.PhaseSet()
#
#             for repeat_id in range(10):
#                 key = dict(n_total=n_total, repeat_id=repeat_id)
#                 self.insert1(key, skip_duplicates=True)
#                 for new_trial_id, trial_id in enumerate(np.random.randint(n, size=n_total)):
#                     key['new_trial_id'] = new_trial_id
#                     key.update(data[trial_id])
#                     ts.insert1(key)
#
#                 key = dict(n_total=n_total, repeat_id=repeat_id)
#                 for new_trial_id, ix in enumerate(np.random.randint(len(df), size=n_total)):
#                     key['new_trial_id'] = new_trial_id
#                     key.update(df.iloc[ix].to_dict())
#                     ps.insert1(key)
#
#     def load_spikes(self, key, centered=True, plot=False):
#         if centered:
#             Phases = (RandomTrials.PhaseSet() * CenteredPUnitPhases()).project('phase', phase_cell='cell_id')
#         else:
#             Phases = (RandomTrials.PhaseSet() * UncenteredPUnitPhases()).project('phase', phase_cell='cell_id')
#         trials = Runs.SpikeTimes() * RandomTrials.TrialSet() * Phases * TrialAlign() & key
#
#         times, phase, align_times = trials.fetch['times', 'phase', 't0']
#
#         dt = 1. / (Runs() & trials).fetch1['samplingrate']
#
#         eod, duration = (Runs() & trials).fetch1['eod', 'duration']
#         rad2period = 1 / 2 / np.pi / eod
#         # get spikes, convert to s, align to EOD, add bootstrapped phase
#         print('Phase std', circ.std(phase), 'Centered', centered)
#
#         if plot:
#             figdir = 'figures/sanity/pyr_lif_stimulus/'
#             mkdir(figdir)
#             fig, ax = plt.subplots(2, 1, sharex=True)
#
#             spikes = [s / 1000 - t0 for s, t0 in zip(times, align_times)]
#             for i, s in enumerate(spikes):
#                 ax[0].plot(s, 0 * s + i, '.k', ms=1)
#             ax[0].set_title('without phase variation')
#             spikes = [s / 1000 - t0 + ph * rad2period for s, t0, ph in zip(times, align_times, phase)]
#             for i, s in enumerate(spikes):
#                 ax[1].plot(s, 0 * s + i, '.k', ms=1)
#             ax[1].set_title('with phase variation')
#             fig.savefig(figdir +
#                 'alignments_{n_total}_{pyr_simul_id}_{repeat_id}_{centered}.pdf'.format(centered=centered, **key))
#
#
#         spikes = [s / 1000 - t0 + ph * rad2period for s, t0, ph in zip(times, align_times, phase)]
#
#         return spikes, dt, eod, duration
#
#
# @schema
# class PyramidalSimulationParameters(dj.Lookup):
#     definition = """
#     pyr_simul_id    : tinyint
#     ---
#     tau_synapse     : double    # time constant of the synapse
#     tau_neuron      : double    # time constant of the lif
#     n               : int       # how many trials to simulate
#     noisesd         : double    # noise standard deviation
#     amplitude       : double    # multiplicative factor on the input
#     offset          : double    # additive factor on the input
#     threshold       : double    # LIF threshold
#     reset           : double    # reset potential
#     """
#
#     contents = [
#         dict(pyr_simul_id=0, tau_synapse=0.001, tau_neuron=0.01, n=50, noisesd=35,
#              amplitude=1.8, threshold=15, reset=0, offset=-30),
#     ]
#
#
# @schema
# class PyramidalLIF(dj.Computed):
#     definition = """
#     ->RandomTrials
#     ->PyramidalSimulationParameters
#     centered        : bool  # whether the phases got centered per fish
#     ---
#
#     """
#
#     class SpikeTimes(dj.Part):
#         definition = """
#         ->PyramidalLIF
#         simul_trial_id  :   smallint # trial number
#         ---
#         times           : longblob  # spike times in s
#         """
#
#     def _make_tuples(self, key):
#         key0 = dict(key)
#         figdir = 'figures/sanity/PyrLIF/'
#         mkdir(figdir)
#         for centered in [True, False]:
#             # load spike trains for randomly selected trials
#             data, dt, eod, duration = RandomTrials().load_spikes(key0, centered=centered)
#             key = dict(key0, centered=centered)
#             print('EOD',eod,'stim',eod-300)
#
#             eod_period = 1 / eod
#
#             # get parameters for simulation
#             params = (PyramidalSimulationParameters() & key).fetch1()
#             params.pop('pyr_simul_id')
#             print('Parameters', key)
#
#             # plot histogram of jittered data
#             fig, ax = plt.subplots()
#             ax.hist(np.hstack(data) % (1 / eod), bins=100)
#             fig.savefig(figdir + 'punitinput_{repeat_id}_{centered}.png'.format(**key))
#             plt.close(fig)
#
#             # convolve with exponential filter
#             tau_s = params.pop('tau_synapse')
#             bins = np.arange(0, duration + dt, dt)
#             t = np.arange(0, 10 * tau_s, dt)
#             h = np.exp(-np.abs(t) / tau_s)
#             trials = np.vstack([np.convolve(np.histogram(sp, bins=bins)[0], h, 'full') for sp in data])[:, :-len(h) + 1]
#
#
#             fig, ax = plt.subplots()
#             inp = trials.sum(axis=0)
#             w = np.fft.fftshift(np.fft.fftfreq(len(inp), dt))
#             a = np.fft.fftshift(np.abs(np.fft.fft(inp)))
#             idx = (w >= -1200) & (w <= 1200)
#             ax.plot(w[idx], a[idx])
#             ax.set_xticks([eod,eod+300])
#             fig.savefig(figdir + 'pyr_spectrum_{repeat_id}_{centered}.png'.format(**key))
#             plt.close(fig)
#
#             # # simulate neuron
#             # t = np.arange(0, duration, dt)
#             # ret, V = simple_lif(t, trials.sum(axis=0),
#             #                     **params)  # TODO mean would be more elegent than sum
#             # isi = [np.diff(r) for r in ret]
#             # # fig, ax = plt.subplots()
#             # # ax.hist(np.hstack(isi), bins=100)
#             # # ax.set_xticks(eod_period * np.arange(0, 50, 10))
#             # # ax.set_xticklabels(np.arange(0, 50, 10))
#             # # fig.savefig(figdir + 'pyr_isi_{repeat_id}_{centered}.png'.format(**key))
#             # # plt.close(fig)
#             #
#             # sisi = np.hstack(isi)
#             # print('Firing rates (min, max, avg)', (1 / sisi).min(), (1 / sisi).max(), np.mean([len(r) for r in ret]))
#             #
#             # self.insert1(key)
#             # st = self.SpikeTimes()
#             # for i, trial in enumerate(ret):
#             #     key['simul_trial_id'] = i
#             #     key['times'] = np.asarray(trial)
#             #     st.insert1(key)
#
#
# #
#
# @schema
# class LIFStimulusLocking(dj.Computed):
#     definition = """
#     -> PyramidalLIF                         # each run has a spectrum
#     ---
#     stimulus_frequency  : float # stimulus frequency
#     vector_strength     : float # vector strength at the stimulus frequency
#     """
#
#     def _make_tuples(self, key):
#         key = dict(key)
#         trials = PyramidalLIF.SpikeTimes() & key
#         aggregated_spikes = np.hstack(trials.fetch['times'])
#
#
#         # compute stimulus frequency
#         delta_f, eod = np.unique((Runs() * RandomTrials() * RandomTrials.TrialSet() & key).fetch['delta_f','eod'])
#         stim_freq = eod + delta_f
#         aggregated_spikes %= 1 / stim_freq
#
#         # with sns.axes_style('whitegrid'):
#         #     fig, ax = plt.subplots()
#         # for i, s in enumerate(trials.fetch['times']):
#         #     ax.scatter(s, 0*s+i)
#         # ax.set_title('centered = {centered}'.format(**key))
#         # plt.show()
#         # plt.close(fig)
#
#
#         # plt.hist(aggregated_spikes * stim_freq * 2 * np.pi, bins=100)
#         # plt.title('centered={centered}'.format(**key))
#         # plt.show()
#         key['vector_strength'] = circ.vector_strength(aggregated_spikes * stim_freq * 2 * np.pi)
#         key['stimulus_frequency'] = stim_freq
#         self.insert1(key)
#
# #
# # @schema
# # class LIFFirstOrderSpikeSpectra(dj.Computed):
# #     definition = """
# #     # table that holds 1st order vector strength spectra
# #
# #     -> PyramidalLIF                         # each run has a spectrum
# #
# #     ---
# #
# #     frequencies             : longblob # frequencies at which the spectra are computed
# #     vector_strengths        : longblob # vector strengths at those frequencies
# #     critical_value          : float    # critical value for significance with alpha=0.001
# #     """
# #
# #     def _make_tuples(self, key):
# #         print('Processing', key)
# #
# #         trials = PyramidalLIF.SpikeTimes() & key
# #         samplingrate = (Runs() & RandomTrials.TrialSet() * PyramidalLIF()).fetch1['samplingrate']
# #         aggregated_spikes = np.hstack(trials.fetch['times'])
# #
# #         key['frequencies'], key['vector_strengths'], key['critical_value'] = \
# #             compute_1st_order_spectrum(aggregated_spikes, samplingrate, alpha=0.001)
# #         vs = key['vector_strengths']
# #         vs[np.isnan(vs)] = 0
# #         self.insert1(key)
# #
# #     def plot_avg_spectrum(self, ax, centered, f_max=2000):
# #         print(self & dict(centered=centered))
# #         freqs, vs = (self & dict(centered=centered)).fetch['frequencies', 'vector_strengths']
# #         f = np.hstack(freqs)
# #         idx = np.argsort(f)
# #         f = f[idx]
# #
# #         v = [interp(f, fr, v) for fr, v in zip(freqs, vs)]
# #         # vm = np.mean(v, axis=0)
# #         # vs = np.std(v, axis=0)
# #         vm = vs[0]
# #         vs = 0*vs[0]
# #         f = freqs[0]
# #
# #         idx = (f >=0) & (f <= f_max)
# #         f = f[idx]
# #         vm = vm[idx]
# #         vs = vs[idx]
# #         ax.fill_between(f, vm-vs, vm+vs, color='silver')
# #         ax.plot(f, vm, '-k')
# #         #----------------------------------
# #         # TODO: Remove this later
# #         from IPython import embed
# #         embed()
# #         # exit()
# #         #----------------------------------
#
#
#
#
# def simple_lif(t, I, n=10, offset=0, amplitude=1, noisesd=30, threshold=15, reset=0, tau_neuron=0.01):
#     dt = t[1] - t[0]
#
#     I = amplitude * I + offset
#
#     Vout = np.ones(n) * reset
#
#     ret = [list() for _ in range(n)]
#
#     sdB = np.sqrt(dt) * noisesd
#     V = np.zeros((n, len(I)))
#     for i, t_step in enumerate(t):
#         Vout += (-Vout + I[i]) * dt / tau_neuron + np.random.randn(n) * sdB
#         idx = Vout > threshold
#         for j in np.where(idx)[0]:
#             ret[j].append(t_step)
#         Vout[idx] = reset
#         V[:, i] = Vout
#     return ret, V
#
