from itertools import count
from warnings import warn

import datajoint as dj
# import os
# import re
# from itertools import count
# from . import colordict
# import sys
# import yaml
# import seaborn as sns
import os
import numpy as np

from pyrelacs.DataClasses.RelacsFile import EmptyException
from .utils.modeling import butter_lowpass_filter
from . import colordict
from .utils.data import peakdet
from .utils.relacs import scan_info, load_traces, get_number_and_unit
from pyrelacs.DataClasses import load, TraceFile
import numpy as np
from pint import UnitRegistry
import pycircstat as circ
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pickle

ureg = UnitRegistry()
BASEDIR = '/data/'
schema = dj.schema('efish_data', locals())
LOWPASS_CUTOFF = 2000  # lowpass cutoff for peak detection


@schema
class PaperCells(dj.Lookup):
    definition = """
    # What cell ids make it into the paper

    cell_id                          : varchar(40)      # unique cell id
    ---
    """

    contents = [{'cell_id': '2014-12-03-al'},
                {'cell_id': '2014-07-23-ae'},
                {'cell_id': '2014-12-11-ae-invivo-1'},
                {'cell_id': '2014-12-11-aj-invivo-1'},
                {'cell_id': '2014-12-11-al-invivo-1'},
                {'cell_id': '2014-09-02-ad'},
                {'cell_id': '2014-12-03-ab'},
                {'cell_id': '2014-07-23-ai'},
                {'cell_id': '2014-12-03-af'},
                {'cell_id': '2014-07-23-ab'},
                {'cell_id': '2014-07-23-ah'},
                {'cell_id': '2014-11-26-ab'},
                {'cell_id': '2014-11-26-ad'},
                {'cell_id': '2014-10-29-aa'},
                {'cell_id': '2014-12-03-ae'},
                {'cell_id': '2014-09-02-af'},
                {'cell_id': '2014-12-11-ah-invivo-1'},
                {'cell_id': '2014-07-23-ad'},
                {'cell_id': '2014-12-11-ac-invivo-1'},
                {'cell_id': '2014-12-11-ab-invivo-1'},
                {'cell_id': '2014-12-11-ag-invivo-1'},
                {'cell_id': '2014-12-03-ad'},
                {'cell_id': '2014-12-11-ak-invivo-1'},
                {'cell_id': '2014-09-02-ag'},
                {'cell_id': '2014-12-03-ao'},
                {'cell_id': '2014-12-03-aj'},
                {'cell_id': '2014-07-23-aj'},
                {'cell_id': '2014-11-26-ac'},
                {'cell_id': '2014-12-03-ai'},
                {'cell_id': '2014-06-06-ak'},
                {'cell_id': '2014-11-13-ab'},
                {'cell_id': '2014-05-21-ab'},
                {'cell_id': '2014-07-23-ag'},
                {'cell_id': '2014-12-03-ah'},
                {'cell_id': '2014-07-23-aa'},
                {'cell_id': '2014-12-11-am-invivo-1'},
                {'cell_id': '2014-12-11-aa-invivo-1'},
                {'cell_id': '2017-08-15-ag-invivo-1'},
                {'cell_id': '2017-08-15-ah-invivo-1'},
                {'cell_id': '2017-08-15-ai-invivo-1'},
                {'cell_id': '2017-10-25-ad-invivo-1'},  # bursty
                {'cell_id': '2017-10-25-ae-invivo-1'},  # bursty
                {'cell_id': '2017-10-25-aj-invivo-1'},
                {'cell_id': '2017-10-25-an-invivo-1'},
                # --- pyramidal cells
                {'cell_id': '2017-11-08-aa-invivo-1'},
                {'cell_id': '2017-11-10-aa-invivo-1'},
                {'cell_id': '2017-08-11-ae-invivo-1'},
                {'cell_id': '2017-08-17-ae-invivo-1'},
                # --- new cells
                {'cell_id': '2018-05-08-aa-invivo-1'},
                {'cell_id': '2018-05-08-ab-invivo-1'},
                {'cell_id': '2018-05-08-ac-invivo-1'},
                {'cell_id': '2018-05-08-ad-invivo-1'},
                {'cell_id': '2018-05-08-ae-invivo-1'},
                {'cell_id': '2018-05-08-af-invivo-1'},
                {'cell_id': '2018-05-08-ag-invivo-1'},
                {'cell_id': '2018-05-08-ah-invivo-1'},
                {'cell_id': '2018-05-08-ai-invivo-1'},
                # --- more from 2017
                {'cell_id': '2017-08-11-aa-invivo-1'},
                {'cell_id': '2017-08-11-ab-invivo-1'},
                {'cell_id': '2017-08-11-ac-invivo-1'},
                {'cell_id': '2017-08-11-ad-invivo-1'},
                {'cell_id': '2017-10-25-ad-invivo-1'},
                {'cell_id': '2017-10-25-ae-invivo-1'},
                # {'cell_id': '2017-10-25-aq-invivo-1'},
                # --- new from 2018
                {'cell_id': '2018-06-26-aa-invivo-1'},
                {'cell_id': '2018-06-26-ak-invivo-1'},
                ]


@schema
class EFishes(dj.Imported):
    definition = """
    # Basics weakly electric fish subject info

    fish_id                                 : int # fish id in the animals fish database          
    -> PaperCells
    ---

    eod_frequency                           : float # EOD frequency in Hz
    species = "Apteronotus leptorhynchus"   : enum('Apteronotus leptorhynchus', 'Eigenmannia virescens') # species
    gender  = "unknown"                     : enum('unknown', 'make', 'female') # gender
    weight                                  : float # weight in g
    size                                    : float  # size in cm
    """

    @property
    def key_source(self):
        return PaperCells()

    def _make_tuples(self, key):
        # we are not making this a direct dependency because that database is not Datajoint compatible
        animals = dj.create_virtual_module('census', 'animal_keeping')

        a = scan_info(key['cell_id'], basedir=BASEDIR)
        a = a['Subject'] if 'Subject' in a else a['Recording']['Subject']

        if not animals.CensusSubject() & dict(name=a['Identifier']):
            if 'zucht' in a['Identifier']:
                tmp = a['Identifier'].split('leptozucht')
                tmp[-1] = '{:04d}'.format(int(tmp[-1]))
                a['Identifier'] = 'leptozucht'.join(tmp)
            else:
                tmp = a['Identifier'].split('lepto')
                tmp[-1] = '{:04d}'.format(int(tmp[-1]))
                a['Identifier'] = 'lepto'.join(tmp)
        fish_id = (animals.CensusSubject() & dict(name=a['Identifier'])).fetch1('id')

        self.insert1(dict(key,
                          fish_id=fish_id,
                          eod_frequency=float(a['EOD Frequency'][:-2]),
                          gender=a['Gender'].lower(),
                          weight=float(a['Weight'][:-1]),
                          size=float(a['Size'][:-2])))


@schema
class Cells(dj.Imported):
    definition = """
    # Recorded cell with additional info

    ->PaperCells                                # cell id to be imported
    ---
    ->EFishes                                   # fish this cell was recorded from
    recording_date                   : date     #  recording date
    cell_type                        : enum('p-unit', 'i-cell', 'e-cell','ampullary','ovoid','unkown') # cell type
    recording_location               : enum('nerve', 'ell') # where
    depth                            : float    # recording depth in mu
    baseline                         : float    # baseline firing rate in Hz
    """

    def _make_tuples(self, key):
        a = scan_info(key['cell_id'], basedir=BASEDIR)
        subj = a['Subject'] if 'Subject' in a else a['Recording']['Subject']
        cl = a['Cell'] if 'Cell' in a else a['Recording']['Cell']
        # fish_id = EFishes() & #subj['Identifier'],
        dat = {'cell_id': key['cell_id'],
               'fish_id': (EFishes() & key).fetch1('fish_id'),
               'recording_date': a['Recording']['Date'],
               'cell_type': cl['CellType'].lower(),
               'recording_location': 'nerve' if cl['Structure'].lower() == 'nerve' else 'ell',
               'depth': float(cl['Depth'][:-2]),
               'baseline': float(cl['Cell properties']['Firing Rate1'][:-2])
               }
        self.insert1(dat)


@schema
class FICurves(dj.Imported):
    definition = """
    # FI  Curves from recorded cells

    ->Cells                             # cell ids
    block_no            : int           # id of the fi curve block

    ---

    inx                 : longblob      # index
    n                   : longblob      # no of repeats
    ir                  : longblob      # Ir in mV
    im                  : longblob      # Im in mV
    f_0                 : longblob      # f0 in Hz
    f_s                 : longblob      # fs in Hz
    f_r                 : longblob      # fr in Hz
    ip                  : longblob      # Ip in mV
    ipm                 : longblob      # Ipm in mV
    f_p                 : longblob      # fp in Hz
    """

    def _make_tuples(self, key):
        filename = BASEDIR + key['cell_id'] + '/ficurves1.dat'
        if os.path.isfile(filename):
            fi = load(filename)

            for i, (fi_meta, fi_key, fi_data) in enumerate(zip(*fi.selectall())):

                fi_data = np.asarray(fi_data).T
                row = {'block_no': i, 'cell_id': key['cell_id']}
                for (name, _), dat in zip(fi_key, fi_data):
                    row[name.lower()] = dat

                self.insert1(row)

    def plot(self, ax, restrictions):
        rel = self & restrictions
        try:
            contrast, f0, fs = (self & restrictions).fetch1('ir', 'f_0', 'f_s')
        except dj.DataJointError:
            return
        ax.plot(contrast, f0, '--k', label='onset response', dashes=(2, 2))
        ax.plot(contrast, fs, '-k', label='stationary response')
        ax.set_xlabel('amplitude [mV/cm]')
        ax.set_ylabel('firing rate [Hz]')
        _, ymax = ax.get_ylim()
        ax.set_ylim((0, 1.5 * ymax))
        mi, ma = np.amin(contrast), np.amax(contrast)
        ax.set_xticks(np.round([mi, (ma + mi) * .5, ma], decimals=1))


@schema
class ISIHistograms(dj.Imported):
    definition = """
    # ISI Histograms

    block_no            : int           # id of the isi curve block
    ->Cells                             # cell ids

    ---

    t                   : longblob      # time
    n                   : longblob      # no of repeats
    eod                 : longblob      # time in eod cycles
    p                   : longblob      # histogram
    """

    def _make_tuples(self, key):
        filename = BASEDIR + key['cell_id'] + '/baseisih1.dat'
        if os.path.isfile(filename):
            fi = load(filename)

            for i, (fi_meta, fi_key, fi_data) in enumerate(zip(*fi.selectall())):

                fi_data = np.asarray(fi_data).T
                row = {'block_no': i, 'cell_id': key['cell_id']}
                for (name, _), dat in zip(fi_key, fi_data):
                    row[name.lower()] = dat

                self.insert1(row)

    def plot(self, ax, restrictions):
        try:
            eod_cycles, p = (ISIHistograms() & restrictions).fetch1('eod', 'p')
        except dj.DataJointError:
            return
        dt = eod_cycles[1] - eod_cycles[0]
        idx = eod_cycles <= 15
        ax.bar(eod_cycles[idx], p[idx], width=dt, color='gray', lw=0, zorder=-10, label='ISI histogram')
        ax.set_xlabel('EOD cycles')


@schema
class BaseEOD(dj.Imported):
    definition = """
        # table holding baseline rate with EOD
        ->Cells
        ---
        eod                     : float # eod rate at trial in Hz
        eod_period              : float # eod period in ms
        firing_rate             : float # firing rate of the cell
        time                    : longblob # sampling bins in ms
        eod_ampl                : longblob # corresponding EOD amplitude
        min_idx                 : longblob # index into minima of eod amplitude
        max_idx                 : longblob # index into maxima of eod amplitude
        """

    def _make_tuples(self, key):
        print('Populating', key)
        basedir = BASEDIR + key['cell_id']
        filename = basedir + '/baseeodtrace.dat'
        if os.path.isfile(filename):
            rate = TraceFile(filename)
        else:
            print('No such file', filename, 'skipping. ')
            return
        info, _, data = [e[0] for e in rate.selectall()]

        key['eod'] = float(info['EOD rate'][:-2])
        key['eod_period'] = float(info['EOD period'][:-2])
        key['firing_rate'] = float(info['firing frequency1'][:-2])

        key['time'], key['eod_ampl'] = data.T
        sample_freq = 1 / np.median(np.diff(key['time']))
        if sample_freq > 2 * LOWPASS_CUTOFF:
            print('\tLowpass filter to 2000Hz')
            dat = butter_lowpass_filter(data[:, 1], highcut=LOWPASS_CUTOFF, fs=sample_freq, order=5)
        else:
            dat = data[:, 1]
        _, key['max_idx'], _, key['min_idx'] = peakdet(dat)
        self.insert1(key)


@schema
class Baseline(dj.Imported):
    definition = """
    # table holding baseline recordings
    ->Cells
    repeat                  : int # index of the run

    ---
    eod                     : float # eod rate at trial in Hz
    duration                : float # duration in s
    samplingrate            : float # sampling rate in Hz

    """

    class SpikeTimes(dj.Part):
        definition = """
        # table holding spike time of trials

        -> Baseline
        ---

        times                      : longblob # spikes times in ms
        """

    class LocalEODPeaksTroughs(dj.Part, dj.Manual):
        definition = """
        # table holding local EOD traces

        -> Baseline
        ---

        peaks                      : longblob
        troughs                      : longblob
        """

    def mean_var(self, restrictions):
        """
        Computes the mean and variance of the baseline psth
        :param restrictions: restriction that identify one baseline trial
        :return: mean and variance
        """
        rel = self & restrictions
        spikes = (Baseline.SpikeTimes() & rel).fetch1('times')
        eod = rel.fetch1('eod')
        period = 1 / eod
        factor = 2 * np.pi / period
        t = (spikes % period)
        mu = circ.mean(t * factor) / factor
        sigma2 = circ.var(t * factor) / factor ** 2
        return mu, sigma2

    def plot_psth(self, ax, restrictions):
        minimum_rep = (Baseline.SpikeTimes() & restrictions).fetch['repeat'].min()
        spikes = (Baseline.SpikeTimes() & restrictions & dict(repeat=minimum_rep)).fetch1(
            'times') / 1000  # convert to s

        eod, sampling_rate = (self & restrictions).fetch1('eod', 'samplingrate')
        if (Baseline.LocalEODPeaksTroughs() & restrictions):
            spikes -= (Baseline.LocalEODPeaksTroughs() & restrictions).fetch1('peaks')[0] / sampling_rate

        period = 1 / eod
        t = (spikes % period)
        nu = circ.vector_strength(t / period * 2 * np.pi)
        print('Vector strength', nu, 'p-value', np.exp(-nu ** 2 * len(t)))
        ax.hist(t, bins=50, color='silver', lw=0, normed=True)
        ax.set_xlim((0, period))
        ax.set_xlabel('EOD cycle', labelpad=-5)
        ax.set_xticks([0, period])
        ax.set_xticklabels([0, 1])
        # ax.set_ylabel('PSTH')
        ax.set_yticks([])

    def plot_raster(self, ax, cycles=21, repeats=20):
        sampl_rate, duration, eod = self.fetch1('samplingrate', 'duration', 'eod')
        peaks, spikes = (self * self.SpikeTimes() * self.LocalEODPeaksTroughs()).fetch1('peaks', 'times')
        spikes = spikes / 1000  # convert to s
        pt = peaks / sampl_rate
        spikes, pt = spikes - pt[0], pt - pt[0]
        dt = (cycles // 2) / eod

        spikes = [spikes[(spikes >= t - dt) & (spikes < t + dt)] - t for t in pt[cycles // 2::cycles]]

        # histogram
        db = 1 / eod

        bins = np.arange(-(cycles // 2) / eod, (cycles // 2) / eod + db, db)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        h, _ = np.histogram(np.hstack(spikes), bins=bins)

        h = h.astype(np.float64)
        f_max = h.max() / db / len(spikes)

        h *= repeats / h.max() / 2
        ax.bar(bin_centers, h, align='center', width=db, color='lightgray', zorder=-20, lw=0, label='PSTH')
        ax.plot(bin_centers[0] * np.ones(2), [repeats // 8, h.max() * 150 / f_max + repeats // 8], '-',
                color='darkslategray',
                lw=3, solid_capstyle='butt')
        ax.text(bin_centers[0] + db / 4, repeats / 6, '150 Hz')
        for y, sp in zip(count(start=repeats // 2 + 1), spikes[:min(repeats, len(spikes))]):
            ax.vlines(sp, 0 * sp + y, 0 * sp + y + 1, 'k', rasterized=False,
                      label='spikes' if y == repeats // 2 + 1 else None)
            # ax.plot(sp, 0 * sp + y, '.k', mfc='k', ms=2, zorder=-10, rasterized=False)
        y += 1
        ax.set_xticks(np.arange(-(cycles // 2) / eod, (cycles // 2 + 1) / eod, 5 / eod))
        ax.set_xticklabels(np.arange(-(cycles // 2), cycles // 2 + 1, 5))
        ax.set_xlim((-(cycles // 2) / eod, (cycles // 2) / eod))
        ax.set_xlabel('time [EOD cycles]')

        # EOD
        if BaseEOD() & self.proj():
            t, e, pe = (BaseEOD() & self.proj()).fetch1('time', 'eod_ampl', 'max_idx')
            t = t / 1000
            pe_t = t[pe]
            t = t - pe_t[cycles // 2]
            fr, to = pe[0], pe[cycles]
            t, e = t[fr:to], e[fr:to]
            e = self.clean_signal(e, eod, t[1] - t[0])

            e = (e - e.min()) / (e.max() - e.min()) * repeats / 2

            ax.plot(t, e + y, lw=2, color=colordict['eod'], zorder=-15, label='EOD')

    @staticmethod
    def clean_signal(s, eod, dt, tol=3):
        f = np.fft.fft(s)
        w = np.fft.fftfreq(len(s), d=dt)

        idx = (w > -2000) & (w < 2000)
        f[(w % eod > tol) & (w % eod < eod - tol)] = 0
        return np.fft.ifft(f).real

    def _make_tuples(self, key):
        print(key)
        repro = 'BaselineActivity'
        basedir = BASEDIR + key['cell_id']
        spikefile = basedir + '/basespikes1.dat'
        if os.path.isfile(spikefile):
            stimuli = load(basedir + '/stimuli.dat')

            traces = load_traces(basedir, stimuli)
            spikes = load(spikefile)
            spi_meta, spi_key, spi_data = spikes.selectall()

            localeod = Baseline.LocalEODPeaksTroughs()
            spike_table = Baseline.SpikeTimes()

            for run_idx, (spi_d, spi_m) in enumerate(zip(spi_data, spi_meta)):
                print("\t%s repeat %i" % (repro, run_idx))

                # match index from stimspikes with run from stimuli.dat
                stim_m, stim_k, stim_d = stimuli.subkey_select(RePro=repro, Run=spi_m['index'])

                if len(stim_m) > 1:
                    raise KeyError('%s and index are not unique to identify stimuli.dat block.' % (repro,))
                else:
                    stim_k = stim_k[0]
                    stim_m = stim_m[0]
                    signal_column = \
                        [i for i, k in enumerate(stim_k) if k[:4] == ('stimulus', 'GlobalEField', 'signal', '-')][0]

                    valid = []

                    if stim_d == [[[0]]]:
                        print("\t\tEmpty stimuli data! Continuing ...")
                        continue

                    for d in stim_d[0]:
                        if not d[signal_column].startswith('FileStimulus-value'):
                            valid.append(d)
                        else:
                            print("\t\tExcluding a reset trial from stimuli.dat")
                    stim_d = valid

                if len(stim_d) > 1:
                    print(
                        """\t\t%s index %i has more one trials. Not including data.""" % (
                            spikefile, spi_m['index'], len(spi_d), len(stim_d)))
                    continue

                start_index, index = [(i, k[-1]) for i, k in enumerate(stim_k) if 'traces' in k and 'V-1' in k][0]
                sample_interval, time_unit = get_number_and_unit(
                    stim_m['analog input traces']['sample interval%i' % (index,)])

                # make sure that everything was sampled with the same interval
                sis = []
                for jj in range(1, 5):
                    si, tu = get_number_and_unit(stim_m['analog input traces']['sample interval%i' % (jj,)])
                    assert tu == 'ms', 'Time unit is not ms anymore!'
                    sis.append(si)
                assert len(np.unique(sis)) == 1, 'Different sampling intervals!'

                duration = ureg.parse_expression(spi_m['duration']).to(time_unit).magnitude

                start_idx, stop_idx = [], []
                # start_times, stop_times = [], []

                start_indices = [d[start_index] for d in stim_d]
                for begin_index, trial in zip(start_indices, spi_d):
                    start_idx.append(begin_index)
                    stop_idx.append(begin_index + duration / sample_interval)

                to_insert = dict(key)
                to_insert['repeat'] = spi_m['index']
                to_insert['eod'] = float(spi_m['EOD rate'][:-2])
                to_insert['duration'] = duration / 1000 if time_unit == 'ms' else duration
                to_insert['samplingrate'] = 1 / sample_interval * 1000 if time_unit == 'ms' else 1 / sample_interval

                self.insert1(to_insert)

                for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
                    if start > 0:
                        tmp = dict(key, repeat=spi_m['index'])
                        leod = traces['LocalEOD-1']['data'][start:stop]
                        if to_insert['samplingrate'] > 2 * LOWPASS_CUTOFF:
                            print('\tLowpass filter to', LOWPASS_CUTOFF, 'Hz for peak detection')
                            leod = butter_lowpass_filter(leod, highcut=LOWPASS_CUTOFF,
                                                         fs=to_insert['samplingrate'],
                                                         order=5
                                                         )
                        _, tmp['peaks'], _, tmp['troughs'] = peakdet(leod)
                        localeod.insert1(tmp, replace=True)
                    else:
                        print("Negative indices in stimuli.dat. Skipping local peak extraction!")

                    spike_table.insert1(dict(key, times=spi_d, repeat=spi_m['index']), replace=True)


@schema
class BaseRate(dj.Imported):
    definition = """
        # table holding baseline rate with EOD
        ->Cells
        ---
        eod                     : float # eod rate at trial in Hz
        eod_period              : float # eod period in ms
        firing_rate             : float # firing rate of the cell
        time                    : longblob # sampling bins in ms
        eod_rate                : longblob # instantaneous rate
        eod_ampl                : longblob # corresponding EOD amplitude
        min_idx                 : longblob # index into minima of eod amplitude
        max_idx                 : longblob # index into maxima of eod amplitude
        """

    def _make_tuples(self, key):
        print('Populating', key)
        basedir = BASEDIR + key['cell_id']
        filename = basedir + '/baserate1.dat'
        if os.path.isfile(filename):
            rate = TraceFile(filename)
        else:
            print('No such file', filename, 'skipping. ')
            return
        info, _, data = [e[0] for e in rate.selectall()]

        key['eod'] = float(info['EOD rate'][:-2])
        key['eod_period'] = float(info['EOD period'][:-2])
        key['firing_rate'] = float(info['firing frequency1'][:-2])
        key['time'], key['eod_rate'], key['eod_ampl'] = data.T

        sample_freq = 1 / np.median(np.diff(key['time']))
        if sample_freq > 2 * LOWPASS_CUTOFF:
            print('\tLowpass filter to 2000Hz')
            dat = butter_lowpass_filter(data[:, 2], highcut=LOWPASS_CUTOFF, fs=sample_freq, order=5)
        else:
            dat = data[:, 2]
        _, key['max_idx'], _, key['min_idx'] = peakdet(dat)
        self.insert1(key)

    def plot(self, ax, ax2, find_range=True):
        t, rate, ampl, mi, ma = self.fetch1('time', 'eod_rate', 'eod_ampl', 'min_idx', 'max_idx')
        n = len(t)
        if find_range:
            if len(mi) < 2:
                if mi[0] < n // 2:
                    mi = np.hstack((mi, [n]))
                else:
                    mi = np.hstack(([0], mi + 1))

            idx = slice(*mi)
        else:
            idx = slice(None)
        dt = t[1] - t[0]
        t = t - t[mi[0]]
        ax2.plot(t[idx], ampl[idx], color=colordict['eod'], label='EOD', zorder=10, lw=2)
        ax.bar(t[idx], rate[idx], color='lightgray', lw=0, width=dt, align='center', label='PSTH', zorder=-10)
        # ax.set_ylabel('firing rate [Hz]')
        ax2.set_ylabel('EOD amplitude [mV]')
        ax.axis('tight')
        ax2.axis('tight')
        ax.set_xlabel('time [ms]')


@schema
class Runs(dj.Imported):
    definition = """
    # table holding trials

    run_id                     : int # index of the run
    repro="SAM"                : enum('SAM', 'Filestimulus')
    ->Cells                    # which cell the trial belongs to

    ---

    delta_f                 : float # delta f of the trial in Hz
    contrast                : float  # contrast of the trial
    eod                     : float # eod rate at trial in Hz
    duration                : float # duration in s
    am                      : int   # whether AM was used
    samplingrate            : float # sampling rate in Hz
    n_harmonics             : int # number of harmonics in the stimulus
    """

    class SpikeTimes(dj.Part):
        definition = """
        # table holding spike time of trials

        -> master
        trial_id                   : int # index of the trial within run

        ---

        times                      : longblob # spikes times in ms
        """

    class GlobalEFieldPeaksTroughs(dj.Part):
        definition = """
        # table holding global efield trace

        -> master
        trial_id                   : int # index of the trial within run
        ---

        peaks               : longblob # peak indices
        troughs             : longblob # trough indices
        """

    class LocalEODPeaksTroughs(dj.Part):
        definition = """
        # table holding local EOD traces

        -> master
        trial_id                   : int # index of the trial within run
        ---

        peaks               : longblob # peak indices
        troughs             : longblob # trough indices
        """

    class LocalEOD(dj.Part):
        definition = """
        # table holding local EOD traces

        -> master
        trial_id                   : int # index of the trial within run
        ---

        local_efield             : longblob # peak indices
        """

    class GlobalEOD(dj.Part):
        definition = """
        # table holding global EOD traces

        -> Runs
        trial_id                   : int # index of the trial within run
        ---

        global_voltage                      : longblob # spikes times
        """

    class GlobalEODPeaksTroughs(dj.Part):
        definition = """
        # table holding global EOD traces

        -> master.GlobalEOD
        ---

        peaks               : longblob # peak indices
        troughs             : longblob # trough indices
        """

    def load_spikes(self):
        """
        Loads all spikes referring to that relation.

        :return: trial ids and spike times in s
        """
        spike_times, trial_ids = (Runs.SpikeTimes() & self).fetch['times', 'trial_id']
        spike_times = [s / 1000 for s in spike_times]  # convert to s
        return trial_ids, spike_times

    def check_run_indices(self):
        for key in self.key_source.fetch('KEY'):
            repro = 'SAM'
            basedir = BASEDIR + key['cell_id']
            spikefile = basedir + '/samallspikes1.dat'
            if os.path.isfile(spikefile):
                stimuli = load(basedir + '/stimuli.dat')
                traces = load_traces(basedir, stimuli)
                spikes = load(spikefile)
                spi_meta, spi_key, spi_data = spikes.selectall()

                globalefield = Runs.GlobalEFieldPeaksTroughs()
                localeodpeaks = Runs.LocalEODPeaksTroughs()
                localeod = Runs.LocalEOD()
                globaleod = Runs.GlobalEOD()
                globaleodpeaks = Runs.GlobalEODPeaksTroughs()
                spike_table = Runs.SpikeTimes()
                # v1trace = Runs.VoltageTraces()

                for run_idx, (spi_d, spi_m) in enumerate(zip(spi_data, spi_meta)):
                    if run_idx != spi_m['index']:
                        print('Indices do not match for key', key)
                    else:
                        print('.', end='', flush=True)

    def _make_tuples(self, key):
        repro = 'SAM'
        basedir = BASEDIR + key['cell_id']
        spikefile = basedir + '/samallspikes1.dat'
        if os.path.isfile(spikefile):
            stimuli = load(basedir + '/stimuli.dat')
            traces = load_traces(basedir, stimuli)
            spikes = load(spikefile)
            spi_meta, spi_key, spi_data = spikes.selectall()
            globalefield = Runs.GlobalEFieldPeaksTroughs()
            localeodpeaks = Runs.LocalEODPeaksTroughs()
            localeod = Runs.LocalEOD()
            globaleod = Runs.GlobalEOD()
            globaleodpeaks = Runs.GlobalEODPeaksTroughs()
            spike_table = Runs.SpikeTimes()
            # v1trace = Runs.VoltageTraces()

            for (spi_d, spi_m) in zip(spi_data, spi_meta):
                run_idx = spi_m['index']

                print("\t%s run %i" % (repro, run_idx))

                # match index from stimspikes with run from stimuli.dat
                try:
                    stim_m, stim_k, stim_d = stimuli.subkey_select(RePro=repro, Run=spi_m['index'])
                except EmptyException:
                    warn('Empty stimuli for ' + repr(spi_m))
                    continue

                if len(stim_m) > 1:
                    raise KeyError('%s and index are not unique to identify stimuli.dat block.' % (repro,))
                else:
                    stim_k = stim_k[0]
                    stim_m = stim_m[0]
                    signal_column = \
                        [i for i, k in enumerate(stim_k) if k[:4] == ('stimulus', 'GlobalEField', 'signal', '-')][0]

                    valid = []

                    if stim_d == [[[0]]]:
                        print("\t\tEmpty stimuli data! Continuing ...")
                        continue

                    for d in stim_d[0]:
                        if not d[signal_column].startswith('FileStimulus-value'):
                            valid.append(d)
                        else:
                            print("\t\tExcluding a reset trial from stimuli.dat")
                    stim_d = valid

                if len(stim_d) != len(spi_d):
                    print(
                        """\t\t%s index %i has %i trials, but stimuli.dat has %i. Trial was probably aborted. Not including data.""" % (
                            spikefile, spi_m['index'], len(spi_d), len(stim_d)))
                    continue

                start_index, index = [(i, k[-1]) for i, k in enumerate(stim_k) if 'traces' in k and 'V-1' in k][0]
                sample_interval, time_unit = get_number_and_unit(
                    stim_m['analog input traces']['sample interval%i' % (index,)])

                # make sure that everything was sampled with the same interval
                sis = []
                for jj in range(1, 5):
                    si, tu = get_number_and_unit(stim_m['analog input traces']['sample interval%i' % (jj,)])
                    assert tu == 'ms', 'Time unit is not ms anymore!'
                    sis.append(si)
                assert len(np.unique(sis)) == 1, 'Different sampling intervals!'

                duration = ureg.parse_expression(spi_m['Settings']['Stimulus']['duration']).to(time_unit).magnitude

                if 'ampl' in spi_m['Settings']['Stimulus']:
                    harmonics = np.array(list(map(float, spi_m['Settings']['Stimulus']['ampl'].strip().split(','))))
                    if np.all(harmonics == 0):
                        nharmonics = 0
                    else:
                        nharmonics = len(harmonics)
                else:
                    nharmonics = 0

                start_idx, stop_idx = [], []
                # start_times, stop_times = [], []

                start_indices = [d[start_index] for d in stim_d]
                for begin_index, trial in zip(start_indices, spi_d):
                    # start_times.append(begin_index*sample_interval)
                    # stop_times.append(begin_index*sample_interval + duration)
                    start_idx.append(begin_index)
                    stop_idx.append(begin_index + duration / sample_interval)

                to_insert = dict(key)
                to_insert['run_id'] = spi_m['index']
                to_insert['delta_f'] = float(spi_m['Settings']['Stimulus']['deltaf'][:-2])
                to_insert['contrast'] = float(spi_m['Settings']['Stimulus']['contrast'][:-1])
                to_insert['eod'] = float(spi_m['EOD rate'][:-2])
                to_insert['duration'] = duration / 1000 if time_unit == 'ms' else duration
                to_insert['am'] = spi_m['Settings']['Stimulus']['am'] * 1
                to_insert['samplingrate'] = 1 / sample_interval * 1000 if time_unit == 'ms' else 1 / sample_interval
                fs = to_insert['samplingrate']
                to_insert['n_harmonics'] = nharmonics
                to_insert['repro'] = 'SAM'

                self.insert1(to_insert)
                for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
                    tmp = dict(run_id=run_idx, trial_id=trial_idx, repro='SAM', **key)
                    # tmp['membrane_potential'] = traces['V-1']['data'][start:stop]
                    # # v1trace.insert1(tmp, replace=True)
                    # del tmp['membrane_potential']
                    # ---
                    global_efield = traces['GlobalEFie']['data'][start:stop].astype(np.float32)
                    if fs > 2 * LOWPASS_CUTOFF:
                        global_efield = butter_lowpass_filter(global_efield, highcut=LOWPASS_CUTOFF,
                                                              fs=fs, order=5)
                    _, peaks, _, troughs = peakdet(global_efield)
                    globalefield.insert1(dict(tmp, peaks=peaks, troughs=troughs), ignore_extra_fields=True)

                    # ---
                    local_efield = traces['LocalEOD-1']['data'][start:stop].astype(np.float32)
                    localeod.insert1(dict(tmp, local_efield=local_efield), ignore_extra_fields=True)
                    if fs > 2 * LOWPASS_CUTOFF:
                        local_efield = butter_lowpass_filter(local_efield, highcut=LOWPASS_CUTOFF,
                                                             fs=fs, order=5)
                    _, peaks, _, troughs = peakdet(local_efield)
                    localeodpeaks.insert1(dict(tmp, peaks=peaks, troughs=troughs), ignore_extra_fields=True)

                    # ---
                    global_voltage = traces['EOD']['data'][start:stop].astype(np.float32)
                    globaleod.insert1(dict(tmp, global_voltage=global_voltage),
                                      ignore_extra_fields=True)
                    if fs > 2 * LOWPASS_CUTOFF:
                        global_voltage = butter_lowpass_filter(global_voltage, highcut=LOWPASS_CUTOFF,
                                                               fs=fs, order=5)
                    _, peaks, _, troughs = peakdet(global_voltage)
                    globaleodpeaks.insert1(dict(tmp, peaks=peaks, troughs=troughs),
                                           ignore_extra_fields=True)

                    # ---
                    tmp['times'] = spi_d[trial_idx]
                    spike_table.insert1(tmp, ignore_extra_fields=True)

    # def post_fill(self):
    #
    #     for k in self.GlobalEOD().fetch.keys():
    #         global_voltage = (self.GlobalEOD() & k).fetch1('global_voltage')
    #         print(k, flush=True)
    #         _, peaks, _, troughs = peakdet(global_voltage)
    #         Runs.GlobalEODPeaksTroughs().insert1(dict(k, peaks=peaks, troughs=troughs))
