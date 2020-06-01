import matplotlib
matplotlib.use('Agg')

from collections import OrderedDict
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import stats

from locker import analysis as alys, colors, colordict
from locker import data
from locker import mkdir
from locker import sanity
from locker.data import Baseline
from locker.analysis import PhaseLockingHistogram
from scripts.config import params as plot_params, FormatedFigure
import pycircstat as circ


def generate_filename(cell, contrast, base='firstorderspectra'):
    dir = 'figures/%s/%s/' % (base, cell['cell_type'],)
    mkdir(dir)
    return dir + '%s_contrast%.2f.png' % (cell['cell_id'], contrast)


class FigurePyramidals(FormatedFigure):
    def __init__(self, filename=None):
        self.filename = filename

    def prepare(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(10, 7))
            gs = plt.GridSpec(2, 13)
            self.ax = {
                'spectrum': self.fig.add_subplot(gs[:2, 6:]),
                'ISI': self.fig.add_subplot(gs[0, :5]),
                'cycle': self.fig.add_subplot(gs[1, :5]),
            }
            self.ax['cycle_ampl'] = self.ax['cycle'].twinx()

    @staticmethod
    def format_ISI(ax):
        sns.despine(ax=ax, left=True, trim=True)
        ax.set_yticks([])
        ax.set_xlim((0, 15))
        ax.legend(frameon=False)
        ax.text(-0.05, 0.95, 'A', transform=ax.transAxes, fontweight='bold')
        ax.set_xticks(np.linspace(0, 15, 6))
        ax.set_xticklabels(np.linspace(0, 15, 6).astype(int))

    @staticmethod
    def format_spectrum(ax):
        ax.set_xlim((0, 1000))
        ax.tick_params('x', length=3, width=1)
        ax.spines['bottom'].set_linewidth(1)
        ax.legend(loc='upper right', ncol=3, bbox_transform=ax.transAxes, frameon=False)
        sns.despine(ax=ax, left=True, trim=True, offset=5)
        yl = ax.get_ylim()
        ax.set_yticks(np.arange(0, max(yl)-(max(yl)/17), max(yl)/17))
        ax.set_yticklabels(np.hstack((np.arange(-400,0,50), np.arange(50,450,50))))
        ax.set_ylabel(r'$\Delta f$ [Hz]', fontsize=12)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_xlabel('frequency [Hz]', fontsize=12)
        ax.text(-0.05, 0.975, 'C', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_cycle(ax):
        ax.text(-0.05, 0.9, 'B', transform=ax.transAxes, fontweight='bold')
        yl, yh = ax.get_ylim()
        ax.set_ylim((yl, 1.6 * yh))
        ax.set_yticks([])
        # ax.tick_params(axis='both', length=0, width=0, which='major')
        ax.set_xlabel('time [ms]', fontsize=12)
        sns.despine(top=True, left=True, right=True, trim=False)
        ax.legend(ncol=1, frameon=False)
    
    @staticmethod
    def format_cycle_ampl(ax):
        #ax.tick_params(axis='y', length=3, width=1, which='major')
        #ax.set_yticks(np.arange(-2,4))
        sns.despine(top=True, left=True, right=True, trim=True)
        ax.yaxis.set_visible(False)
        yl, yh = ax.get_ylim()
        ax.set_ylim((yl, 1.6 * yh))
        ax.set_yticks([])
        ax.legend(ncol=1, bbox_to_anchor=((.6, 1.)), frameon=False)

    def format_figure(self):
        fig.tight_layout()
        fig.subplots_adjust(left=0.075, right=0.95)


if __name__ == "__main__":
    f_max = 2000  # Hz
    contrast = 20
    restr = dict(cell_id='2014-11-26-ad', contrast=contrast, am=0, n_harmonics=0, refined=True)

    line_colors = colors
    # target_trials = alys.FirstOrderSpikeSpectra() * data.Runs() & restr
    target_trials = alys.FirstOrderSpikeSpectra() * data.Runs() & restr

    with FigurePyramidals(filename='figures/figure09factor-pyramidals.pdf') as (fig, ax):
        # --- plot ISI histogram
        data.ISIHistograms().plot(ax=ax['ISI'], restrictions=restr)


        if Baseline.SpikeTimes() & restr:
            times = (Baseline.SpikeTimes() & restr).fetch1('times') / 1000
            eod, sampling_rate = (Baseline() & restr).fetch1('eod', 'samplingrate')
            period = 1 / eod
            t = (times % period)
            nu = circ.vector_strength(t / period * 2 * np.pi)
            print('Vector strength', nu)
            print('p-value', np.exp(-len(times) * nu ** 2))
            # Baseline().plot_psth(ax['cycle'], restr)
        # --- plot baseline psths
        if data.BaseRate() & restr:
            (data.BaseRate() & restr).plot(ax['cycle'], ax['cycle_ampl'], find_range=False)
        #====================================================================================
        # --- plot spectra
        y = [0]
        stim_freq, eod_freq, deltaf_freq = [], [], []
        freq_log = []
        for i, spec in enumerate(sorted(target_trials.fetch(as_dict=True), key=lambda x: x['delta_f'])):
            print(u"\t\t\u0394 f=%.2f" % spec['delta_f'])

            if i == 0:
                ax['spectrum'].plot([20, 20], [12.5, 13], '-', color='darkslategray', lw=2,
                                    solid_capstyle='butt')
                ax['spectrum'].text(40, 12.65, '0.5 vector strength', fontsize=6)


            f, v = spec['frequencies'], spec['vector_strengths']
            if spec['delta_f'] in freq_log:
                continue
            else:
                freq_log.append(spec['delta_f'])
            idx = (f >= 0) & (f <= f_max) & ~np.isnan(v)
            ax['spectrum'].fill_between(f[idx], y[-1] + 0 * f[idx], y[-1] + v[idx], lw=0, color='darkslategray')

            y.append(y[-1] + .8)
            stim_freq.append(spec['eod'] + spec['delta_f'])
            deltaf_freq.append(spec['delta_f'])
            eod_freq.append(spec['eod'])

        ax['spectrum'].plot(eod_freq, y[:-1], '-',  alpha=.25,  zorder=-10, lw=4, color=colordict['eod'], label='EOD')
        ax['spectrum'].plot(stim_freq, y[:-1], '-',  alpha=.25,  zorder=-10, lw=4, color=colordict['stimulus'],
                            label='stimulus')
        ax['spectrum'].plot(np.abs(deltaf_freq), y[:-1], '-', alpha=.25,  zorder=-10, lw=4, color=colordict['delta_f'],
                            label=r'$|\Delta f|$')

        #====================================================================================
        rel_pu = data.Runs() * alys.FirstOrderSignificantPeaks() * alys.StimulusSpikeJitter() * data.Cells() \
                 & dict(eod_coeff=0, baseline_coeff=0, refined=1, cell_type='p-unit', am=0, n_harmonics=0) \
                 & 'stimulus_coeff = 1' \
                 & 'frequency > 0' \

        df_pu = pd.DataFrame(rel_pu.fetch())
        df_pu['spread'] = df_pu['stim_std'] / df_pu['eod'] / 2 / np.pi
        df_pu['jitter'] = df_pu['stim_std']  # rename to avoid conflict with std function
        df_pu['cell type'] = 'p-units'
        
        #====================================================================================
        # exclude runs that have only one spike and, thus, artificially high locking
        rel_py = data.Runs() * alys.FirstOrderSignificantPeaks() * alys.StimulusSpikeJitter() \
                 * data.Cells() * sanity.SpikeCheck.SpikeCount() \
                 & 'spike_count > 1' \
                 & dict(eod_coeff=0, baseline_coeff=0, refined=1, am=0, n_harmonics=0) \
                 & 'stimulus_coeff = 1' \
                 & 'frequency > 0' \
                 & ['cell_type="i-cell"', 'cell_type="e-cell"']
        print('n={0} cells tested'.format(len(data.Cells & ['cell_type="i-cell"', 'cell_type="e-cell"'])))
        print('n={0} cells locking'.format(len(data.Cells.proj() & rel_py)))

        df_py = pd.DataFrame(rel_py.fetch())
        df_py['spread'] = df_py['stim_std'] / df_py['eod'] / 2 / np.pi
        df_py['jitter'] = df_py['stim_std']  # rename to avoid conflict with std function
        df_py['cell type'] = 'pyramidal'


        #====================================================================================
