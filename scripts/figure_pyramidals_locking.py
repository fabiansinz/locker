import matplotlib
matplotlib.use('Agg')

from collections import OrderedDict

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
            self.fig = plt.figure(figsize=(9, 5))
            gs = plt.GridSpec(2, 3)
            self.ax = {
                'vs_freq': self.fig.add_subplot(gs[0, 0]),
                'vs_freq_beat': self.fig.add_subplot(gs[1, 0]),
            }

            self.ax['circ'] = self.fig.add_subplot(gs[0, 1])
            self.ax['contrast'] = self.fig.add_subplot(gs[0, 2])
            self.ax['circ_beat'] = self.fig.add_subplot(gs[1, 1])
            self.ax['contrast_beat'] = self.fig.add_subplot(gs[1, 2])


    @staticmethod
    def format_vs_freq(ax):
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('vector strength stimulus')
        ax.set_xlim((0, 1800))
        ax.set_ylim((0, 1))
        ax.tick_params('y', length=3, width=1)
        ax.text(-0.2, 1, 'A', transform=ax.transAxes, fontweight='bold')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.set_xticks(np.arange(0,2000,500))
        sns.despine(ax=ax, trim=True)
        # ax.legend(bbox_to_anchor=(.5,.8), frameon=False)

    @staticmethod
    def format_circ(ax):
        ax.set_ylim((0, 1))

        ax.set_xticks([0, np.pi/2, np.pi])
        ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax.set_xlabel('circular std')

        ax.set_yticks([])
        ax.text(-0.1, 1, 'B', transform=ax.transAxes, fontweight='bold')
        # ax.tick_params('y', length=0, width=0)
        ax.spines['bottom'].set_linewidth(1)
        sns.despine(ax=ax, left=True, trim=True)
        # ax.legend(loc='upper left', bbox_to_anchor=(-.1,1.1), frameon=False, ncol=3)

    @staticmethod
    def format_contrast(ax):
        # ax.get_legend().remove()
        ax.set_ylim((0, 1.0))
        ax.set_xlabel('contrast [%]')

        ax.set_ylabel('')
        ax.text(-0.1, 1, 'C', transform=ax.transAxes, fontweight='bold')
        # ax.tick_params('y', length=0, width=0)
        ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left',
                  bbox_to_anchor=(.1,1.1), frameon=False)

        ax.spines['bottom'].set_linewidth(1)
        sns.despine(ax=ax, trim=True, left=True)
        
    @staticmethod
    def format_vs_freq_beat(ax):
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('vector strength stimulus')
        ax.set_xlim((0, 600))
        ax.set_ylim((0, 1))
        ax.tick_params('y', length=3, width=1)
        ax.text(-0.2, 1, 'D', transform=ax.transAxes, fontweight='bold')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.set_xticks(np.arange(0,600,100))
        sns.despine(ax=ax, trim=True)
        # ax.legend(bbox_to_anchor=(.5,.8), frameon=False)

    @staticmethod
    def format_circ_beat(ax):
        ax.set_ylim((0, 1))

        ax.set_xticks([0, np.pi/2, np.pi])
        ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax.set_xlabel('circular std')

        ax.set_yticks([])
        ax.text(-0.1, 1, 'E', transform=ax.transAxes, fontweight='bold')
        # ax.tick_params('y', length=0, width=0)
        ax.spines['bottom'].set_linewidth(1)
        sns.despine(ax=ax, left=True, trim=True)
        # ax.legend(loc='upper left', bbox_to_anchor=(-.1,1.1), frameon=False, ncol=3)

    @staticmethod
    def format_contrast_beat(ax):
        # ax.get_legend().remove()
        ax.set_ylim((0, 1.0))
        ax.set_xlabel('contrast [%]')

        ax.set_ylabel('')
        ax.text(-0.1, 1, 'F', transform=ax.transAxes, fontweight='bold')
        # ax.tick_params('y', length=0, width=0)
        ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left',
                  bbox_to_anchor=(.1,1.1), frameon=False)
        ax.legend().set_visible(False)
        ax.spines['bottom'].set_linewidth(1)
        sns.despine(ax=ax, trim=True, left=True)

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

    with FigurePyramidals(filename='figures/figure010factor-pyramidals.pdf') as (fig, ax):
        
        rel_pu = data.Runs() * alys.FirstOrderSignificantPeaks() * alys.StimulusSpikeJitter() * data.Cells() \
                 & dict(eod_coeff=0, baseline_coeff=0, refined=1, cell_type='p-unit', am=0, n_harmonics=0) \
                 & 'stimulus_coeff = 1' \
                 & 'frequency > 0' \
        
        # exclude runs that have only one spike and, thus, artificially high locking
        rel_py = data.Runs() * alys.FirstOrderSignificantPeaks() * alys.StimulusSpikeJitter() \
                 * data.Cells() * sanity.SpikeCheck.SpikeCount() \
                 & 'spike_count > 1' \
                 & dict(eod_coeff=0, baseline_coeff=0, refined=1, am=0, n_harmonics=0) \
                 & 'stimulus_coeff = 1' \
                 & 'frequency > 0' \
                 & ['cell_type="i-cell"', 'cell_type="e-cell"']
        
        rel_pu_beat = data.Runs() * alys.FirstOrderSignificantPeaks() * alys.StimulusSpikeJitter() * data.Cells() \
                 & dict(eod_coeff=-1, baseline_coeff=0, refined=1, cell_type='p-unit', am=0, n_harmonics=0) \
                 & 'stimulus_coeff = 1' \
                 & 'frequency > 0' \
        
        rel_py_beat = data.Runs() * alys.FirstOrderSignificantPeaks() * alys.StimulusSpikeJitter() \
                 * data.Cells() * sanity.SpikeCheck.SpikeCount() \
                 & 'spike_count > 1' \
                 & dict(eod_coeff=-1, baseline_coeff=0, refined=1, am=0, n_harmonics=0) \
                 & 'stimulus_coeff = 1' \
                 & 'frequency > 0' \
                 & ['cell_type="i-cell"', 'cell_type="e-cell"']

        #====================================================================================
        print('n={0} cells tested'.format(len(data.Cells & ['cell_type="i-cell"', 'cell_type="e-cell"'])))
        print('n={0} cells locking'.format(len(data.Cells.proj() & rel_py)))

        df_pu = pd.DataFrame(rel_pu.fetch())
        df_pu['spread'] = df_pu['stim_std'] / df_pu['eod'] / 2 / np.pi
        df_pu['jitter'] = df_pu['stim_std']  # rename to avoid conflict with std function
        df_pu['cell type'] = 'p-units'

        df_py = pd.DataFrame(rel_py.fetch())
        df_py['spread'] = df_py['stim_std'] / df_py['eod'] / 2 / np.pi
        df_py['jitter'] = df_py['stim_std']  # rename to avoid conflict with std function
        df_py['cell type'] = 'pyramidal'
        
        df_pu_b = pd.DataFrame(rel_pu_beat.fetch())
        df_pu_b['spread'] = df_pu_b['stim_std'] / df_pu_b['eod'] / 2 / np.pi
        df_pu_b['jitter'] = df_pu_b['stim_std']  # rename to avoid conflict with std function
        df_pu_b['cell type'] = 'p-units'

        df_py_b = pd.DataFrame(rel_py_beat.fetch())
        df_py_b['spread'] = df_py_b['stim_std'] / df_py_b['eod'] / 2 / np.pi
        df_py_b['jitter'] = df_py_b['stim_std']  # rename to avoid conflict with std function
        df_py_b['cell type'] = 'pyramidal'
        #====================================================================================
        df = pd.concat([df_pu[df_pu.stimulus_coeff == 1], df_py[df_py.stimulus_coeff == 1]])
        for (c, ct), dat in df.groupby(['cell_id', 'cell type']):
            mu = dat.groupby('contrast').mean().reset_index()
            s = dat.groupby('contrast').std().reset_index()
            pp = sns.pointplot('contrast', 'vector_strength', data=dat, ax=ax['contrast'],
                          palette={'p-units': sns.xkcd_rgb['azure'],
                                   'pyramidal': sns.xkcd_rgb['dark fuchsia']}, scale=.4,
                          order=[10, 20], hue='cell type', alpha=1)

        df_b = pd.concat([df_pu_b[df_pu_b.stimulus_coeff == 1], df_py_b[df_py_b.stimulus_coeff == 1]])
        for (c, ct), dat in df_b.groupby(['cell_id', 'cell type']):
            mu = dat.groupby('contrast').mean().reset_index()
            s = dat.groupby('contrast').std().reset_index()
            pp = sns.pointplot('contrast', 'vector_strength', data=dat, ax=ax['contrast_beat'],
                          palette={'p-units': sns.xkcd_rgb['azure'],
                                   'pyramidal': sns.xkcd_rgb['dark fuchsia']}, scale=.4,
                          order=[10, 20], hue='cell type', alpha=1)
        #====================================================================================
        print(r"contrast: \rho={0}    p={1}".format(*stats.pearsonr(df_py.contrast, df_py.vector_strength)))
        print(r"contrast: \rho={0}    p={1}".format(*stats.pearsonr(df_py_b.contrast, df_py_b.vector_strength)))

        df_py = df_py[df_py.contrast == 20]
        df_pu = df_pu[df_pu.contrast == 20]
        
        df_py_b = df_py_b[df_py_b.contrast == 20]
        df_pu_b = df_pu_b[df_pu_b.contrast == 20]

        print(
            r'Correlation stimulus frequency and locking \rho={:.2g}, p={:.2g}'.format(
                *stats.pearsonr(df_py.eod + df_py.delta_f,
                               df_py.vector_strength)))
        print(r'Correlation jitter and locking \rho={:.2g}, p={:.2g}' \
              .format(*stats.pearsonr(df_py.jitter, df_py.vector_strength)))
        # print(r'Correlation spread and locking \rho=%.2g, p=%.2g' % \
        #       stats.pearsonr(df_py.spread, df_py.vector_strength))

        print(r'Correlation stimulus frequency and locking beat \rho={:.2g}, p={:.2g}'.format(
                *stats.pearsonr(df_py_b.eod + df_py_b.delta_f,
                               df_py_b.vector_strength)))
        print(r'Correlation jitter and locking beat \rho={:.2g}, p={:.2g}' \
              .format(*stats.pearsonr(df_py_b.jitter, df_py_b.vector_strength)))
        # print(r'Correlation spread and locking \rho=%.2g, p=%.2g' % \
        #       stats.pearsonr(df_py.spread, df_py.vector_strength))

        #====================================================================================

        point_size = 10
        ax['vs_freq'].scatter(df_pu.frequency, df_pu.vector_strength, edgecolors='w', lw=.5,
                              color=sns.xkcd_rgb['azure'], \
                              label='p-units', s=point_size)
        ax['vs_freq'].scatter(df_py.frequency, df_py.vector_strength, edgecolors='w', lw=.5,
                              color=sns.xkcd_rgb['dark fuchsia'], \
                              label='pyramidal', s=point_size)

        # --- circular variance scatter plots
        ax['circ'].scatter(df_pu.jitter, df_pu.vector_strength, edgecolors='w', lw=.5,
                           color=sns.xkcd_rgb['azure'], \
                           label='p-units', s=point_size
                           )
        ax['circ'].scatter(df_py.jitter, df_py.vector_strength, edgecolors='w', lw=.5,
                           color=sns.xkcd_rgb['dark fuchsia'], \
                           label='pyramidal', s=point_size
                           )
        
        ax['vs_freq_beat'].scatter(df_pu_b.frequency, df_pu_b.vector_strength, edgecolors='w', lw=.5,
                              color=sns.xkcd_rgb['azure'], \
                              label='p-units', s=point_size)
        ax['vs_freq_beat'].scatter(df_py_b.frequency, df_py_b.vector_strength, edgecolors='w', lw=.5,
                              color=sns.xkcd_rgb['dark fuchsia'], \
                              label='pyramidal', s=point_size)

        # --- circular variance scatter plots
        ax['circ_beat'].scatter(df_pu_b.jitter, df_pu_b.vector_strength, edgecolors='w', lw=.5,
                           color=sns.xkcd_rgb['azure'], \
                           label='p-units', s=point_size
                           )
        ax['circ_beat'].scatter(df_py_b.jitter, df_py_b.vector_strength, edgecolors='w', lw=.5,
                           color=sns.xkcd_rgb['dark fuchsia'], \
                           label='pyramidal', s=point_size
                           )


