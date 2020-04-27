import matplotlib

matplotlib.use('Agg')
import seaborn as sns
from locker import mkdir
from locker.data import ISIHistograms, BaseRate
from locker.data import *
from scripts.config import params as plot_params, FormatedFigure
import matplotlib.pyplot as plt
from locker.analysis import *
import pycircstat as circ
from matplotlib.ticker import FormatStrFormatter

def generate_filename(cell, contrast):
    dir = 'figures/figure_intro/{cell_type}/'.format(**cell)
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)

def gauss(t, m, v):
    return np.exp(-(t - m) ** 2 / 2 / v)

class FigureIntroPunit(FormatedFigure):
    def prepare(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(12, 10))
            gs = plt.GridSpec(3, 3)
            self.ax = {
                'scatter': self.fig.add_subplot(gs[-1, :]),
                'ISI': self.fig.add_subplot(gs[1, 2]),
                'EOD': self.fig.add_subplot(gs[1, 0]),
                'polar': self.fig.add_subplot(gs[1,1], projection='polar')

            }
            self.ax['scatter_base'] = self.fig.add_subplot(gs[0, :])
            self.ax['EOD_ampl'] = self.ax['EOD'].twinx()
        self.gs = gs

    @staticmethod
    def format_ISI(ax):
        sns.despine(ax=ax, left=True)
        ax.set_yticks([])
        ax.set_xticks(np.linspace(0, 15, 6))
        ax.tick_params(labelsize=10)
        ax.set_xticklabels(np.linspace(0, 15, 6).astype(int))
        ax.legend()
        ax.text(-0.1, 1.01, 'D', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_scatter_base(ax):
        sns.despine(ax=ax, left=True)
        ax.set_yticks([])
        ax.legend(ncol=4, bbox_to_anchor=(1, 1.1), loc=1, frameon=False)
        ax.text(-.07, 1.01, 'A', transform=ax.transAxes, fontweight='bold')
        ax.tick_params(labelsize=10)

    @staticmethod
    def format_EOD(ax):
        ax.text(-0.25, 1.01, 'B', transform=ax.transAxes, fontweight='bold')
        yl, yh = ax.get_ylim()
        ax.set_ylim((yl, 1.2 * yh))
        ax.set_yticks([])
        #ax.set_xlim((0, 1.2))

        ax.tick_params(axis='both', length=0, width=0, which='major', labelsize=10)
        ax.set_xlabel('Phase', fontsize=12)
        sns.despine(ax=ax, left=True, right=True, trim=True)
        ax.legend(ncol=1)
        ax.tick_params(labelsize=10)
        xl, xh = ax.get_xlim()

    @staticmethod
    def format_EOD_ampl(ax):
        ax.tick_params(axis='y', length=3, width=1, which='major')
        sns.despine(ax=ax, trim=True)
        yl, yh = ax.get_ylim()
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_ticks_position("left")
        ax.set_ylim((yl, 1.3 * yh))
        ax.set_xlim((0, 1.16))
        ax.tick_params(labelsize=10)
        

        # ax.legend(ncol=1, bbox_to_anchor=((.3,1.)))
        ax.legend(ncol=1, bbox_to_anchor=((.6, 1.)))

    @staticmethod
    def format_scatter(ax):
        sns.despine(ax=ax, left=True)
        ax.set_xlabel('time [EOD cycles]', fontsize=12)
        ax.legend(ncol=4, loc=0, bbox_to_anchor=(1, 1.1))
        ax.text(-0.07, 1.01, 'E', transform=ax.transAxes, fontweight='bold')
        ax.tick_params(labelsize=10)
        
    @staticmethod
    def format_polar(ax):
        thetaticks = np.arange(0, 360, 45)
        ax.spines['polar'].set_color(colordict['eod'])
        ax.set_thetagrids(thetaticks, frac=1.3)
        ax.set_xticklabels([r'$0$', r'$\frac{1}{8f_{EOD}}$', r'$\frac{1}{4f_{EOD}}$', r'$\frac{3}{8f_{EOD}}$', r'$\frac{1}{2f_{EOD}}$', \
                            r'$\frac{5}{8f_{EOD}}$', r'$\frac{3}{4f_{EOD}}$', r'$\frac{7}{8f_{EOD}}$'], fontsize=10)
        #ax.set_yticks([])
        ax.text(-0.07, 1.01, 'C', transform=ax.transAxes, fontweight='bold')
        ax.set_yticks([])
        ax.tick_params(labelsize=10)
        

    def format_figure(self):
        # self.fig.tight_layout()
        self.fig.subplots_adjust(left=.1, top=0.95, hspace=.3)
        # self.gs.tight_layout(self.fig)


if __name__ == "__main__":
    f_max = 2000  # Hz
    N = 50
    M = 10
    delta_f = 200
    frequency_restriction = '(delta_f > -319) or (delta_f < -381)'
    runs = Runs()
    for cell in (Cells() & dict(cell_type='p-unit', cell_id='2014-12-03-ao')).fetch("KEY"):
        unit = (Cells & cell).fetch1('cell_type')
        cell['cell_type'] = unit
        print('Processing', cell['cell_id'])
        contrast = 20

        target_trials = runs & cell & dict(contrast=contrast, am=0, n_harmonics=0, delta_f=200)
        target_trials_polar = SecondOrderSpikeSpectra() * runs & cell & \
                            dict(contrast=contrast, am=0, n_harmonics=0) & frequency_restriction
        if len(target_trials) > 0:
            with FigureIntroPunit(filename=generate_filename(cell, contrast=contrast)) as (fig, ax):
                # --- plot baseline spikes
                if Baseline() & cell:
                    (Baseline() & cell).plot_raster(ax['scatter_base'])

                # --- plot baseline psths
                if BaseRate() & cell:
                    (BaseRate() & cell).plot(ax['EOD'], ax['EOD_ampl'])

                if Baseline.SpikeTimes() & cell:
                    times = (Baseline.SpikeTimes() & cell).fetch1('times') / 1000
                    eod, sampling_rate = (Baseline().proj('eod', 'samplingrate') & cell).fetch1('eod', 'samplingrate')
                    period = 1 / eod
                    t = (times % period)
                    nu = circ.vector_strength(t / period * 2 * np.pi)
                    print('Vector strength', nu)
                    print('p-value', np.exp(-len(times) * nu ** 2))

                # --- plot ISI histogram
                ISIHistograms().plot(ax=ax['ISI'], restrictions=cell)

                EODStimulusPSTSpikes().plot_single(ax=ax['scatter'], restrictions=target_trials.fetch1("KEY"))

                # --- polar
                if BaseRate() & cell:
                    (BaseRate() & cell).plot_polar(ax['polar'])

                
                # --- plot polar 
               # eod = target_trials_polar.fetch('eod').mean()
               # delta_f = eod / 6
              #  stim_period = 1 / eod

               # print('Beat has period', eod / delta_f, 'EOD cycles')
                #var = (0.5 / 12 / eod) ** 2
              # t = np.linspace(-N / eod, N / eod, 10000)
               # t_rad = np.linspace(-5 * M, 5 * M, 10000) * 2 * np.pi
               # rad2t = 1 / 2 / np.pi * stim_period

              #  f_base = lambda t: sum(gauss(t, mu, var) for mu in np.arange(-N * 5 / eod, N * 5 / eod, 1 / eod))


               # p = f_base(t_rad * rad2t)
               # p /= p.sum()
               # x,y = (p*np.cos(t_rad)).sum(), (p*np.sin(t_rad)).sum()
               # ax['polar'].bar(t_rad, 0 * t_rad, f_base(t_rad * rad2t), color='grey', lw=0)
               # ax['polar'].plot(np.arctan2(y,x), np.sqrt(x**2 + y**2), 'ok', mfc='k', lw=0, ms=4)
            
# ----------------------------------------------------------------------------------------------------------------------------
