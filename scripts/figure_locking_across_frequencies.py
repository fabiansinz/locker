import matplotlib
matplotlib.use('Agg')
from matplotlib.collections import PolyCollection
from numpy.fft import fft, fftfreq, fftshift
from locker import mkdir
from locker.analysis import *
from locker.data import *
from scripts.config import params as plot_params, FormatedFigure


def generate_filename(cell, contrast):
    dir = 'figures/figure_locking_across_frequencies/%s/' % (cell['cell_type'],)
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)


def gauss(t, m, v):
    return np.exp(-(t - m) ** 2 / 2 / v)


class FigureMechanisms(FormatedFigure):
    def prepare(self):
        sns.set_context('paper')
        sns.set_style('ticks')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(7, 5), dpi=400)
            gs = plt.GridSpec(3, 4)
            self.ax = {}
            self.ax['violin'] = self.fig.add_subplot(gs[:3, 3])
            self.ax['spectrum'] = self.fig.add_subplot(gs[:3, :3])

        self.gs = gs

    @staticmethod
    def format_spectrum(ax):
        ax.set_xlim((0, 1500))
        ax.set_xticks(np.linspace(0, 1500, 7))
        ax.legend(bbox_to_anchor=(1.05, 1), bbox_transform=ax.transAxes, ncol=3)
        sns.despine(ax=ax, left=True, trim=True, offset=0)
        ax.set_yticks([])
        ax.set_ylim((-.5, 9.5))
        ax.set_xlabel('frequency [Hz]')
        ax.text(-0.01, 0.99, 'A', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_violin(ax):
        ax.set_xlim((0, 2 * np.pi))
        ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
        ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{4}$', r'$2\pi$'])
        ax.set_ylabel(r'$\Delta f$ [Hz]')
        ax.set_xlabel('phase')
        for art in ax.get_children():
            if isinstance(art, PolyCollection):
                art.set_edgecolor(None)
        leg = ax.legend(ncol=1, title='PSTH per cycle of', bbox_to_anchor=(1, 0.97))

        plt.setp(leg.get_title(), fontsize=leg.get_texts()[0].get_fontsize())
        ax.text(-0.15, 1.01, 'B', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        sns.despine(ax=ax, trim=True, offset=0)


    def format_figure(self):
        self.ax['violin'].set_ylim([e / .8 for e in self.ax['spectrum'].get_ylim()])
        for a in self.ax.values():
            a.tick_params(length=3, width=1)
            a.spines['bottom'].set_linewidth(1)
            a.spines['left'].set_linewidth(1)

        self.gs.tight_layout(self.fig)


if __name__ == "__main__":
    f_max = 2000  # Hz
    N = 10
    delta_f = 200
    frequency_restriction = '(delta_f > -319) or (delta_f < -381)'
    runs = Runs()
    for cell in (Cells() & dict(cell_type='p-unit', cell_id="2014-12-03-aj")).fetch(as_dict=True):
        # for cell in (Cells() & dict(cell_type='p-unit')).fetch.as_dict:

        unit = cell['cell_type']
        print('Processing', cell['cell_id'])

        # for contrast in [5, 10, 20]:
        for contrast in [20]:
            print("contrast: %.2f%%" % (contrast,))

            target_trials = SecondOrderSpikeSpectra() * runs & cell & \
                            dict(contrast=contrast, am=0, n_harmonics=0) & frequency_restriction
            if target_trials:
                with FigureMechanisms(filename=generate_filename(cell, contrast=contrast)) as (fig, ax):

                    # --- plot spectra
                    y = [0]
                    stim_freq, eod_freq, deltaf_freq = [], [], []
                    done = []
                    for i, spec in enumerate(sorted(target_trials.fetch(as_dict=True), key=lambda x: x['delta_f'])):
                        if spec['delta_f'] in done:
                            continue
                        else:
                            done.append(spec['delta_f'])
                        print(u"\t\t\u0394 f=%.2f" % spec['delta_f'])

                        f, v = spec['frequencies'], spec['vector_strengths']
                        idx = (f >= 0) & (f <= f_max) & ~np.isnan(v)
                        ax['spectrum'].fill_between(f[idx], y[-1] + 0 * f[idx], y[-1] + v[idx], lw=0,
                                                    color='k')
                        if i == 0:
                            ax['spectrum'].plot([20, 20], [8., 8.5], '-', color='k', lw=2,
                                                solid_capstyle='butt')
                            ax['spectrum'].text(40, 8.15, '0.5 vector strength', fontsize=6)
                        y.append(y[-1] + .8)
                        stim_freq.append(spec['eod'] + spec['delta_f'])
                        deltaf_freq.append(spec['delta_f'])
                        eod_freq.append(spec['eod'])

                    ax['spectrum'].plot(eod_freq, y[:-1], '-', alpha=.25,  zorder=-10, lw=4, color=colordict['eod'],
                                        label='EODf')
                    ax['spectrum'].plot(stim_freq, y[:-1], '-',  alpha=.25,  zorder=-10, lw=4,
                                        color=colordict['stimulus'],
                                        label='stimulus')
                    ax['spectrum'].plot(np.abs(deltaf_freq), y[:-1], '-', alpha=.25, zorder=-10, lw=4,
                                        color=colordict['delta_f'],
                                        label=r'$|\Delta f|$')

                    # --- plot locking
                    PhaseLockingHistogram().violin_plot(ax['violin'], restrictions=target_trials.proj(),
                                                        palette=[colordict['eod'], colordict['stimulus']])
                    ax['violin'].legend().set_visible(False)

