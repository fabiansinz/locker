import matplotlib
matplotlib.use('Agg')

from numpy.fft import fft, fftfreq, fftshift
from locker import mkdir
from locker.analysis import *
from locker.data import *
from scripts.config import params as plot_params, FormatedFigure


def generate_filename(cell, contrast):
    dir = 'figures/figure_mechanisms/%s/' % (cell['cell_type'],)
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)


def gauss(t, m, v):
    return np.exp(-(t - m) ** 2 / 2 / v)


class FigureMechanisms(FormatedFigure):
    def prepare(self):
        sns.set_context('paper')
        sns.set_style('ticks')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(7, 4), dpi=400)
            gs = plt.GridSpec(2, 4)
            self.ax = {}
            self.ax['cartoon_psth'] = self.fig.add_subplot(gs[0, :2])
            self.ax['cartoon_psth_stim'] = self.fig.add_subplot(gs[1, :2])

            self.ax['spectrum_base'] = self.fig.add_subplot(gs[0, 3])
            with sns.axes_style('white'):
                self.ax['polar_base'] = self.fig.add_subplot(gs[0, 2], projection='polar')
                self.ax['polar_stim'] = self.fig.add_subplot(gs[1, 2], projection='polar')
            self.ax['spectrum_stim'] = self.fig.add_subplot(gs[1, 3])
        self.gs = gs

    @staticmethod
    def format_cartoon_psth(ax):
        sns.despine(ax=ax, left=True, trim=True, offset=0)
        ax.set_yticks([])
        ax.legend(ncol=2, bbox_to_anchor=(1, 1.35))
        ax.text(-0.01, 1.3, 'A', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_cartoon_psth_stim(ax):
        sns.despine(ax=ax, left=True, trim=True, offset=0)
        ax.set_yticks([])
        ax.legend(ncol=3, bbox_to_anchor=(1,1.35), loc=1)
        ax.text(-0.01, 1.3, 'B', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_spectrum_base(ax):
        # --- spectrum
        sns.despine(ax=ax, left=True, trim=True, offset=0)
        ax.set_title('vector strength spectrum' , fontsize=ax.xaxis.get_ticklabels()[0].get_fontsize())
        ax.set_yticks([])
        ax.set_xlabel('')

    @staticmethod
    def format_polar_base(ax):
        thetaticks = np.arange(0, 360, 45)
        ax.spines['polar'].set_color(colordict['stimulus'])
        ax.set_thetagrids(thetaticks, frac=1.3)
        ax.set_xticklabels([r'$0$', r'$\frac{1}{8f_s}$', r'$\frac{1}{4f_s}$', r'$\frac{3}{8f_s}$', r'$\frac{1}{2f_s}$', \
                            r'$\frac{5}{8f_s}$', r'$\frac{3}{4f_s}$', r'$\frac{7}{8f_s}$'])
        ax.set_yticks([])

    @staticmethod
    def format_polar_stim(ax):
        thetaticks = np.arange(0, 360, 45)
        ax.spines['polar'].set_color(colordict['stimulus'])

        ax.set_thetagrids(thetaticks, frac=1.3)
        ax.set_xticklabels([r'$0$', r'$\frac{1}{8f_s}$', r'$\frac{1}{4f_s}$', r'$\frac{3}{8f_s}$', r'$\frac{1}{2f_s}$', \
                            r'$\frac{5}{8f_s}$', r'$\frac{3}{4f_s}$', r'$\frac{7}{8f_s}$'])
        ax.set_yticks([])

    @staticmethod
    def format_spectrum_stim(ax):
        # --- spectrum
        sns.despine(ax=ax, left=True, trim=True)
        ax.set_yticks([])

    def format_figure(self):
        for a in self.ax.values():
            a.tick_params(length=3, width=1)
            if 'bottom' in a.spines:
                a.spines['bottom'].set_linewidth(1)
            if 'left' in a.spines:
                a.spines['left'].set_linewidth(1)
        # self.gs.tight_layout(self.fig)
        fig.subplots_adjust(hspace=.9, left=.05, right=.95, top=.85)


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

                    # --- plot time cartoon_psth baseline
                    eod = target_trials.fetch('eod').mean()
                    delta_f = eod / 4
                    stim_period = 1 / (eod - delta_f)
                    print('Beat has period', eod / delta_f, 'EOD cycles')
                    var = (1 / 12 / eod) ** 2
                    t = np.linspace(-N / eod, N / eod, 10000)
                    t_rad = np.linspace(-5 * N, 5 * N, 10000) * 2 * np.pi
                    rad2t = 1 / 2 / np.pi * stim_period

                    base = lambda t: np.cos(2 * np.pi * eod * t) + 1
                    beat = lambda t: np.cos(2 * np.pi * delta_f * t) + 1
                    stim = lambda t: np.cos(2 * np.pi * (eod - delta_f) * t) + 1
                    stim_eod = lambda t: base(t) + stim(t)
                    f_base = lambda t: sum(gauss(t, mu, var) for mu in np.arange(-N * 5 / eod, N * 5 / eod, 1 / eod))
                    f_stim = lambda t: sum(
                        gauss(t, mu, var) * beat(mu) for mu in np.arange(-N * 5 / eod, N * 5 / eod, 1 / eod))

                    ax['cartoon_psth'].fill_between(t, 0 * t, f_base(t), color='grey', lw=0, label='PSTH')
                    ax['cartoon_psth'].plot(t, base(t), '-', color=colordict['eod'], label='EOD')
                    ax['cartoon_psth'].set_ylim((0, 2.1))

                    p = f_base(t_rad * rad2t)
                    p /= p.sum()
                    x,y = (p*np.cos(t_rad)).sum(), (p*np.sin(t_rad)).sum()
                    ax['polar_base'].fill_between(t_rad, 0 * t_rad, f_base(t_rad * rad2t), color='grey', lw=0)
                    ax['polar_base'].plot(np.arctan2(y,x), np.sqrt(x**2 + y**2), 'ok', mfc='k', lw=0)

                    ax['cartoon_psth_stim'].fill_between(t, 0 * t, f_stim(t), color='grey', lw=0, label='PSTH')
                    ax['cartoon_psth_stim'].plot(t, stim_eod(t), '-', color=colordict['stimulus'])
                    ax['cartoon_psth_stim'].plot(t, stim_eod(t), '--', color=colordict['eod'], dashes=(10, 10))
                    ax['cartoon_psth_stim'].plot(t, stim(t) * .5 + 4.1, '-', color=colordict['stimulus'], lw=1,
                                                 label='stimulus')
                    ax['cartoon_psth_stim'].plot(t, base(t) * .5 + 4.1, '-', color=colordict['eod'], lw=1, label='EOD')
                    ax['cartoon_psth_stim'].set_ylim((0, 5.2))

                    p = f_stim(t_rad * rad2t)
                    p /= p.sum()
                    x,y = (p*np.cos(t_rad)).sum(), (p*np.sin(t_rad)).sum()
                    ax['polar_stim'].plot(np.arctan2(y,x), np.sqrt(x**2 + y**2), 'ok', mfc='k', lw=0)
                    ax['polar_stim'].fill_between(t_rad, 0 * t_rad, f_stim(t_rad * rad2t), color='grey', lw=0)
                    ax['polar_stim'].plot(t_rad[::5], stim(t_rad[::5] * rad2t), '.', ms=2.5, color=colordict['stimulus'])

                    for k in ['cartoon_psth', 'cartoon_psth_stim']:
                        ax[k].set_xticks(np.arange(-N / eod, (N + 1) / eod, 5 / eod))
                        ax[k].set_xlim((-N / eod, N / eod))
                        ax[k].set_xticklabels([])
                        ax[k].set_xticklabels(np.arange(-N, N + 1, 5))
                        ax[k].set_xlabel('time [EOD cycles]')

                    F_base = fftshift(fft(f_base(t)))
                    w_base = fftshift(fftfreq(f_base(t).size, t[1] - t[0]))

                    idx = abs(w_base) < f_max
                    ax['spectrum_base'].plot(w_base[idx], abs(F_base[idx]), '-', color='gray')

                    F_stim = fftshift(fft(f_stim(t)))
                    w_stim = fftshift(fftfreq(f_stim(t).size, t[1] - t[0]))
                    idx = abs(w_stim) < f_max

                    ax['spectrum_stim'].plot(w_stim[idx], abs(F_stim[idx]), '-', color='gray')

                    for k in ['spectrum_base', 'spectrum_stim']:
                        ax[k].set_xticks(np.arange(-2 * eod, 3 * eod, eod))
                        ax[k].set_xlim((-f_max, f_max))
                        ax[k].set_ylim((0, F_stim.max()*1.1))
                        ax[k].set_xticklabels([])
                        ax[k].set_xticklabels(np.arange(-2, 3))
                        ax[k].set_xlabel('frequency [EODf]')
