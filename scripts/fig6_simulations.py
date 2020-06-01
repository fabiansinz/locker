import matplotlib
matplotlib.use('Agg')
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from locker.analysis import SecondOrderSpikeSpectra
from locker.modeling import *
from scripts.config import params as plot_params
from matplotlib.patches import ConnectionPatch

class SimulationFigure:
    def __init__(self, filename=None):
        self.filename = filename

    def __enter__(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        with plt.rc_context(plot_params):
            gs = plt.GridSpec(4, 7)
            self.fig = plt.figure(figsize=(5, 4.5), dpi=600)

            self.ax = {
                'stimulus_spectrum': self.fig.add_subplot(gs[0, 0:5]), 
                'membrane_spectrum': self.fig.add_subplot(gs[1, 0:5]), 
                'sim_spike_spectrum': self.fig.add_subplot(gs[2, 0:5]), 
                'real_spike_spectrum': self.fig.add_subplot(gs[3, 0:5]),
                'flowchart1': self.fig.add_subplot(gs[0, 5:]),
                'flowchart2': self.fig.add_subplot(gs[1, 5:]),
                'flowchart3': self.fig.add_subplot(gs[2, 5:])
            }

            # with sns.axes_style('ticks'):
            #     self.ax['sim_isi'] = inset_axes(self.ax['sim_spike_spectrum'], width=.6, height=.6, loc=1 ,
            #                                     bbox_to_anchor=(1.15, 1.1),
            #                                     bbox_transform=self.ax['sim_spike_spectrum'].transAxes
            #                                     )
            #     self.ax['real_isi'] = inset_axes(self.ax['real_spike_spectrum'], width=.6, height=.6, loc=1,
            #                                      bbox_to_anchor=(1.15, 1.1),
            #                                      bbox_transform=self.ax['real_spike_spectrum'].transAxes
            #                                      )

        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        fig, ax = self.fig, self.ax
        ax['stimulus_spectrum'].yaxis.labelpad = 17
        ax['membrane_spectrum'].yaxis.labelpad = 17
        ax['real_spike_spectrum'].yaxis.labelpad = 2
        ax['sim_spike_spectrum'].yaxis.labelpad = 2
        #ax['real_spike_spectrum'].set_ylim((0, 1.5))
        sns.despine(ax=ax['stimulus_spectrum'], left=True, offset=0)
        ax['stimulus_spectrum'].set_xticklabels([])
        sns.despine(ax=ax['membrane_spectrum'], offset=0, left=True)
        sns.despine(ax=ax['sim_spike_spectrum'], offset=0)
        sns.despine(ax=ax['real_spike_spectrum'], offset=0, trim=True)

        ax['real_spike_spectrum'].set_xlabel('frequency [Hz]')
        ax['real_spike_spectrum'].set_yticks([0, .5, 1])

        for a in ax.values():
            a.tick_params('both', length=3, width=1, which='both')

        fig.tight_layout()
        fig.subplots_adjust(left=0.1, right=0.95)
        
        if ax['real_spike_spectrum'].legend_ is not None:
            ax['real_spike_spectrum'].legend_.set_bbox_to_anchor((1., 1.2),
                                                             transform=ax['real_spike_spectrum'].transAxes)
            ax['real_spike_spectrum'].legend_.get_frame().set_linewidth(0.0)
        ax['stimulus_spectrum'].text(-0.1, 1, 'A', transform=ax['stimulus_spectrum'].transAxes, fontweight='bold')
        ax['membrane_spectrum'].text(-0.1, 1, 'B', transform=ax['membrane_spectrum'].transAxes, fontweight='bold')
        ax['sim_spike_spectrum'].text(-0.1, 1, 'C', transform=ax['sim_spike_spectrum'].transAxes, fontweight='bold')
        ax['real_spike_spectrum'].text(-0.1, 1, 'D', transform=ax['real_spike_spectrum'].transAxes, fontweight='bold')

        ax['real_spike_spectrum'].set_xticklabels(['%.0f' % xt for xt in ax['real_spike_spectrum'].get_xticks()])
        for aname in ['stimulus_spectrum', 'membrane_spectrum', 'sim_spike_spectrum']:
            for tk in  ax[aname].get_xticklabels():
                tk.set_visible(False)

        if self.filename is not None:
            self.fig.savefig(self.filename)

        plt.close(fig)

    def __call__(self, *args, **kwargs):
        return self

for ri, hs in itertools.product([5,13],[0,1]):
    restrictions = dict(
        id='nwgimproved',
        cell_id='2014-12-03-ai',
        # run_id=13,
        run_id=ri,
        harmonic_stimulation=hs

    )

    for key in (PUnitSimulations() & restrictions).fetch.keys():
        print(key)
        print('Processing', key)
        dir = 'figures/figure_simulation/' + key['id']
        mkdir(dir)
        df = (Runs() & key).fetch1('delta_f')
        with SimulationFigure(filename='{dir}/{cell_id}_{df}_harmonics{harmonic_stimulation}.pdf'.format(dir=dir, df=df, **key)) as (fig, ax):
            PUnitSimulations().plot_stimulus_spectrum(key, ax['stimulus_spectrum'])
            PUnitSimulations().plot_membrane_potential_spectrum(key, ax['membrane_spectrum'])
            PUnitSimulations().plot_spike_spectrum(key, ax['sim_spike_spectrum'])
            im1 = plt.imread("flowchart1.png")
            im2 = plt.imread("flowchart2.png")
            im3 = plt.imread("flowchart3.png")
            ax['flowchart1'].imshow(im1)
            arrow = ax['flowchart1'].get_position()
            arrow2 = ax['flowchart2'].get_position()

            ax['flowchart1'].axis('off')
            ax['flowchart2'].imshow(im2)
            ax['flowchart2'].axis('off')
            ax['flowchart3'].imshow(im3)
            ax['flowchart3'].axis('off')
            ax['flowchart1'].annotate("", (0.5, -0.45), (0.5, -0.05), xycoords=ax['flowchart1'].transAxes, arrowprops=dict(facecolor='black', shrink=0.02, width = 2))
            ax['flowchart2'].annotate("", (0.5, -0.45), (0.5, -0.05), xycoords=ax['flowchart2'].transAxes, arrowprops=dict(facecolor='black', shrink=0.02, width = 2))
            
            ax['flowchart1'].annotate("", (0.45, 0.45), (0.25, 0.63), xycoords=ax['flowchart1'].transAxes, arrowprops=dict(facecolor='black', shrink=0.02, width = 2))
            ax['flowchart1'].annotate("", (0.55, 0.45), (0.75, 0.63), xycoords=ax['flowchart1'].transAxes, arrowprops=dict(facecolor='black', shrink=0.02, width = 2))
            
            ax['flowchart2'].annotate("", (0.5, 0.4), (0.5, 0.6), xycoords=ax['flowchart2'].transAxes, arrowprops=dict(facecolor='black', shrink=0.02, width = 2))
            ax['flowchart3'].annotate("", (0.5, 0.35), (0.5, 0.58), xycoords=ax['flowchart3'].transAxes, arrowprops=dict(facecolor='black', shrink=0.02, width = 2))
            #plt.arrow(arrow.x0, arrow.y0, arrow2.x0-arrow.x0, arrow2.y0 - arrow.y0)
            #con = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data", coordsB="data", axesA=ax['flowchart2'], axesB=ax['flowchart1'], arrowstyle="->", shrinkB=5)
            #ax['flowchart2'].add_artist(con)
            restrictions = dict(key, refined=True)
            SecondOrderSpikeSpectra().plot(ax['real_spike_spectrum'], restrictions, f_max=2000, ncol=4)
