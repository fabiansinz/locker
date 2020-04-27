import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scripts.config import params as plot_params
from locker.analysis import *
from matplotlib import cm
from itertools import repeat

def jitter_yvals(ax, start_range, offset_range, col='Blues'):
    unsorted_lines = []

    unsorted_yvals = []
    for i,l in enumerate(ax.lines):
        if i%3==0:
            temp = l.get_data()
            unsorted_lines.append(temp[1])
            unsorted_yvals.append(temp[1][0])
        
    sorted_yvals = np.argsort(unsorted_yvals)
    sorted_yvals = [i * 3 for i in sorted_yvals]
    line_cols = cm.get_cmap(col)
    colors = line_cols(np.linspace(0.5,1,len(sorted_yvals)))
    colors = [c for item in colors for c in repeat(item, 3)]
    new_sort = []
    for lin in sorted_yvals:
        new_sort.append(lin)
        new_sort.append(lin+1)
        new_sort.append(lin+2)
    ax.lines = [ax.lines[i] for i in new_sort]
    offset = np.linspace(-offset_range+start_range, offset_range+start_range*2, (len(ax.lines)/3))
    for i, (l, c) in enumerate(zip(ax.lines, colors)):
        ofs = offset[int(np.floor(i/3))]
        temp = l.get_data()
        l.set_data(([temp[0][0]+ofs, temp[0][1]+ofs], temp[1]))
        l.set_color(c)      


rel = Runs() * SecondOrderSignificantPeaks() * StimulusSpikeJitter() * Cells() \
      & dict(stimulus_coeff=1, eod_coeff= 0, baseline_coeff=0, refined=1, \
             cell_type='p-unit', am=0, n_harmonics=0) \
      & 'frequency > 0'
# & 'stimulus_coeff = 1' \

rel_beat = Runs() * SecondOrderSignificantPeaks() * StimulusSpikeJitter() * Cells() \
      & dict(stimulus_coeff=1, eod_coeff=-1, baseline_coeff=0, refined=1, \
             cell_type='p-unit', am=0, n_harmonics=0) \
      & 'frequency > 0'

df = pd.DataFrame(rel.fetch())
df['spread'] = df['stim_std'] / df['eod'] / 2 / np.pi
df['jitter'] = df['stim_std']  # rename to avoid conflict with std function

df2 = pd.DataFrame(rel_beat.fetch())
df2['spread'] = df2['stim_std'] / df2['eod'] / 2 / np.pi
df2['jitter'] = df2['stim_std']  # rename to avoid conflict with std function

sns.set_context('paper')
sns.set_style('ticks')

fig = plt.figure(figsize=(12, 6), dpi=400)
gs = plt.GridSpec(2, 3)
ax = {}
ax['contrast_s'] = fig.add_subplot(gs[1, 0]) 
ax['stimulus_s'] = fig.add_subplot(gs[1, 1])
ax['cstd_s'] = fig.add_subplot(gs[1, 2])
ax['contrast_b'] = fig.add_subplot(gs[0, 0]) 
ax['stimulus_b'] = fig.add_subplot(gs[0, 1])
ax['cstd_b'] = fig.add_subplot(gs[0, 2])
plt.tight_layout()

# =============================================================================
# -- plot contrast vs. vector strength stimulus

df_contrast = df
df2_contrast = df2

print(df_contrast.columns.to_series())
sns.pointplot('contrast', 'vector_strength', data=df[df.stimulus_coeff == 1],
              order=[10, 20],
              ax=ax['contrast_s'], hue='cell_id',
              palette={ci: sns.xkcd_rgb['azure'] for ci in pd.unique(df.cell_id)},
              join=True, errwidth=1., scale=.5, markers=['' for ci in pd.unique(df.cell_id)])
leg = ax['contrast_s'].legend(title=None, ncol=1, fontsize=6, loc='upper left')
leg.set_visible(False)
ax['contrast_s'].set_ylabel('vector strength stimulus', fontsize=12)
ax['contrast_s'].tick_params('y', length=3, width=1)
# ax['contrast'].set_xlim((-.2, 3.5))
ax['contrast_s'].set_xlabel('contrast [%]', fontsize=12)
ax['contrast_s'].text(-0.18, 1.05, 'D', transform=ax['contrast_s'].transAxes, fontweight='bold', fontsize = 14)
ax['contrast_s'].set_ylim((0, 1))
jitter_yvals(ax['contrast_s'], 0, 0.15)

# -- plot contrast vs. vector strength stimulus
sns.pointplot('contrast', 'vector_strength', data=df2[df2.stimulus_coeff == 1],
              order=[10, 20],
              ax=ax['contrast_b'], hue='cell_id',
              palette={ci: sns.xkcd_rgb['azure'] for ci in pd.unique(df2.cell_id)},
              join=True, errwidth=1., scale=.5, markers=['' for ci in pd.unique(df2.cell_id)])
leg = ax['contrast_b'].legend(title=None, ncol=1, fontsize=6, loc='upper left')
leg.set_visible(False)
ax['contrast_b'].set_ylabel('vector strength beat', fontsize=12)
ax['contrast_b'].tick_params('y', length=3, width=1)
ax['contrast_b'].set_xlabel('contrast [%]', fontsize=12)
ax['contrast_b'].text(-0.18, 1.05, 'A', transform=ax['contrast_b'].transAxes, fontweight='bold', fontsize = 14)
ax['contrast_b'].set_ylim((0, 1))
print('LOOK HERE'*10)
print(ax['contrast_b'].get_xticks())
print("")
jitter_yvals(ax['contrast_b'], 0, 0.15)

# =============================================================================
# --- statistical analysis stimulus

glm = smf.glm('vector_strength ~ frequency * jitter + contrast', data=df, family=sm.families.Gamma()).fit()

print(glm.summary())
print(glm.pvalues)

print('1 sigma in Frequency domain', np.mean(1 / (2 * np.pi * df.spread)))
print('min sigma in Frequency domain', np.min(1 / (2 * np.pi * df.spread)))
print('max sigma in Frequency domain', np.max(1 / (2 * np.pi * df.spread)))
print('2 sigma in Frequency domain', np.mean(2 / (2 * np.pi * df.spread)))

print(r"contrast: \rho={0}    p={1}".format(*stats.pearsonr(df.contrast, df.vector_strength)))
df = df[df.contrast == 20]
print(r"jitter: \rho={0}    p={1}".format(*stats.pearsonr(df.jitter, df.vector_strength)))
print(r"frequency: \rho={0}    p={1}".format(*stats.pearsonr(df.frequency, df.vector_strength)))

# --- statistical analysis beat

glm = smf.glm('vector_strength ~ frequency * jitter + contrast', data=df2, family=sm.families.Gamma()).fit()

print(glm.summary())
print(glm.pvalues)

print('1 sigma in Frequency domain', np.mean(1 / (2 * np.pi * df2.spread)))
print('min sigma in Frequency domain', np.min(1 / (2 * np.pi * df2.spread)))
print('max sigma in Frequency domain', np.max(1 / (2 * np.pi * df2.spread)))
print('2 sigma in Frequency domain', np.mean(2 / (2 * np.pi * df2.spread)))

print(r"contrast: \rho={0}    p={1}".format(*stats.pearsonr(df2.contrast, df2.vector_strength)))
df2 = df2[df2.contrast == 20]
print(r"jitter: \rho={0}    p={1}".format(*stats.pearsonr(df2.jitter, df2.vector_strength)))
print(r"frequency: \rho={0}    p={1}".format(*stats.pearsonr(df2.frequency, df2.vector_strength)))

# =============================================================================
# --- plot frequency vs. vector strength stimulus
sc = ax['stimulus_s'].scatter(df.frequency, df.vector_strength, c=df.jitter, cmap=plt.get_cmap('viridis'), edgecolors='w',
                            lw=.5, s=20)
#position = fig.add_axes([0.59,0.12,0.008,0.34])
cb = fig.colorbar(sc, ax=ax['stimulus_s'], pad=-0.1, shrink=0.9)
cb.set_ticks((np.pi / 4 , np.pi / 2,3 *np.pi / 4, np.pi))
cb.set_ticklabels([r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{2}$', r'$\pi$'])
cb.set_label('circular std at EODf', fontsize=12)
ax['stimulus_s'].set_ylim((0, 1))
ax['stimulus_s'].tick_params('y', length=0, width=0)
ax['stimulus_s'].set_yticklabels([])
ax['stimulus_s'].set_xlim((0, 1800))
ax['stimulus_s'].set_xticks(np.arange(0, 2000, 500))
ax['stimulus_s'].set_xlabel('stimulus frequency [Hz]', fontsize=12)
ax['stimulus_s'].text(-0.1, 1.05, 'E', transform=ax['stimulus_s'].transAxes, fontweight='bold', fontsize = 14)

# --- plot frequency vs. vector strength beat
sc = ax['stimulus_b'].scatter(df2.frequency, df2.vector_strength, c=df2.jitter, cmap=plt.get_cmap('viridis'), edgecolors='w',
                            lw=.5, s=20)
#position2 = fig.add_axes([0.59,0.58,0.008,0.35])
cb = fig.colorbar(sc, ax=ax['stimulus_b'], pad=-0.1, shrink=0.9)
cb.set_ticks((np.pi / 4 , np.pi / 2,3 *np.pi / 4, np.pi))
cb.set_ticklabels([r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{2}$', r'$\pi$'])
cb.set_label('circular std at EODf', fontsize=12)
ax['stimulus_b'].set_ylim((0, 1))
ax['stimulus_b'].tick_params('y', length=0, width=0)
ax['stimulus_b'].set_yticklabels([])
ax['stimulus_b'].set_xlim((0, 600))
ax['stimulus_b'].set_xticks(np.arange(0, 600, 100))
ax['stimulus_b'].set_xlabel('beat frequency [Hz]', fontsize=12)
ax['stimulus_b'].text(-0.1, 1.05, 'B', transform=ax['stimulus_b'].transAxes, fontweight='bold', fontsize = 14)

# =============================================================================
# --- plot jitter vs. vector strength stimulus
sc = ax['cstd_s'].scatter(df.jitter, df.vector_strength, c=df.frequency, cmap=plt.get_cmap('plasma'), edgecolors='w',
                        lw=.5, s=20)
cb = fig.colorbar(sc, ax=ax['cstd_s'],shrink=0.9)
cb.set_label('stimulus frequency [Hz]', fontsize=12)
# t = np.linspace(1e-6, 1-1e-6,100)
ax['cstd_s'].set_xlim((0, 3.14))
ax['cstd_s'].set_xticks((0, np.pi / 4, np.pi / 2, 3 / 4 * np.pi, np.pi))
ax['cstd_s'].set_xticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
ax['cstd_s'].tick_params('y', length=0, width=0)
ax['cstd_s'].set_xlabel('circular std at EODf', fontsize=12)
ax['cstd_s'].set_yticklabels([])
ax['cstd_s'].set_ylim((0, 1))
# ax['cstd'].plot(np.sqrt(-2*np.log(t)), t , '--k')
ax['cstd_s'].text(-0.15, 1.05, 'F', transform=ax['cstd_s'].transAxes, fontweight='bold', fontsize = 14)

# --- plot jitter vs. vector strength beat
sc = ax['cstd_b'].scatter(df2.jitter, df2.vector_strength, c=df2.frequency, cmap=plt.get_cmap('plasma'), edgecolors='w',
                        lw=.5, s=20)
cb = fig.colorbar(sc, ax=ax['cstd_b'],shrink=0.9)
cb.set_label('beat frequency [Hz]', fontsize=12)
ax['cstd_b'].set_xlim((0, 3.14))
ax['cstd_b'].set_xticks((0, np.pi / 4, np.pi / 2, 3 / 4 * np.pi, np.pi))
ax['cstd_b'].set_xticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
ax['cstd_b'].tick_params('y', length=0, width=0)
ax['cstd_b'].set_xlabel('circular std at EODf', fontsize=12)
ax['cstd_b'].set_yticklabels([])
ax['cstd_b'].set_ylim((0, 1))
ax['cstd_b'].text(-0.15, 1.05, 'C', transform=ax['cstd_b'].transAxes, fontweight='bold', fontsize = 14)

# =============================================================================
for a in ax.values():
    a.tick_params('x', length=3, width=1)

sns.despine(fig, trim=True)
for a in ax.values():
    for axis in ['top', 'bottom', 'left', 'right']:
        a.spines[axis].set_linewidth(1)

sns.despine(ax=ax['cstd_s'], left=True)
sns.despine(ax=ax['stimulus_s'], left=True)
sns.despine(ax=ax['cstd_b'], left=True)
sns.despine(ax=ax['stimulus_b'], left=True)

fig.tight_layout()
fig.subplots_adjust(left=.1, right=0.9, top=.9)
fig.savefig('figures/fig08-factors-punit.pdf')
fig.savefig('figures/fig08-factors-punit.png')
