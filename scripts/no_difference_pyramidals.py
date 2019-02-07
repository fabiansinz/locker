import datajoint as dj
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import stats

from locker import analysis as alys, colors, colordict
from locker import data

alys.FirstOrderSignificantPeaks() * data.Cells() & 'eod_coeff = 0 and abs(stimulus_coeff) = 1 and baseline_coeff=0 and refined=1'

restr = dj.AndList([
    ['cell_type="e-cell"', 'cell_type = "i-cell"'],
        "n_harmonics = 0",
        "am = 0",
        "contrast = 20.0",
         'refined=1',
        '(abs(delta_f) between 95 and 105)',
        'eod_coeff = 0 and abs(stimulus_coeff) = 1 and baseline_coeff=0'
]
)
temp = alys.FirstOrderSignificantPeaks() * data.Runs() * data.Cells() & restr
temp = temp.proj('vector_strength', 'frequency','cell_type', pos='delta_f > 0')

pd_pyr = pd.DataFrame(temp.fetch())

g = sns.catplot("cell_type", "vector_strength", hue="pos", data=pd_pyr, kind="bar")
g.set_ylabels('Cell Type')
g.set_xlabels('Vector Strength')
sns.despine()

fig = plt.gcf()
fig.savefig('figures/no_difference_pyramidal_cells.png')

pd_pyr.groupby(["cell_type", "pos"]).count()
f_tab, p_val = stats.ttest_ind(pd_pyr.loc[pd_pyr['cell_type']=='e-cell', 'vector_strength'], pd_pyr.loc[pd_pyr['cell_type']=='i-cell', 'vector_strength'])

print(p_val)