# import matplotlib
# matplotlib.use('Agg')
import os
from collections import OrderedDict

colors = ['#F47F17', '#3673A4', '#BA2D22', '#AAB71B', "gray"]
colordict = OrderedDict(zip(['stimulus', 'eod', 'baseline', 'delta_f', 'combinations'], colors))
markers = [(4, 0, 90), '^', 'D', 's', 'o']
markerdict = OrderedDict(zip(['stimulus', 'eod', 'baseline', 'delta_f', 'combinations'], markers))

def mkdir(newdir):
    if os.path.isdir(newdir):
        pass
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            mkdir(head)
        if tail:
            os.mkdir(newdir)