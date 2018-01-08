from locker import data, sanity
import numpy as np
import datajoint as dj

print('These Runs have no spikes at all and should be deleted')
print(data.Runs() * sanity.SpikeCheck() & 'all_zeros > 0')


for k in (data.Runs() * sanity.SpikeCheck() & 'all_zeros > 0').proj().fetch.keys():
    (data.Runs() & k).delete()

for k in (sanity.SpikeCheck.SpikeCount() * data.Runs.SpikeTimes() & 'is_empty=1').fetch.keys():
    (data.Runs().SpikeTimes() & k)._update('times', np.array([]))

print('Deleting trials where Relacs and peakdet deviate too much')
for k in sanity.PeakTroughCheck().inconsistent_peakdet_relacs_runs.fetch.keys():
    (data.Runs() & k).delete()