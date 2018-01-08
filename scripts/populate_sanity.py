from locker import sanity as s

for rel in [s.SpikeCheck, s.SecondOrderSignificantPeaks, s.PowerAnalysis, s.PeakTroughCheck]:
    print('Populating', rel.__name__)
    rel().populate(reserve_jobs=True)
    print(80*'-')
