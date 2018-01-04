from locker import analysis as a

for rel in [a.TrialAlign, a.FirstOrderSpikeSpectra, a.FirstOrderSignificantPeaks,
            a.SecondOrderSpikeSpectra, a.SecondOrderSignificantPeaks,
            a.PhaseLockingHistogram, a.EODStimulusPSTSpikes,
            a.BaselineSpikeJitter, a.Decoding, a.StimulusSpikeJitter]:
    print('Populating', rel.__name__)
    rel().populate(reserve_jobs=True)
    print(80*'-')
