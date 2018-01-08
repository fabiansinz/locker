from locker import modeling as m

for rel in [m.EODFit, m.LIFPUnit, m.PUnitSimulations]:
    print('Populating', rel.__name__)
    rel().populate(reserve_jobs=True)
    print(80*'-')
