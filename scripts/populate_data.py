from locker import data

for rel in [data.EFishes, data.Cells, data.FICurves, data.ISIHistograms, \
            data.Baseline, data.Runs, data.BaseRate, data.BaseEOD]:
    print('Populating', rel.__name__)
    rel().populate(reserve_jobs=True)
    print(80*'=')

# fix misslabelled cell
(data.Cells() & dict(cell_id='2014-05-21-ab'))._update('cell_type', 'p-unit')
