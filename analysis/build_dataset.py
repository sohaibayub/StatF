# Rebuilds the site-level dataset with URLs, energy in joules, mean power,
# load time, the Lighthouse performance score and its six metric scores.
import pandas as pd, numpy as np, glob, os

ROOT = '../'
POW = ROOT + 'PowerAPI/Chrome-Selenium/'
INTERVAL = 0.5  # SmartWatts sampling interval in seconds

runs = sorted(glob.glob(POW + 'cleaned-data/*.csv'))
per_run = []
for i, f in enumerate(runs, 1):
    d = pd.read_csv(f)
    g = d.groupby('url')['power_watts']
    per_run.append(pd.DataFrame({'power%d' % i: g.sum().round(2),
                                 'meanW%d' % i: g.mean(),
                                 'n%d' % i: g.count()}))
p = pd.concat(per_run, axis=1, join='inner').reset_index()

feats = pd.read_csv('StaticFeats.csv')
lab = pd.read_csv('StaticFeats_avgPowWithin5stdev_avgPowClusters.csv')
key = ['power1', 'power2', 'power3']
m = feats.merge(p, on=key, how='left')
print('sites', len(feats), 'matched to url', m['url'].notna().sum(), 'duplicate keys', feats.duplicated(key).sum())
m = m.merge(lab[key + ['LOGavgPowClusters3', 'LOGavgPowClusters4', 'LOGavgPowClusters5']], on=key, how='left')

m['energyJ'] = m['avgPow'] * INTERVAL
m['meanPowerW'] = m[['meanW1', 'meanW2', 'meanW3']].mean(axis=1)
m['durationS'] = m[['n1', 'n2', 'n3']].mean(axis=1) * INTERVAL

six = ['first-contentful-paint', 'speed-index', 'largest-contentful-paint',
       'interactive', 'total-blocking-time', 'cumulative-layout-shift']
rows = []
for f in glob.glob(ROOT + 'Lighthouse/Google/Top1-1000-CSVs/*.csv'):
    url = os.path.basename(f).replace('.report.csv', '').replace('_', '.')
    x = pd.read_csv(f)
    x = x[x['category'] == 'Performance'].drop_duplicates('name').set_index('name')['score']
    r = {'url': url, 'performance-score': x.get('performance-score', np.nan)}
    for s in six:
        r[s] = x.get(s, np.nan)
    rows.append(r)
lh = pd.DataFrame(rows)
m = m.merge(lh, on='url', how='left')
print('with perf score', m['performance-score'].notna().sum(), 'with all six', m[six].notna().all(axis=1).sum())
print((m[six] < 0).sum().to_dict())
m.to_csv('sites_dataset.csv', index=False)
print(m[['energyJ', 'meanPowerW', 'durationS']].describe().T)
