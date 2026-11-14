# Regression with cross-validation, the split of energy into average power and
# active time, and tier prediction for average power.
import numpy as np, pandas as pd, warnings
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
SEED = 389798
SIX = ['first-contentful-paint', 'speed-index', 'largest-contentful-paint',
       'interactive', 'total-blocking-time', 'cumulative-layout-shift']
df = pd.read_csv('sites_dataset.csv'); df = df[df['LOGavgPowClusters3'].notna()].reset_index(drop=True)
cols = df.columns.tolist()
F = {'ALL': cols[0:48], 'GLH': cols[33:48], 'WPT': cols[0:33], 'PERF1': ['performance-score'], 'PERF6': SIX}

print('n =', len(df))
le, lp, lt = np.log(df.energyJ), np.log(df.meanPowerW), np.log(df.durationS)
print('Spearman energy~active time %.3f, energy~avg power %.3f, avg power~active time %.3f' % (
    spearmanr(df.energyJ, df.durationS)[0], spearmanr(df.energyJ, df.meanPowerW)[0], spearmanr(df.meanPowerW, df.durationS)[0]))
print('share of var(log E): time %.3f, power %.3f, covariance %.3f' % (lt.var() / le.var(), lp.var() / le.var(), 2 * np.cov(lp, lt)[0, 1] / le.var()))
print(df[['energyJ', 'meanPowerW', 'durationS']].describe().round(2).T[['min', '50%', 'mean', 'max']])
for name, target in [('energy', df.energyJ), ('avg power', df.meanPowerW), ('active time', df.durationS)]:
    print('Spearman of GLH performance score with %s: %.3f' % (name, spearmanr(df['performance-score'], target)[0]))

rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=SEED)
rows = []
for tname, t in [('log energy', le), ('log avg power', lp), ('log active time', lt)]:
    for fs in ['ALL', 'PERF1', 'PERF6']:
        for mname, m in [('linear', make_pipeline(StandardScaler(), LinearRegression())),
                         ('RF', RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1))]:
            r2 = cross_val_score(m, df[F[fs]].values, t.values, cv=rkf, scoring='r2')
            fit = m.fit(df[F[fs]].values, t.values).score(df[F[fs]].values, t.values)
            rows.append(dict(target=tname, features=fs, model=mname, cv_r2=r2.mean(), cv_r2_std=r2.std(), in_sample_r2=fit))
R = pd.DataFrame(rows); R.to_csv('regression_cv.csv', index=False); print(R.round(3).to_string(index=False))

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED)
rows = []
for tname, t in [('avg power', lp), ('active time', lt)]:
    km = KMeans(n_clusters=3, n_init=20, random_state=SEED).fit(t.values.reshape(-1, 1))
    order = np.argsort(km.cluster_centers_.ravel()); lab = np.array([list(order).index(c) for c in km.labels_])
    print(tname, 'tier sizes', np.bincount(lab))
    for fs, clf in [('ALL', RandomForestClassifier(n_estimators=100, random_state=SEED)),
                    ('PERF1', make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))),
                    ('PERF6', RandomForestClassifier(n_estimators=100, random_state=SEED)),
                    ('PERF6', make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000)))]:
        acc = cross_val_score(clf, df[F[fs]].values, lab, cv=rskf, scoring='accuracy')
        auc = cross_val_score(clf, df[F[fs]].values, lab, cv=rskf, scoring='roc_auc_ovr')
        rows.append(dict(target=tname, features=fs, model=type(clf).__name__, acc=acc.mean(), acc_std=acc.std(), auc=auc.mean(), auc_std=auc.std()))
C = pd.DataFrame(rows); C.to_csv('tiers_power_time_cv.csv', index=False); print(C.round(3).to_string(index=False))
