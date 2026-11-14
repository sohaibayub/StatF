# Repeated stratified 10-fold cross-validation (10 repeats) for StatF and the
# performance-based baselines, with paired significance tests.
import numpy as np, pandas as pd, warnings, json, time
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

SEED = 389798
LABEL = 'LOGavgPowClusters3'
SIX = ['first-contentful-paint', 'speed-index', 'largest-contentful-paint',
       'interactive', 'total-blocking-time', 'cumulative-layout-shift']

df = pd.read_csv('sites_dataset.csv')
df = df[df[LABEL].notna()].reset_index(drop=True)
# relabel tiers as 0 = low, 1 = moderate, 2 = high (ordered by mean energy)
order = df.groupby(LABEL)['energyJ'].mean().sort_values().index.tolist()
y = df[LABEL].map({k: i for i, k in enumerate(order)}).values
cols = df.columns.tolist()
FEATS = {'GLH': cols[33:48], 'WPT': cols[0:33], 'ALL': cols[0:48],
         'PERF1': ['performance-score'], 'PERF6': SIX}

def models():
    return {
        'NB': GaussianNB(),
        'LR': LogisticRegression(solver='lbfgs', penalty='l2', max_iter=10000),
        'KNN': KNeighborsClassifier(n_neighbors=21, weights='distance'),
        'SVM': SVC(probability=True, random_state=SEED),
        'DT': DecisionTreeClassifier(random_state=SEED),
        'RF': RandomForestClassifier(n_estimators=100, random_state=SEED),
        'MLP': MLPClassifier(hidden_layer_sizes=(50, 100, 50), max_iter=500, random_state=SEED),
    }

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED)
splits = list(rskf.split(np.zeros(len(y)), y))

def run(fs, name):
    X = df[FEATS[fs]].values.astype(float)
    out, preds = [], np.full((10, len(y)), -1)
    for k, (tr, te) in enumerate(splits):
        clf = make_pipeline(StandardScaler(), models()[name])
        t0 = time.time(); clf.fit(X[tr], y[tr]); tt = time.time() - t0
        p = clf.predict(X[te]); pr = clf.predict_proba(X[te])
        preds[k // 10, te] = p
        out.append(dict(acc=accuracy_score(y[te], p),
                        prec=precision_score(y[te], p, average='macro', zero_division=0),
                        rec=recall_score(y[te], p, average='macro', zero_division=0),
                        f1=f1_score(y[te], p, average='macro', zero_division=0),
                        auc=roc_auc_score(y[te], pr, multi_class='ovr', average='macro'),
                        time=tt))
    return fs, name, pd.DataFrame(out), preds

jobs = [(fs, m) for fs in FEATS for m in models()]
res = Parallel(n_jobs=-1)(delayed(run)(fs, m) for fs, m in jobs)
R = {(fs, m): (d, p) for fs, m, d, p in res}

rows = []
for (fs, m), (d, p) in R.items():
    r = {'features': fs, 'model': m}
    for c in d.columns:
        r[c + '_mean'] = d[c].mean(); r[c + '_std'] = d[c].std()
    rows.append(r)
summ = pd.DataFrame(rows); summ.to_csv('cv_summary.csv', index=False)
pd.set_option('display.width', 200)
print(summ.round(3).to_string(index=False))

def corrected_t(a, b, n_train, n_test):
    # Nadeau and Bengio corrected resampled t-test for repeated k-fold CV
    d = np.asarray(a) - np.asarray(b); n = len(d)
    var = d.var(ddof=1) * (1.0 / n + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return d.mean(), t, 2 * stats.t.sf(abs(t), n - 1)

def mcnemar(p1, p2):
    c1, c2 = (p1 == y), (p2 == y)
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    return b, c, stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0

n_te = len(y) / 10.0; n_tr = len(y) - n_te
best = {}
for fs in FEATS:
    s = summ[summ.features == fs]
    best[fs] = {'acc': s.loc[s.acc_mean.idxmax(), 'model'], 'auc': s.loc[s.auc_mean.idxmax(), 'model']}
tests = []
ref = R[('ALL', 'RF')]
for base in ['PERF1', 'PERF6', 'GLH', 'WPT']:
    for metric in ['acc', 'f1', 'auc']:
        bm = best[base]['auc' if metric == 'auc' else 'acc']
        d, t, pv = corrected_t(ref[0][metric], R[(base, bm)][0][metric], n_tr, n_te)
        tests.append(dict(compare='ALL/RF vs %s/%s' % (base, bm), metric=metric, diff=d, t=t, p=pv))
    bm = best[base]['acc']
    ps = [mcnemar(ref[1][r], R[(base, bm)][1][r]) for r in range(10)]
    tests.append(dict(compare='ALL/RF vs %s/%s' % (base, bm), metric='mcnemar (median over 10 repeats)',
                      diff=np.median([a - b for a, b, _ in ps]), t=np.nan, p=float(np.median([q for _, _, q in ps]))))
T = pd.DataFrame(tests); T.to_csv('significance_tests.csv', index=False)
print(T.round(4).to_string(index=False))

cm = sum(confusion_matrix(y, ref[1][r]) for r in range(10)) / 10.0
print('confusion matrix (mean over 10 repeats, rows actual low/moderate/high)'); print(np.round(cm, 1), cm.sum())
np.savetxt('confusion_matrix_all_rf.csv', cm, delimiter=',', fmt='%.1f')
json.dump(best, open('best_models.json', 'w'))
