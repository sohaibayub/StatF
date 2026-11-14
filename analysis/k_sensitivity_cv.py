# Sensitivity of the tier classification to the number of tiers (k = 3, 4, 5).
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
warnings.filterwarnings('ignore')
SEED = 389798
df = pd.read_csv('sites_dataset.csv'); df = df[df['LOGavgPowClusters3'].notna()].reset_index(drop=True)
X = df[df.columns[0:48]].values
for k in [3, 4, 5]:
    y = df['LOGavgPowClusters%d' % k].astype(int).values
    a, u = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED).split(X, y):
        m = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
        a.append(accuracy_score(y[te], m.predict(X[te]))); u.append(roc_auc_score(y[te], m.predict_proba(X[te]), multi_class='ovr'))
    print('k=%d accuracy %.3f +- %.3f, AUC-ROC %.3f +- %.3f, chance %.3f' % (k, np.mean(a), np.std(a, ddof=1), np.mean(u), np.std(u, ddof=1), np.bincount(y).max() / len(y)))
