# Paired tests for the average-power tiers and for cross-validated regression.
import numpy as np, pandas as pd, warnings
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
warnings.filterwarnings('ignore')
SEED=389798
SIX=['first-contentful-paint','speed-index','largest-contentful-paint','interactive','total-blocking-time','cumulative-layout-shift']
df=pd.read_csv('sites_dataset.csv'); df=df[df['LOGavgPowClusters3'].notna()].reset_index(drop=True)
c=df.columns.tolist(); ALL=c[0:48]
def ct(a,b):
    d=np.array(a)-np.array(b);n=len(d);v=d.var(ddof=1)*(1/n+1/9);t=d.mean()/np.sqrt(v);return d.mean(),2*stats.t.sf(abs(t),n-1)
lp=np.log(df.meanPowerW.values)
km=KMeans(n_clusters=3,n_init=20,random_state=SEED).fit(lp.reshape(-1,1)); o=np.argsort(km.cluster_centers_.ravel()); y=np.array([list(o).index(k) for k in km.labels_])
sp=list(RepeatedStratifiedKFold(n_splits=10,n_repeats=10,random_state=SEED).split(y,y))
def cls(cols,mk):
    X=df[cols].values;a=[];u=[]
    for tr,te in sp:
        m=mk().fit(X[tr],y[tr]); a.append(accuracy_score(y[te],m.predict(X[te]))); u.append(roc_auc_score(y[te],m.predict_proba(X[te]),multi_class='ovr'))
    return a,u
rf=lambda:RandomForestClassifier(n_estimators=100,random_state=SEED,n_jobs=-1); lr=lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=10000))
s=cls(ALL,rf); p6=cls(SIX,lr); p1=cls(['performance-score'],lr)
print('avg power tiers: ALL/RF vs PERF6/LR acc diff %.3f p %.4f'%ct(s[0],p6[0]),'auc diff %.3f p %.4f'%ct(s[1],p6[1]))
print('avg power tiers: ALL/RF vs PERF1/LR acc diff %.3f p %.4f'%ct(s[0],p1[0]),'auc diff %.3f p %.4f'%ct(s[1],p1[1]))
le=np.log(df.energyJ.values); sk=list(RepeatedKFold(n_splits=10,n_repeats=10,random_state=SEED).split(le))
def reg(cols,mk,t):
    X=df[cols].values;r=[]
    for tr,te in sk:
        m=mk().fit(X[tr],t[tr]); r.append(r2_score(t[te],m.predict(X[te])))
    return r
rr=lambda:RandomForestRegressor(n_estimators=200,random_state=SEED,n_jobs=-1); ln=lambda:make_pipeline(StandardScaler(),LinearRegression())
a=reg(ALL,rr,le); b=reg(SIX,rr,le); c1=reg(['performance-score'],ln,le); d6=reg(SIX,ln,le)
print('log energy R2: ALL/RF %.3f, PERF6/RF %.3f, PERF6/lin %.3f, PERF1/lin %.3f'%(np.mean(a),np.mean(b),np.mean(d6),np.mean(c1)))
print('ALL/RF vs PERF6/RF diff %.3f p %.4f'%ct(a,b),'| vs PERF1/lin diff %.3f p %.4f'%ct(a,c1))
