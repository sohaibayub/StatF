import numpy as np, pandas as pd, warnings
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
warnings.filterwarnings('ignore')
SEED=389798
SIX=['first-contentful-paint','speed-index','largest-contentful-paint','interactive','total-blocking-time','cumulative-layout-shift']
df=pd.read_csv('sites_dataset.csv'); df=df[df['LOGavgPowClusters3'].notna()].reset_index(drop=True)
order=df.groupby('LOGavgPowClusters3')['energyJ'].mean().sort_values().index.tolist()
y=df['LOGavgPowClusters3'].map({k:i for i,k in enumerate(order)}).values
c=df.columns.tolist(); F={'ALL':c[0:48],'PERF6':SIX,'ALL+PERF6':c[0:48]+SIX,'ALL+PERF1':c[0:48]+['performance-score']}
sp=list(RepeatedStratifiedKFold(n_splits=10,n_repeats=10,random_state=SEED).split(y,y))
res={}
for k,cols in F.items():
    X=df[cols].values; a=[];u=[]
    for tr,te in sp:
        m=RandomForestClassifier(n_estimators=100,random_state=SEED,n_jobs=-1).fit(X[tr],y[tr])
        a.append(accuracy_score(y[te],m.predict(X[te]))); u.append(roc_auc_score(y[te],m.predict_proba(X[te]),multi_class='ovr'))
    res[k]=(np.array(a),np.array(u)); print(k,'acc %.3f+-%.3f auc %.3f+-%.3f'%(np.mean(a),np.std(a,ddof=1),np.mean(u),np.std(u,ddof=1)))
def ct(a,b):
    d=a-b;n=len(d);v=d.var(ddof=1)*(1/n+1/9);t=d.mean()/np.sqrt(v);return d.mean(),2*stats.t.sf(abs(t),n-1)
for x,yk in [('ALL+PERF6','PERF6'),('ALL+PERF6','ALL'),('ALL+PERF1','ALL')]:
    print(x,'vs',yk,'acc diff %.3f p %.4f'%ct(res[x][0],res[yk][0]),'| auc diff %.3f p %.4f'%ct(res[x][1],res[yk][1]))
