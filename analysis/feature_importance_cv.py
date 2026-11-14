# Mean Gini importance of the 48 static features over the Random Forest
# models of the repeated 10-fold cross-validation (100 fits).
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
SEED = 389798
CODE = {'requests':'RQS','bytesIn':'BIN','domElements':'DOM','smallImageCount':'SIC','bigImageCount':'BIC',
 'bytes.html':'BYH','requests.html':'RQH','bytesUncompressed.html':'BUH','bytes.js':'BYJ','requests.js':'RQJ','bytesUncompressed.js':'BUJ',
 'bytes.css':'BYC','requests.css':'RQC','bytesUncompressed.css':'BUC','bytes.image':'BYI','requests.image':'RQI','bytesUncompressed.image':'BUI',
 'bytes.font':'BYF','requests.font':'RQF','bytesUncompressed.font':'BUF','bytes.video':'BYV','requests.video':'RQV','bytesUncompressed.video':'BUV',
 'bytes.other':'BYO','requests.other':'RQO','bytesUncompressed.other':'BUO','score_cache':'SCC','score_cdn':'SCD','score_compress':'SCO',
 'score_gzip':'SGZ','gzip_total':'GZT','gzip_savings':'GZS','score_keep-alive':'SKA','render-blocking-resources':'RBR',
 'uses-responsive-images':'URI','offscreen-images':'OSI','unminified-css':'UMC','unminified-javascript':'UMJ','unused-css-rules':'UCR',
 'uses-optimized-images':'UOI','uses-text-compression':'UTC','uses-rel-preconnect':'URP','redirects':'RDR','efficient-animated-content':'EAC',
 'total-byte-weight':'TBW','uses-long-cache-ttl':'ULC','dom-size':'DOS','font-display':'FOD'}
df = pd.read_csv('sites_dataset.csv'); df = df[df['LOGavgPowClusters3'].notna()].reset_index(drop=True)
cols = df.columns[0:48].tolist(); X = df[cols].values; y = df['LOGavgPowClusters3'].astype(int).values
imp = []
for tr, te in RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED).split(X, y):
    imp.append(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1).fit(X[tr], y[tr]).feature_importances_)
imp = np.array(imp)
out = pd.DataFrame({'feature': cols, 'code': [CODE[c] for c in cols], 'importance': imp.mean(0), 'std': imp.std(0, ddof=1)}).sort_values('importance', ascending=False)
out.to_csv('feature_importance_cv.csv', index=False); print(out.head(12).round(4).to_string(index=False))
