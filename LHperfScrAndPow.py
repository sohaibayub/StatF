import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
import sklearn.metrics as metrics
import math
import pickle
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest
import glob
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from  collections import Counter
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, confusion_matrix
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



readDataPath = 'Data-PowerAPI-Lighthouse-WebpageTest/'

def getIndivualAndAveragePowConsumption():
    pathPow = 'PowerAPI/cleaned-data/'
    pow = 'powChromeSelenium-Clean-run'
    dfpow = pd.read_csv(readDataPath + pathPow + pow + '1.csv', delimiter=',')[['power_watts', 'url']]
    dfpow = dfpow.groupby('url').sum().reset_index()
    dfpow.columns = ['url','power1']
    dfpow2 = pd.read_csv(readDataPath + pathPow + pow + '2.csv', delimiter=',')[['power_watts', 'url']]
    dfpow2 = dfpow2.groupby('url').sum().reset_index()
    dfpow2.columns = ['url','power2']
    dfpow3 = pd.read_csv(readDataPath + pathPow + pow + '3.csv', delimiter=',')[['power_watts', 'url']]
    dfpow3 = dfpow3.groupby('url').sum().reset_index()
    dfpow3.columns = ['url','power3']
    dfpow = pd.merge(dfpow, dfpow2, on='url', how='inner')
    dfpow = pd.merge(dfpow, dfpow3, on='url', how='inner')
    #dfpow.to_csv('Pow-Runs-Combined.csv', sep=',', index=False)
    dfpow['avgPow'] = dfpow[['power1', 'power2','power3']].mean(axis=1)
    #dfpow.drop(columns=['power1','power2','power3'], inplace=True)
    return dfpow

def getChromeLighthouseOverallPerformanceScore():
    pathLH = 'Lighthouse/Chrome/CSVs/'
    lighthouseCols = ['performance-score']#open(columnsListFile, "r").read().split('\n')
    dfglhCols = ['url'] + lighthouseCols
    dfglh = pd.DataFrame(columns=dfglhCols)
    for f in glob.glob(readDataPath + pathLH + '*.csv'):
        w = f.split('\\')[1].split('.report.csv')[0].replace('_', '.')
        x = pd.read_csv(f, delimiter=',')
        x = x[x['name'].isin(lighthouseCols)].reset_index(drop=True)
        if (x[x['score'] == -1].shape[0] == 0):
            # print(set(lighthouseCols)-set(x['name']))
            dfglh.loc[len(dfglh)] = [w] + list(x['score'])
    return dfglh

def consolidateData(filename):
    dfglh = getChromeLighthouseOverallPerformanceScore()
    dfpow = getIndivualAndAveragePowConsumption()
    print('GLH Data Dimensions: ')
    print(dfglh.shape)
    print('PowerAPI Data Dimensions: ')
    print(dfpow.shape)

    df = pd.merge(dfglh, dfpow, on='url', how='inner')
    df.drop(columns=['url'], inplace=True)
    df = df.round(2)
    print('Common Joined Data Dimensions: ')
    print(df.shape)
    df = df.dropna()
    print('Dimensions after Missing Data Instances Removed: ')
    print(df.shape)
    dfDesc = df.describe().round(2).transpose()
    print(dfDesc)
    df.to_csv(filename+'.csv', sep=',', index=False)
    dfDesc.to_csv('Summary_'+filename+'.csv', sep=',')

def powerClustering(filename,powCol,logTransform):
    print(filename)
    df = pd.read_csv(filename+'.csv', delimiter=',')
    print('Original: ',df.shape)
    avg_powCol = df[powCol].mean()
    std_powCol = df[powCol].std()
    df = df[df[powCol] < avg_powCol + (5 * std_powCol)]
    print('Within5StdDev: ',df.shape)

    pow = [[x] for x in df[powCol]]
    clusters = KMeans(n_clusters=3, random_state=0, max_iter=1000).fit(pow)
    df[powCol+'Clusters3'] = clusters.labels_
    print('3 Power Clusters Distribution')
    print(dict(sorted(Counter(clusters.labels_).items())))

    if (logTransform):
        pow = [[np.log2(x)] for x in df[powCol]]
        print('Log Transformed Power')
        clusters = KMeans(n_clusters=3, random_state=0, max_iter=1000).fit(pow)
        df['LOG' + powCol + 'Clusters3'] = sortClusterLabelsBasedOnMinOf1DvaluesInClustersAndRelabel \
            (3, clusters.labels_, [[np.log2(x)] for x in df[powCol]])
        #df['LOG' + powCol + 'Clusters3'] = clusters.labels_
        print('3 Power Clusters Distribution')
        print(dict(sorted(Counter(clusters.labels_).items())))


    df.to_csv(filename+'_'+powCol+'Within5stdevClusters.csv', sep=',', index=False)


def sortClusterLabelsBasedOnMinOf1DvaluesInClustersAndRelabel(numClusters,labels,data1D):
    colClusters = [[] for j in range(numClusters)]
    for j in range(len(data1D)):
        colClusters[labels[j]].append(data1D[j][0])
    comparisonValues = [min(colClusters[j]) for j in range(numClusters)]
    clusterOrderAscending = [j for j in range(numClusters)]
    for x in range(1, numClusters):
        y = x - 1
        z = clusterOrderAscending[x]
        while y >= 0 and comparisonValues[clusterOrderAscending[y]] > comparisonValues[z]:
            clusterOrderAscending[y + 1] = clusterOrderAscending[y]
            y = y - 1
        clusterOrderAscending[y + 1] = z
    replacements = {x: clusterOrderAscending.index(x) + 1 for x in range(numClusters)}
    return [replacements.get(x, x) for x in labels]

def visualizeData():
    df = pd.read_csv('Data_LHScrPow_avgPowWithin5stdevClusters.csv', sep=',')
    powCol = 'avgPow'
    numPowClusters = 3
    powLabels = df['LOG' + powCol + 'Clusters' + str(numPowClusters)]
    #df['LOG' + powCol + 'OrdinalClusters' + str(
    #     numPowClusters)] = sortClusterLabelsBasedOnMinOf1DvaluesInClustersAndRelabel \
    #     (numPowClusters, powLabels, [[np.log2(x)] for x in df[powCol]])

    plt.figure(figsize=(5, 5))

    dfLow = df[df['LOG' + powCol + 'Clusters' + str(numPowClusters)] == 1]
    dfMed = df[df['LOG' + powCol + 'Clusters' + str(numPowClusters)] == 2]
    dfHig = df[df['LOG' + powCol + 'Clusters' + str(numPowClusters)] == 3]
    print(dfLow.shape)
    print(dfMed.shape)
    print(dfHig.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    dfLow.plot(kind="scatter", x="performance-score", y='avgPow', color="g", label="Low", marker="x", ax=ax)
    dfMed.plot(kind="scatter", x="performance-score", y='avgPow', label="Moderate", ax=ax, marker="o", color="none",
               edgecolor="r")
    dfHig.plot(kind="scatter", x="performance-score", y='avgPow', label="High", ax=ax, marker="^", color="none",
               edgecolor="b")

    # sns.scatterplot(
    #     x="performance-score", y="avgPow",
    #     hue='LOG' + powCol + 'Clusters' + str(numPowClusters),#'LOG' + powCol + 'OrdinalClusters' + str(numPowClusters),
    #     # size="avgPow",
    #     palette=sns.color_palette(n_colors=numPowClusters),
    #     # "muted",#sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True),#sns.color_palette("hls", 10),
    #     data=df,
    #     legend=True,
    #     alpha=0.8
    # )
    x = df['performance-score']
    m, b = np.polyfit(x, df['avgPow'], 1)
    plt.plot(x, m * x + b)
    plt.xlabel("Lighthouse Performance Score")
    plt.ylabel("Power Consumption (Watts)")
    #plt.title('LHScrPow_avgPowWithin5stdevClusters')
    plt.savefig('LHScrPow_avgPowWithin5stdevClusters.jpg', dpi=1200,bbox_inches="tight")
    plt.savefig('LHScrPow_avgPowWithin5stdevClusters.eps', dpi=1200,bbox_inches="tight")
    plt.savefig('LHScrPow_avgPowWithin5stdevClusters.png', dpi=1200,bbox_inches="tight")
    plt.savefig('LHScrPow_avgPowWithin5stdevClusters.svg', dpi=1200,bbox_inches="tight")
    plt.savefig('LHScrPow_avgPowWithin5stdevClusters.pgf', dpi=1200,bbox_inches="tight")
    #plt.show()
    plt.clf()

def StandardizeZscore(X_train,X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train,X_test

def takeLogBase2OfXplus2(x):
    return np.log2(x+2)

def reverseLog(x):
    return (2 ** x)-2

def classify(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = [accuracy_score(y_test, y_pred),precision_score(y_test, y_pred,average='macro'),
               recall_score(y_test, y_pred,average='macro'),f1_score(y_test, y_pred,average='macro')]
    conf = confusion_matrix(y_test, y_pred)
    return metrics,conf

runs_random = [389798,987646,123567,454687,546546,738383]
num_runs = 5
def main():
    #consolidateData('Data_LHScrPow')
    #powerClustering('Data_LHScrPow','avgPow',True)
    visualizeData()
    return



    df = pd.read_csv('Data_LHScrPow_avgPowWithin5stdevClusters.csv',sep=',')
    data = np.array(df['performance-score']).reshape(-1, 1)
    power = df['avgPow']
    labels = df['LOGavgPowClusters3']
    num_labels=3
    for algo in ['DT','RF','KNN','SVC']:
        metrics_sum = [0, 0, 0, 0]
        conf = [[0 for j in range(num_labels)] for i in range(0,num_labels) ]
        for i in range(num_runs):
            X_train, X_test, y_train, y_test = train_test_split\
               (data, labels, test_size=0.1, random_state=runs_random[i],stratify=labels)
            model = 0
            if algo == 'DT':
                model = DecisionTreeClassifier()
            if algo == 'RF':
                model = RandomForestClassifier(n_estimators=100)
            if algo == 'KNN':
                model = KNeighborsClassifier(n_neighbors=21, weights='distance')
            if algo == 'SVC':
                model = svm.SVC()
            metrics,confMat = classify(model, X_train, y_train, X_test, y_test)
            metrics_sum = [metrics_sum[i] + metrics[i] for i in range(len(metrics_sum))]
            conf = [[conf[i][j] + confMat[i][j] for j in range(num_labels)] for i in range(num_labels)]
        metrics = [x / num_runs for x in metrics_sum]
        conf = np.array([[conf[i][j] / num_runs for j in range(num_labels)] for i in range(num_labels)])

        print('Algo: ',algo, 'Acc: ',  metrics[0], 'Prec: ',metrics[1], 'Rec: ', metrics[2],'f1: ', metrics[3])
        print(conf)

    lr_rmse = 0
    lr_mae = 0
    lr_mape = 0
    for i in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(data, power, test_size=0.1)#,random_state=runs_random[i])
        X_train = np.vectorize(takeLogBase2OfXplus2)(X_train)
        X_test = np.vectorize(takeLogBase2OfXplus2)(X_test)
        #StandardizeZscore(X_train, X_test)
        y_train = np.vectorize(takeLogBase2OfXplus2)(y_train)

        reg = LinearRegression()

        reg = reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        y_pred = [round(y, 2) for y in y_pred]

        y_pred = np.vectorize(reverseLog)(y_pred)
        #y_train = np.vectorize(reverseLog)(y_train)
        y_test = list(y_test)
        # ae = [abs(y_pred[i] - y_test[i]) for i in range(0, len(y_test))]
        # ape = [abs(y_pred[i] - y_test[i]) / y_test[i] for i in range(0, len(y_test))]
        # d = pd.DataFrame()
        # # d['train_set'] = y_train
        # d['test_set'] = y_test
        # d['pred_val'] = y_pred
        # d['abs err'] = ae
        # d['abs % err'] = ape
        # np.set_printoptions(threshold=np.inf)
        # pd.set_option('display.max_columns', None)
        # print(d.describe().round(1).transpose())
        # percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        # for i in range(len(percentiles)):
        #     ae_p = np.percentile(ae, percentiles[i]).round(1)
        #     y_test_ae_p = [y_test[i] for i in range(len(y_test)) if abs(y_pred[i] - y_test[i]) <= ae_p]
        #     y_pred_ae_p = [y_pred[i] for i in range(len(y_pred)) if abs(y_pred[i] - y_test[i]) <= ae_p]
        #     print('Abs Err < ', ae_p, 'for ', percentiles[i], '%  of sites with')
        #     d = pd.DataFrame()
        #     d['testVals'] = y_test_ae_p
        #     d['predVals'] = y_pred_ae_p
        #     print(d.describe().round(1).transpose())
        #
        # for i in range(len(percentiles)):
        #     ape_p = np.percentile(ape, percentiles[i]).round(1)
        #     y_test_ape_p = [y_test[i] for i in range(len(y_test)) if abs(y_pred[i] - y_test[i]) / y_test[i] <= ape_p]
        #     y_pred_ape_p = [y_pred[i] for i in range(len(y_pred)) if abs(y_pred[i] - y_test[i]) / y_test[i] <= ape_p]
        #     print('Abs % Err < ', ape_p, 'for ', percentiles[i], '%  of sites with')
        #     d = pd.DataFrame()
        #     d['testVal'] = y_test_ape_p
        #     d['predVal'] = y_pred_ape_p
        #     print(d.describe().round(1).transpose())

        lr_rmse += mean_squared_error(y_test, y_pred, squared=False)
        lr_mae += mean_absolute_error(y_test, y_pred)
        lr_mape += mean_absolute_percentage_error(y_test, y_pred)

    lr_rmse /= num_runs
    lr_mae /= num_runs
    lr_mape /= num_runs

    print('RMSE: ',lr_rmse)
    print('MAE: ',lr_mae)
    print('MAPE: ',lr_mape)

    print('Correlation: ', df.corr(method='spearman')['avgPow']['performance-score'])
    power = [takeLogBase2OfXplus2(x) for x in df['avgPow']]
    reg = LinearRegression()
    reg = reg.fit(data, power)
    pred = reg.predict(data)
    r2 = r2_score(power, pred)
    adjusted_r2 = 1 - (((len(power) - 1) / (len(power) - 1 - data.shape[1])) * r2)
    print('R2: ', r2)
    print('Adjusted R2: ', adjusted_r2)

if __name__ == '__main__':
    main()
