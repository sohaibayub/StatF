
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from matplotlib.colors import ListedColormap
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from  collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from  collections import Counter
from sklearn.manifold import TSNE
import random

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

def boxplotPowDistBy10IndFeatCLusters(topNum,topCols,df,powCol):
    fig, axes = plt.subplots(2, 5, figsize=(50, 500))
    axes = axes.flatten()
    for i in range(topNum):
        iCol = [[x] for x in df[topCols[i]]]
        clustering = KMeans(n_clusters=3, random_state=1, max_iter=1000).fit(iCol)
        colLabels = clustering.labels_
        colLabels = sortClusterLabelsBasedOnMinOf1DvaluesInClustersAndRelabel(3, colLabels, iCol)
        d = {1: [], 2: [], 3: []}
        for j in range(len(colLabels)):
            d[colLabels[j]].append(df[powCol][j])
        # print(d)

        axes[i].boxplot(d.values())
        if i > 0:
            axes[i].set_yticks([])
        axes[i].set_xticklabels(d.keys())
        axes[i].set_xlabel(topCols[i])

    fig.text(0.5, 0.04, '\nOrdinal Clusters of Websites by Individual Feature', ha='center', va='center')
    fig.text(0.06, 0.5, 'Power Consumption (watts)', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.94,
             'Distribution of Power Consumption of Web Apps Grouped Individually By Top ' + str(topNum) + ' Features ',
             ha='center', va='center')
    plt.show()
    plt.clf()

def tsnePlot(topNum,topCols,perplexity,powCol,numPowClusters,df):
    topCols = topCols[:topNum]
    random.shuffle(topCols)
    feats = StandardScaler().fit_transform(df[topCols].to_numpy())
    powLabels = df['LOG' + powCol + 'Clusters' + str(numPowClusters)]
    df['LOG' + powCol + 'OrdinalClusters' + str(
        numPowClusters)] = sortClusterLabelsBasedOnMinOf1DvaluesInClustersAndRelabel \
        (numPowClusters, powLabels, [[np.log2(x)] for x in df[powCol]])
    for i in range(len(perplexity)):
        tsne = TSNE(n_components=2, method='exact', perplexity=perplexity[i], learning_rate='auto', n_iter=1000)
        tsne_results = tsne.fit_transform(feats)
        df['tsne-2d-one'] = tsne_results[:, 0]
        df['tsne-2d-two'] = tsne_results[:, 1]
        plt.figure(figsize=(5, 5))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue='LOG' + powCol + 'OrdinalClusters' + str(numPowClusters),
            # size="avgPow",
            palette=sns.color_palette(n_colors=numPowClusters),
            # "muted",#sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True),#sns.color_palette("hls", 10),
            data=df,
            legend=False,
            alpha=0.8
        )
        plt.title(
            powCol + 'Clusters' + str(numPowClusters) + '_UsingTopFeats' + str(topNum) + '_tSNEperplexity' + str(
                perplexity[i]))  # +'_avgPowStdev'+s)
        plt.savefig('tsnePlots/' +  powCol + 'Clusters' + str(numPowClusters) + '_tSNEperplexity' + str(
             perplexity[i]) + '_topFeats'+str(topNum)+'.png')

        #plt.show()
        plt.clf()
    return


def tsnePlotByMarker(perplexity,powCol,numPowClusters,df):
    #topCols = topCols[:topNum]
    #random.shuffle(topCols)
    cols = list(df.columns)
    cols = cols[:48]#+ [powCol, powCol + 'Clusters' + str(numPowClusters)]#ADDING THESE COLS IS WRONG
    print(len(cols))
    feats = StandardScaler().fit_transform(df[cols].to_numpy())
    #feats = df[cols].to_numpy()
    powLabels = df['LOG' +powCol + 'Clusters' + str(numPowClusters)]
    df['LOG'+powCol + 'OrdinalClusters' + str(
        numPowClusters)] = sortClusterLabelsBasedOnMinOf1DvaluesInClustersAndRelabel \
        (numPowClusters, powLabels, [[np.log2(x)] for x in df[powCol]])

    #TO DO: separate out rows of 0,1,2 clusters and plot each separately on same axis as done for corrWithPow


    for i in range(len(perplexity)):
        tsne = TSNE(n_components=2, method='exact', perplexity=perplexity[i], n_iter=1000,learning_rate='auto')#,#method='exact',
        tsne_results = tsne.fit_transform(feats)
        df['tsne-2d-one'] = tsne_results[:, 0]
        df['tsne-2d-two'] = tsne_results[:, 1]
        dfLow = df[df['LOG'+ powCol + 'OrdinalClusters' + str(numPowClusters)] == 1]
        dfMed = df[df['LOG'+powCol + 'OrdinalClusters' + str(numPowClusters)] == 2]
        dfHig = df[df['LOG'+powCol + 'OrdinalClusters' + str(numPowClusters)] == 3]
        print(dfLow.shape)
        print(dfMed.shape)
        print(dfHig.shape)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        dfLow.plot(kind="scatter", x="tsne-2d-one", y='tsne-2d-two', color="g", label="Low", marker="x", ax=ax)
        dfMed.plot(kind="scatter", x="tsne-2d-one", y='tsne-2d-two', label="Moderate", ax=ax, marker="o", color="none",   edgecolor="r")
        dfHig.plot(kind="scatter", x="tsne-2d-one", y='tsne-2d-two', label="High", ax=ax, marker="^", color="none",  edgecolor="b")

        plt.yticks(size=10)
        plt.legend(fontsize=10)
        plt.xlabel("")
        plt.ylabel("")
        #plt.title(
        #    'tsnePlotMarkers_LOG' +  powCol + 'Clusters' + str(numPowClusters) + '_tSNEperplexity' + str(
        #     perplexity[i]))
        plt.savefig('tsnePlotsMarker/LOG' +  powCol + 'Clusters' + str(numPowClusters) + '_tSNEperplexity' + str(
             perplexity[i])+ '_WithPowerValAndClusterLabel.jpg', dpi=1200)
        plt.savefig('tsnePlotsMarker/LOG' +  powCol + 'Clusters' + str(numPowClusters) + '_tSNEperplexity' + str(
             perplexity[i])+ '_WithPowerValAndClusterLabel.pgf', dpi=1200)
        #plt.savefig('tsnePlotsMarker/LOG' + powCol + 'Clusters' + str(numPowClusters) + '_tSNEperplexity' + str(
        #    perplexity[i]) + '.pgf',dpi=1200)
        #plt.show()WithPowerValAndClusterLabel
        plt.clf()
    return


def boxplotPowDistBy5IndFeatCLusters(topNum,topCols,df,powCol):

    #fig = plt.figure(figsize=(10, 5))
    #axes = plt.subplots(1,5,1)
    fig, axes = plt.subplots(1, 5, figsize=(10, 5))
    axes = axes.flatten()
    for i in range(topNum):
        iCol = [[x] for x in df[topCols[i]]]
        clustering = KMeans(n_clusters=3, random_state=1, max_iter=1000).fit(iCol)
        colLabels = clustering.labels_
        colLabels = sortClusterLabelsBasedOnMinOf1DvaluesInClustersAndRelabel(3, colLabels, iCol)
        d = {1: [], 2: [], 3: []}
        for j in range(len(colLabels)):
            d[colLabels[j]].append(df[powCol][j])
        # print(d)

        axes[i].boxplot(d.values())
        if i > 0:
            axes[i].set_yticks([])
        axes[i].set_xticklabels(['$C_1$','$C_2$','$C_3$'])
        axes[i].set_xlabel(topCols[i])

    #fig.text(0.5, 0.04, '\nOrdinal Clusters of Websites by Individual Feature', ha='center', va='center')
    fig.text(0.06, 0.5, 'Power Consumption (watts)', ha='center', va='center', rotation='vertical')

    plt.savefig("FeatBasedDistOfPow.jpg",dpi=1200,bbox_inches="tight")
    plt.savefig("FeatBasedDistOfPow.pgf",dpi=1200,bbox_inches="tight")
    #fig.text(0.5, 0.94,
    #         'Distribution of Power Consumption of Web Apps Grouped Individually By Top ' + str(topNum) + ' Features ',
    #         ha='center', va='center')
    plt.show()
    plt.clf()

def main():
    topCols = open('SingleFeatureBasedClustering/FeatureListByDescendingImportanceRFacc72.txt', "r").read().split('\n')
    top5colsByCorr = ['requests', 'gzip_total', 'uses-long-cache-ttl', 'bytesIn', 'domElements']
    powCol = 'avgPow'
    numPowClusters = 3
    #df = pd.read_csv(r'StaticFeats_' + powCol + 'Clusters.csv', sep=',')
    df= pd.read_csv(r'StaticFeats_' + powCol + 'Within5stdev_'+powCol + 'Clusters.csv', sep=',')

    tsnePlotByMarker([45],powCol,numPowClusters,df)

    #boxplotPowDistBy5IndFeatCLusters(len(top5colsByCorr),top5colsByCorr,df,powCol)
    return


    topCols = topCols + [powCol, powCol + 'Clusters' + str(numPowClusters)]
    topCols = topCols[:topNum]
    perplexity = [30]
    topCols = topCols + [powCol, powCol + 'Clusters' + str(numPowClusters)]
    feats = StandardScaler().fit_transform(df[topCols].to_numpy())
    powLabels = df['LOG' + powCol + 'Clusters' + str(numPowClusters)]
    df['LOG' + powCol + 'OrdinalClusters' + str(numPowClusters)] = sortClusterLabelsBasedOnMinOf1DvaluesInClustersAndRelabel\
        (numPowClusters, powLabels,[[np.log2(x)] for x in df[powCol]])
    for i in range(len(perplexity)):
        tsne = TSNE(n_components=2, method='exact', perplexity=perplexity[i], learning_rate='auto', n_iter=1000)
        tsne_results = tsne.fit_transform(feats)
        df['tsne-2d-one'] = tsne_results[:, 0]
        df['tsne-2d-two'] = tsne_results[:, 1]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue='LOG' + powCol + 'OrdinalClusters' + str(numPowClusters),
            # size="avgPow",
            palette=sns.color_palette(n_colors=numPowClusters),
            # "muted",#sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True),#sns.color_palette("hls", 10),
            data=df,
            legend="auto",
            alpha=0.8
        )
        plt.title(
            powCol + 'Clusters' + str(numPowClusters) + '_UsingTopFeats'+str(topNum)+'_tSNEperplexity' + str(perplexity[i]))  # +'_avgPowStdev'+s)
        # plt.savefig('tSNEplots/stdev' + s + '/' + pow_col + 'Clusters' + str(c) + '_tSNEperplexity' + str(
        #     perplexity[i]) + '.png')

        plt.show()
        plt.clf()
    return

    fig, axes = plt.subplots(2, 5, figsize=(50, 500))
    axes = axes.flatten()
    for i in range(0,topNum):
        scaler = StandardScaler()
        topiCols = topCols[:i+3]
        featCols = df[topiCols].to_numpy()
        featCols = scaler.fit_transform(featCols)
        clustering = KMeans(n_clusters=3, random_state=1, max_iter=1000).fit(featCols)
        featLabels = clustering.labels_
        d = {0:[], 1:[], 2:[]}
        for j in range(len(featLabels)):
            d[featLabels[j]].append(df[powCol][j])
        #print(d)
        axes[i].boxplot(d.values())
        axes[i].set_xticklabels(d.keys())
        axes[i].set_xlabel('Top '+ str(i+2))
    fig.text(0.5, 0.04, '\n Clusters of Web Apps by Top k Features', ha='center', va='center')
    fig.text(0.06, 0.5, 'Power Consumption (watts)', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.94,
             'Distribution of Power Consumption of Web Apps Grouped Jointly By Top k Features ',
             ha='center', va='center')
    plt.show()
    plt.clf()


    return




    for i in range(len(topCols)):
        iCol = [[x] for x in df[topCols[i]]]
        clustering = KMeans(n_clusters=3, random_state=1, max_iter=1000).fit(iCol)
        colLabels = clustering.labels_
        colLabels = sortClusterLabelsBasedOnMinOf1DvaluesInClustersAndRelabel(3, colLabels, iCol)
        plt.scatter(colLabels, df[powCol])
        plt.xlabel('Ordinal Clusters of ' + topCols[i])
        plt.ylabel('Power Consumption')
        plt.show()
        plt.clf()

    return
    fig = plt.figure()
    ax = fig.add_subplot(111 )
    x = np.array(df[topCols[0]])
    y = np.array(df[topCols[1]])
    #colorsList = [(tuple rgb color 1), (tuple rgb color 2), (tuple rgb color 3)]
    #CustomCmap = plt.colors.ListedColormap(colorsList)
    ax.scatter(x, y, c=df['LOG' + powCol + 'Clusters' + str(numPowClusters)], cmap="RdBu")

    plt.show()
    return




if __name__ == '__main__':
    main()
