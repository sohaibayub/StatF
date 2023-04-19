import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def corrWithPower():
#     df_result = pd.DataFrame()
#     df = pd.read_csv('StaticFeats.csv', sep=',').dropna()
#     df_result['Feature'] = df.columns #‘pearson’, ‘kendall’, ‘spearman’
#     df_result['Pearson'] = df.corr(method='pearson')['avgPow'].values
#     df_result['Kendall'] = df.corr(method='kendall')['avgPow'].values
#     df_result['Spearman'] = df.corr(method='spearman')['avgPow'].values
#
#     df_result.to_csv('CorrelationWithPower.csv',sep=',', index=False)

    df_result = pd.read_csv('CorrelationWithPower.csv', sep=',')
    #ylabels.reverse()
    df_result.sort_values(by="Spearman", ascending=False, inplace=True)
    df_result['FeatNum'] = range(df_result.shape[0]-1,-1,-1)
    ylabels = [x for x in df_result['Feature']]

    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111)
    df_result.plot(kind="scatter", x="Kendall", y='FeatNum', color="g", label="Kendall",  marker="x", ax=ax)
    df_result.plot(kind="scatter", x="Spearman", y='FeatNum', label="Spearman", ax=ax, marker="o", color="none", edgecolor="r")
    df_result.plot(kind="scatter", x="Pearson", y='FeatNum',  label="Pearson", ax=ax,marker="s", color="none", edgecolor="b" )
    plt.yticks(ticks = df_result['FeatNum'], labels =ylabels,size=8)
    plt.grid(axis="y", which = 'both', linestyle='dotted', linewidth=0.3)
    ax.set_xlabel("Correlation with Power Consumption")
    ax.set_ylabel("Website Feature")
    ax.tick_params(axis='y', which='major')
    plt.rcParams['figure.dpi'] = 1600
    plt.savefig("corrWithPowDPI12k.png",dpi=1200,bbox_inches="tight")
    plt.savefig("corrWithPowDPI12k.eps",dpi=1200,bbox_inches="tight")
    plt.savefig("corrWithPowDPI12k.jpg",dpi=1200,bbox_inches="tight")
    plt.savefig("corrWithPowDPI12k.svg",dpi=1200,bbox_inches="tight")
    plt.savefig("corrWithPowDPI12k.pgf",dpi=1200,bbox_inches="tight")
    #plt.show()
    return

if __name__ == '__main__':
    corrWithPower()