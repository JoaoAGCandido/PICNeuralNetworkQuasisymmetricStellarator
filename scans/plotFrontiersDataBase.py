import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# read file
dataSet = "Clean"
df = pd.read_csv("scan7/scan7Clean.csv.zip")

xFeat = "axLenght"


for yFeat in ['max_elong', 'RotTrans']:
    # using sublot for each color and to differentiate QA from QH
    # fig, ax = plt.subplots()

    colors = ["", "red", "darkorange", "goldenrod", "green",
              "darkgreen", "blue", "darkblue", "purple"]

    for nfp in np.arange(8, 0, -1):
        fig, ax = plt.subplots()
        newdf = df[df['nfp'] == nfp]
        # dataframe that stores QA stel with a certain nfp
        dfQA = newdf[newdf['heli'] == False]
        # dataframe that stores QH stel with a certain nfp
        dfQH = newdf[newdf['heli'] == True]
        # dataFrame that stores values on boundary of rc1 QA
        dfRc1QA = dfQA[dfQA['rc1'] == 0.3]
        # dataFrame that stores values on boundary of zs1 QA
        dfZs1QA = dfQA[dfQA['zs1'] == -0.3]
        # dataFrame that stores values on boundary of eta QA
        dfEtaQA = dfQA[dfQA['eta'] == -3]
        # dataFrame that stores values on boundary of rc1 QH
        dfRc1QH = dfQH[dfQH['rc1'] == 0.3]
        # dataFrame that stores values on boundary of zs1 QH
        dfZs1QH = dfQH[dfQH['zs1'] == -0.3]
        # dataFrame that stores values on boundary of eta QH
        dfEtaQH = dfQH[dfQH['eta'] == -3]

        # plot with color representing nfp and ^ marker representing Qh
        ax.scatter(dfQA[xFeat], dfQA[yFeat], c=colors[nfp], s=15,
                   linewidths=0, edgecolors="black", label="{}".format(nfp) + " QA")
        ax.scatter(dfQH[xFeat], dfQH[yFeat], c=colors[nfp], s=15,
                   linewidths=0, edgecolors="black", label="{}".format(nfp) + " QH", marker='^')
        ax.scatter(dfRc1QA[xFeat], dfRc1QA[yFeat], c="black", s=15,
                   linewidths=0, edgecolors="black", label="min eta QA")
        ax.scatter(dfZs1QA[xFeat], dfZs1QA[yFeat], c="dimgray", s=15,
                   linewidths=0, edgecolors="black", label="max rc1 QA")
        ax.scatter(dfEtaQA[xFeat], dfEtaQA[yFeat], c="silver", s=15,
                   linewidths=0, edgecolors="black", label="min eta QA")
        ax.scatter(dfRc1QH[xFeat], dfRc1QH[yFeat], c="black", s=15,
                   linewidths=0, edgecolors="black", label="min eta QH", marker='^')
        ax.scatter(dfZs1QH[xFeat], dfZs1QH[yFeat], c="dimgray", s=15,
                   linewidths=0, edgecolors="black", label="max rc1 QH", marker='^')
        ax.scatter(dfEtaQH[xFeat], dfEtaQH[yFeat], c="silver", s=15,
                   linewidths=0, edgecolors="black", label="min eta QA", marker='^')
        # plot labels, legend
        plt.xlabel(xFeat)
        plt.ylabel(yFeat)
        ax.legend()
        # plt.ylim(0, 0.5)
        plt.title(yFeat + " " + dataSet)
        # plt.savefig("Plots/" + yFeat + dataSet + ".png")

        plt.show()
