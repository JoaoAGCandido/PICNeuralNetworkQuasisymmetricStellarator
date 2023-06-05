import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# read file
for dataSet in ["Dirty", "Clean"]:
    if dataSet == "Clean":
        df = pd.read_csv("scan7/scan7Clean.csv.zip")
    else:
        df = pd.read_csv("scan7/scan7.csv.zip")

    xFeat = 'axLenght'
    # yFeat = 'RotTrans'
    # yFeat = 'max_elong'

    # plot distribution of values for ax_lenght
    # Create a histogram
    plt.hist(df["axLenght"], bins=20, alpha=0.1, label="axLenght")

    # Add labels and title
    plt.xlabel("axLengt")
    plt.ylabel('Frequency')
    plt.title('Distribution of ' + "axLenght" + " " + dataSet)

    # Show the histogram
    plt.show()

    for yFeat in ['max_elong', 'RotTrans']:
        # using sublot for each color and to differentiate QA from QH
        fig, ax = plt.subplots()

        colors = ["", "red", "darkorange", "goldenrod", "green",
                  "darkgreen", "blue", "darkblue", "purple"]

        for nfp in np.arange(8, 0, -1):
            newdf = df[df['nfp'] == nfp]
            # dataframe that stores QA stel with a certain nfp
            dfQA = newdf[newdf['heli'] == False]
            # dataframe that stores QH stel with a certain nfp
            dfQH = newdf[newdf['heli'] == True]

            # plot with color representing nfp and ^ marker representing Qh
            ax.scatter(dfQA[xFeat], dfQA[yFeat], c=colors[nfp], s=15,
                       linewidths=0, edgecolors="black", label="{}".format(nfp) + " QA")
            ax.scatter(dfQH[xFeat], dfQH[yFeat], c=colors[nfp], s=15,
                       linewidths=0, edgecolors="black", label="{}".format(nfp) + " QH", marker='^')

        # plot labels, legend
        plt.xlabel(xFeat)
        plt.ylabel(yFeat)
        ax.legend()
        # plt.ylim(0, 0.5)
        plt.title(yFeat + " " + dataSet)
        plt.savefig("Plots/" + yFeat + dataSet + ".png")

        plt.show()

        # plot distribution of values
        # Create a histogram
        plt.hist(df[yFeat], bins=20, alpha=0.1, label=yFeat)

        # Add labels and title
        plt.xlabel(yFeat)
        plt.ylabel('Frequency')
        plt.title('Distribution of ' + yFeat + " " + dataSet)

        # Show the histogram
        plt.show()
