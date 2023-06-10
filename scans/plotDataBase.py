import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# read file
for dataSet in ["Complete", "Clean"]:#["Clean"]:#
    if dataSet == "Clean":
        df = pd.read_csv("scan7/scan7Clean.csv.zip")
        #df = df.sample(frac=0.1)
    else:
        df = pd.read_csv("scan7/scan7.csv.zip")
        #df = df.sample(frac=0.01)

    xFeat = 'axLenght'
    # yFeat = 'RotTrans'
    # yFeat = 'max_elong'
    """
    # plot distribution of values for ax_lenght
    # Create a histogram
    plt.hist(df["axLenght"], bins=20, alpha=0.1, label="axLength")

    # Add labels and title
    plt.xlabel("Axis Length")
    plt.ylabel('Frequency')
    plt.title('Distribution of ' + "axLength" + " " + dataSet + "Data Set")

    # Show the histogram
    plt.show()"""

    for yFeat in ['RotTrans', 'max_elong']:
        # using sublot for each color and to differentiate QA from QH
        fig, ax = plt.subplots()

        colors = ["", "red", "darkorange", "goldenrod", "green",
                  "darkgreen", "blue", "darkblue", "purple"]
        colorsQA = ["", "sienna", "lightsalmon", "gold", "palegreen",
                  "darkseagreen", "skyblue", "steelblue", "orchid"]
        
        for nfp in np.arange(8, 0, -1):
            newdf = df[df['nfp'] == nfp]
            #fig, ax = plt.subplots()
            # dataframe that stores QA stel with a certain nfp
            dfQA = newdf[newdf['heli'] == False]
            # dataframe that stores QH stel with a certain nfp
            dfQH = newdf[newdf['heli'] == True]
                    
            # plot with color representing nfp and ^ marker representing Qh
            if len(dfQH.index):
                ax.scatter(dfQH[xFeat], dfQH[yFeat], c=colors[nfp], s=15,
                        linewidths=0, edgecolors="black", label="{}".format(nfp) + " QH", marker='^')
            if len(dfQA.index):
                ax.scatter(dfQA[xFeat], dfQA[yFeat], c=colorsQA[nfp], s=15,
                        linewidths=0, edgecolors="black", label="{}".format(nfp) + " QA")
 
            """ 
            #plot without distinguishing QA/QH
            if len(newdf.index):
                ax.scatter(newdf[xFeat], newdf[yFeat], c=colors[nfp], s=15,
                            linewidths=0, edgecolors="black", label="{}".format(nfp) + " nfp")
            """
        # plot labels, legend
        if xFeat=="axLenght":
            plt.xlabel("Axis Length")
        if yFeat=="max_elong":
            plt.ylabel("Max Elongation")
            #plt.title("Max Elongation " + dataSet + " Dataset")
        if yFeat=="RotTrans":
            plt.ylabel("Rotational Transform")
            #plt.title("Rotational Transform " + dataSet + " Dataset")
        ax.legend()
        #plt.ylim(0, 4)
        #plt.xlim(0.97, 1.1)
        plt.savefig("Plots/" + yFeat + dataSet + "QAQH" + ".png")

        plt.show()
    """
        # plot distribution of values
        # Create a histogram
        plt.hist(df[yFeat], bins=20, alpha=0.1, label=yFeat)

        # Add labels and title
        plt.xlabel(yFeat)
        plt.ylabel('Frequency')
        plt.title('Distribution of ' + yFeat + " " + dataSet)

        # Show the histogram
        plt.show()
"""
