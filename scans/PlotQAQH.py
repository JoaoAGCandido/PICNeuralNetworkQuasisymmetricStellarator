import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qsc import Qsc

print("\n")

#read file
#df=pd.read_csv("scan7/scan7Clean.csv.zip")
df=pd.read_csv("scan7/scan7.csv.zip")


colors=["","red", "orange", "goldenrod", "green", "darkgreen", "blue", "darkblue", "purple" ]
colorsQA = ["", "sienna", "lightsalmon", "gold", "palegreen",
                  "darkseagreen", "skyblue", "steelblue", "orchid"]

#df = df[df['rc1'] == -df['zs1']]
for nfp in np.arange(1,9,1):
    fig, ax = plt.subplots()

    newdf = df[df['nfp'] == nfp]
    #dataframe that stores QA stel with a certain nfp  
    dfQA = newdf[newdf['heli'] == False]
    #dataframe that stores QH stel with a certain nfp  
    dfQH = newdf[newdf['heli'] == True]
    delta = 1. / (nfp*nfp +1)
    #plt.axline([0, -delta], [0.3, -delta], color='black')
    plt.axline([delta, 0], [delta, -0.3], color='black')
    #plot with color representing nfp and ^ marker representing Qh 
    ax.scatter(dfQA['rc1'], dfQA['zs1'], c = colorsQA[nfp], s = 15, linewidths = 0, edgecolors="black", label="{}".format(nfp) + " QA")
    ax.scatter(dfQH['rc1'], dfQH['zs1'], c = colors[nfp], s = 15, linewidths = 0, edgecolors="black", label="{}".format(nfp) + " QH", marker='^')

    #plot labels, legend
    plt.xlim(delta-0.0005, delta+0.0005)
    plt.xlabel('rc1')
    plt.ylabel('zs1')
    ax.legend()
    plt.show()