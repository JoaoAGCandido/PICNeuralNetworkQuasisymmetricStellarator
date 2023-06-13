import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qsc import Qsc

print("\n")

# read file
df = pd.read_csv("scan7/scan7Clean.csv.zip")
df = df[df['rc1'] != 0]
df = df[df['zs1'] != 0]
df = df.sample(frac=0.05)

colors = ["", "red", "orange", "goldenrod", "green",
          "darkgreen", "blue", "darkblue", "purple"]
colorsQA = ["", "sienna", "lightsalmon", "gold", "palegreen",
            "darkseagreen", "skyblue", "steelblue", "orchid"]


dfOverlay = pd.read_csv("scan7/overlay.csv")
fig, ax = plt.subplots()
for nfp in np.arange(8, 0, -1):
    newdf = df[df['nfp'] == nfp]
    #fig, ax = plt.subplots()
    # dataframe that stores QA stel with a certain nfp
    dfQA = newdf[newdf['heli'] == False]
    # dataframe that stores QH stel with a certain nfp
    dfQH = newdf[newdf['heli'] == True]
            
    # plot with color representing nfp and ^ marker representing Qh
    if len(dfQH.index):
        ax.scatter(dfQH['axLenght'], dfQH['RotTrans'], c=colors[nfp], s=15,
                linewidths=0, edgecolors="black", label="{}".format(nfp) + " QH", marker='^')
    if len(dfQA.index):
        ax.scatter(dfQA['axLenght'], dfQA['RotTrans'], c=colorsQA[nfp], s=15,
                linewidths=0, edgecolors="black", label="{}".format(nfp) + " QA")

ax.scatter(dfOverlay['axLenght'], dfOverlay['RotTrans'], c="red", s=15,
        linewidths=0, edgecolors="black", label="Overlay", marker="D")
# plot labels, legend
plt.ylabel("Rotational Transform")
plt.xlabel("Axis Length")

ax.legend()
#plt.ylim(0, 4)
#plt.xlim(0.97, 1.1)
plt.savefig("Plots/overlay.png")

plt.show()