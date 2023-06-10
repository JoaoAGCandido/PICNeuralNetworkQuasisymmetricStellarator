import argparse
import numpy as np
import pandas as pd
from qsc import Qsc
import matplotlib.pyplot as plt

# part of the radial component of the axis
rc0 = 1


# arrays to store output
out = {
    'axLenght': [],
    'RotTrans': [],
    'nfp': [],
    'heli': [],
    'rc1': [],
    'zs1': [],
    'eta': [],
    'max_elong': [],
    'LgradB': [],
    'min_R0': []
}


df = pd.read_csv("scan7/scan7Clean.csv.zip")
df = df[df['nfp'] >= 4]
df = df[df['heli'] == False]

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
    ax.scatter(dfQA["axLenght"], dfQA["RotTrans"], c=colors[nfp], s=15,
                linewidths=0, edgecolors="black", label="{}".format(nfp) + " QA")
    ax.scatter(dfQH["axLenght"], dfQH["RotTrans"], c=colors[nfp], s=15,
                linewidths=0, edgecolors="black", label="{}".format(nfp) + " QH", marker='^')

plt.ylabel("Rotational Transform")
plt.xlabel("Axis Length")
ax.legend()
# plt.ylim(0, 0.5)
#plt.savefig("Plots/" + yFeat + dataSet + ".png")
plt.show()

print("number of points: ", len(df.index))
#pd.set_option('display.max_rows', 30)
print("find good stel")
dfNew = df[df['nfp'] == 4]
dfNew = dfNew[dfNew['RotTrans'] > 1]
dfNew = dfNew[dfNew['max_elong'] < 6]
print("number of points: ", len(dfNew.index))
print(dfNew.sort_values("RotTrans", ascending=False, inplace=True))

i=0
rc1=round(dfNew["rc1"].iloc[i],3)
zs1=round(dfNew["zs1"].iloc[i],3)
eta=round(dfNew["eta"].iloc[i],3)
stel = Qsc(rc=[rc0, rc1], zs=[0, zs1], nfp=dfNew["nfp"].iloc[i], etabar=eta)
stel.plot_boundary()

print("nfp",dfNew["nfp"].iloc[i])
print("rc1", rc1)
print("zs1", zs1)
print("eta", eta)
print("max elong", stel.max_elongation)
print("lgradB",min(stel.L_grad_B))
print("minR0", stel.min_R0)
print("axis_len",stel.axis_length / 2 / np.pi / rc0)
print("rotTrans", stel.iota)
# check if stel is QA <- heli=False or QH <- heli=True
print("heli", False if stel.helicity == 0 else True)
