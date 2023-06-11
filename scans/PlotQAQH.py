import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qsc import Qsc

print("\n")

#read file
#df=pd.read_csv("scan7/scan7Clean.csv.zip")
df=pd.read_csv("scan7/scan7.csv.zip")
df = df[df['rc1'] != 0]
df = df[df['zs1'] != 0]

colors=["","red", "orange", "goldenrod", "green", "darkgreen", "blue", "darkblue", "purple" ]
colorsQA = ["", "sienna", "lightsalmon", "gold", "palegreen",
                  "darkseagreen", "skyblue", "steelblue", "orchid"]


for nfp in np.arange(1,9,1):
    fig, ax = plt.subplots()

    newdf = df[df['nfp'] == nfp]
    #dataframe that stores QA stel with a certain nfp  
    dfQA = newdf[newdf['heli'] == False]
    #dataframe that stores QH stel with a certain nfp  
    dfQH = newdf[newdf['heli'] == True]
    delta = 1. / (nfp*nfp + 1)
    print("nfp:", nfp)
    print("delta", delta)
    dfWrongQA = dfQA[dfQA["rc1"] > delta]
    print("num wrong QA:",len(dfWrongQA.index))
    #print(dfWrongQA.sort_values("rc1"))
    dfWrongQH = dfQH[dfQH["rc1"] < delta]
    print("num wrong QH:",len(dfWrongQH.index))


    #recalculate wrong QA with higher resolution
    out ={
        "rc1" : [],
        "zs1" : [],
        "heli" : []
    }
    for i in range(len(dfWrongQA.index)):
        rc1=dfWrongQA["rc1"].iloc[i]
        zs1=dfWrongQA["zs1"].iloc[i]
        eta=dfWrongQA["eta"].iloc[i]
        stel = Qsc(rc=[1, rc1], zs=[0, zs1], nfp=dfWrongQA["nfp"].iloc[i], etabar=eta, nphi=100)
        out['heli'].append(
            False if stel.helicity == 0 else True)
        out['rc1'].append(rc1)
        out['zs1'].append(zs1)
    dfCorrectedQA = pd.DataFrame(out)
    dfWrongQA = dfCorrectedQA[dfCorrectedQA['heli'] == False]
    dfCorrectedQA = dfCorrectedQA[dfCorrectedQA['heli'] == True]
    

    #recalculate wrong QH with higher resolution
    out ={
        "rc1" : [],
        "zs1" : [],
        "heli" : []
    }
    for i in range(len(dfWrongQH.index)):
        rc1=dfWrongQH["rc1"].iloc[i]
        zs1=dfWrongQH["zs1"].iloc[i]
        eta=dfWrongQH["eta"].iloc[i]
        stel = Qsc(rc=[1, rc1], zs=[0, zs1], nfp=dfWrongQH["nfp"].iloc[i], etabar=eta, nphi=100)
        out['heli'].append(
            False if stel.helicity == 0 else True)
        out['rc1'].append(rc1)
        out['zs1'].append(zs1)
    dfCorrectedQH = pd.DataFrame(out)
    dfWrongQH = dfCorrectedQH[dfCorrectedQH['heli'] == True]
    dfCorrectedQH = dfCorrectedQH[dfCorrectedQH['heli'] == False]


    #theoric divisory line between QA and QH
    plt.axline([delta, 0], [delta, -0.3], color='black')
    #plot with color representing nfp and ^ marker representing Qh 
    if (len(dfQA.index)):
        ax.scatter(dfQA['rc1'], dfQA['zs1'], c = colorsQA[3], s = 15, linewidths = 0, edgecolors="black", label="{}".format(nfp) + " QA")
    if (len(dfQH.index)): 
        ax.scatter(dfQH['rc1'], dfQH['zs1'], c = colors[3], s = 15, linewidths = 0, edgecolors="black", label="{}".format(nfp) + " QH", marker='^')
    if (len(dfWrongQA.index)):
        ax.scatter(dfWrongQA['rc1'], dfWrongQA['zs1'], c = "red", s = 15, linewidths = 0, edgecolors="black", label="Incorrect")
    if (len(dfCorrectedQA.index)): 
        ax.scatter(dfCorrectedQA['rc1'], dfCorrectedQA['zs1'], c = "green", s = 15, linewidths = 0, edgecolors="black", label="Corrected", marker='^')
    if (len(dfWrongQH.index)):
        ax.scatter(dfWrongQH['rc1'], dfWrongQH['zs1'], c = "red", s = 15, linewidths = 0, edgecolors="black", label="Incorrect", marker='^')
    if (len(dfCorrectedQH.index)): 
        ax.scatter(dfCorrectedQH['rc1'], dfCorrectedQH['zs1'], c = "green", s = 15, linewidths = 0, edgecolors="black", label="Corrected")

    
    #plot labels, legend...
    plt.xlim(delta-0.0005, delta+0.0005)
    plt.xlabel('rc1')
    plt.ylabel('zs1')
    ax.legend()
    plt.savefig("Plots/QAQH/" + str(nfp) + "QAQHdivision.png")
    plt.show()
