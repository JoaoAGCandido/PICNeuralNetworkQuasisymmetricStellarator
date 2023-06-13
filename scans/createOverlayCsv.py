import numpy as np
import pandas as pd


# read file
df = pd.read_csv("scans/scan7/scan7Clean.csv.zip")
df = df[df['rc1'] != 0]
df = df[df['zs1'] != 0]
#df = df.sample(frac=0.1)


out = {
    'axLenght': [],
    'RotTrans': [],
    'max_elong': [],
}

threshold = 0.01
for nfp1 in np.arange(1, 8, 1):
    df1 = df[df['nfp'] == nfp1]
    for nfp2 in np.arange(nfp1+1, 9, 1):
        print(nfp1, nfp2)
        df2 = df[df['nfp'] == nfp2]
        
        for i in df1.index:
            for j in df2.index:
                if np.abs(df1.loc[i, "RotTrans"] - df2.loc[j, "RotTrans"]) / df1.loc[i, "RotTrans"] < threshold:
                    if np.abs(df1.loc[i,"axLenght"] - df2.loc[j,"axLenght"] ) / df1.loc[i,"axLenght"] < threshold:
                        if np.abs(df1.loc[i,"max_elong"] - df2.loc[j,"max_elong"] ) / df1.loc[i,"max_elong"] < threshold:
                            out['axLenght'].append(df1.loc[i,"axLenght"])
                            out['max_elong'].append(df1.loc[i,"max_elong"])
                            out['RotTrans'].append(df1.loc[i,"RotTrans"])
                            break
dfOverlay = pd.DataFrame(out)
dfOverlay.to_csv("scans/scan7/overlay.csv", index=False)
print(len(dfOverlay.index))
print(dfOverlay)
