import pandas as pd
from datetime import datetime

out = {
    'hiddenLayerSize': [],
    'lossMean': [],
    'lossStd': [],
    'bestValidationScoreMean': [],
    'bestValidationScoreStd': [],
    'trainTimeMean': [],
    'trainTimeStd': []
}

df = pd.read_csv("NeuralNetwork/nfp3/hiddenLayerLoss.csv")
df.sort_values("hiddenLayerSize", inplace=True)
df['trainTime'] = pd.to_timedelta(df['trainTime'])

size = len(df.index)
num = 10  # number of rows for hiddenLayerSize
# size/num must be int (all hiddenLayerSize must have the same number of rows)
for i in range(int(size/num)):
    print(i)
    df2 = df.iloc[num*i:num*i+num]
    print(df2, "\n\n")
    out["hiddenLayerSize"].append(df2.iloc[0][0])
    out["lossMean"].append(df2.loc[:, 'loss'].mean())
    out["lossStd"].append(df2.loc[:, 'loss'].std())
    out["bestValidationScoreMean"].append(
        df2.loc[:, 'bestValidationScore'].mean())
    out["bestValidationScoreStd"].append(
        df2.loc[:, 'bestValidationScore'].std())
    out["trainTimeMean"].append(df2.loc[:, 'trainTime'].mean())
    out["trainTimeStd"].append(df2.loc[:, 'trainTime'].std())


df3 = pd.DataFrame(out)
df3.sort_values("bestValidationScoreMean", inplace=True, ascending=False)
print(df3)
df3.to_csv("NeuralNetwork/nfp3/std.csv", index=False)
