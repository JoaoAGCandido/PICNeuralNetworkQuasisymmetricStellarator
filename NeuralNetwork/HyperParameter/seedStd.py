import pandas as pd
from datetime import datetime

"""Plot standart deviation from the seed, in function
of the number of points"""

out = {
    'numPoints': [],
    "testR2Std": [],
    "testR2RealStd": [],
    "testR2RealNoOutilerStd":[],
}

#df = pd.read_csv("../nfp3Default/seedSearchDefault2.csv")
#df = pd.read_csv("../nfp3/seedSearchOptimized2.csv")
df = pd.read_csv("../../OtherModels/seedSearchPoly2.csv")
#df = pd.read_csv("../../OtherModels/seedSearchLinear2.csv")

size = len(df.index)
for i in range(1, 100):
    out["numPoints"].append(i+1)
    out["testR2Std"].append(df.loc[0:i, 'testR2'].std())
    out["testR2RealStd"].append(df.loc[0:i, 'testR2Real'].std())
    out["testR2RealNoOutilerStd"].append(df.loc[0:i, 'testR2RealNoOutiler'].std())

df3 = pd.DataFrame(out)
print(df3.sort_values("numPoints", ascending=False))
df3.plot("numPoints", "testR2Std")
df3.plot("numPoints", "testR2RealStd")
df3.plot("numPoints", "testR2RealNoOutilerStd")
