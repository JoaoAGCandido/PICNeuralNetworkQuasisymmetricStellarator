import pandas as pd
from datetime import datetime

"""Plot standart deviation from the seed, in function
of the number of points"""

out = {
    'numPoints': [],
    'std': [],
}

#df = pd.read_csv("../nfp3Default/seedSearch.csv")
df = pd.read_csv("../nfp3/seedSearch.csv")

size = len(df.index)
for i in range(1, 100):
    out["numPoints"].append(i+1)
    out["std"].append(df.loc[0:i, 'testR2'].std())


df3 = pd.DataFrame(out)
print(df3.sort_values("numPoints", ascending=False))
df3.plot("numPoints", "std")
