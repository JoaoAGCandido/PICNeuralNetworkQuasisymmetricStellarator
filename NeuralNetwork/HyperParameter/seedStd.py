import pandas as pd
from datetime import datetime

"""Plot standart deviation from the seed, in function
of the number of points"""

out = {
    'numPoints': [],
    'std': [],
}

df = pd.read_csv("../nfp3/seedSearchDefault.csv")
# df = pd.read_csv("../nfpAll/seedSearchDefault.csv")

size = len(df.index)
for i in range(1, 50):
    out["numPoints"].append(i+1)
    out["std"].append(df.loc[0:i, 'bestValidationScore'].std())


df3 = pd.DataFrame(out)
# print(df3)
df3.plot("numPoints", "std")
