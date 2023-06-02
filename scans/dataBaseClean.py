import pandas as pd

print("\n")

"""
Conditions from paper Mapping the space of quasi
|rot transfor| > 
"""

# read file
df = pd.read_csv("scans/scan7/scan7.csv.zip")

print(df)

# df = df[ (df['axLenght'] > 6.28 * 2) & (df['axLenght'] <  6.29 * 2) ]
df = df[df['RotTrans'] > 0.2]
df = df[df['max_elong'] < 10]
df = df[df['LgradB'] > 0.2]
df = df[df['min_R0'] > 0.4]

print(df)

df.to_csv("scans/scan7/scan7Clean.csv.zip", index=False)
