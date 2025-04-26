import pandas as pd
import numpy as np

df = pd.read_csv("final/ip/ip_summary.csv")
df = df[(df["dataset"]=="indian_pines")&(df["algorithm"]=="v0")&(df["target_size"]==30)]
sfs = df.iloc[0]["selected_features"]
weights = df.iloc[0]["selected_weights"]

sfs = np.array([int(i) for i in sfs.split("|")])
weights = np.array([float(i) for i in weights.split("|")])

print(sfs)
print(weights)
a = np.argsort(weights)[::-1]

print(a)
print(sfs[a])
print(weights[a])