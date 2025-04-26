import os
import pandas as pd

for f in os.listdir("data"):
    if not f.endswith(".csv"):
        continue
    p = os.path.join("data", f)
    df = pd.read_csv(p)
    unique_values_count = df.iloc[:, -1].value_counts()
    print(f)
    print(len(unique_values_count))
    print(len(df))
    print(len(df.columns)-1)
    print(unique_values_count)
