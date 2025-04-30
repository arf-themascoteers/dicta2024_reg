import pandas as pd
import os

def replace(d):
    files = os.listdir(d)
    paths = [os.path.join(d, f) for f in files if f.endswith("_details.csv") and "all_features" not in f]

    for p in paths:
        df = pd.read_csv(p)
        s = p.replace("_details.csv", "_summary.csv")
        df2 = pd.read_csv(s)

        grouped = df.groupby(["algorithm", "target_size"])["rpd"].mean()

        for (a, t), rpd_val in grouped.items():
            df2.loc[(df2["algorithm"] == a) & (df2["target_size"] == t), "rpd"] = rpd_val
        df2.to_csv(s, index=False)

replace("../lucas_results/")


