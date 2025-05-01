import pandas as pd
import os

file = "../refined.csv"
df = pd.read_csv(file)
dbs = ["pcal", "mcuve", "spa", "bsnet", "v0", "v9", "all"]
df = df[df["algorithm"].isin(dbs)]

ALGS = {
    "v0": "BS-Net-Regressor",
    "v9": "Proposed SABS",
    "all": "All Bands",
    "mcuve": "MCUVE",
    "spa": "SPA",
    "bsnet": "BS-Net-FC",
    "pcal": "PCAL",
}

order = ["all", "pcal", "mcuve","spa", "bsnet", "v0", "v9"]
df["sort_order"] = df["algorithm"].apply(lambda x: order.index(x) if x in order else len(order) + ord(x[0]))
df = df.sort_values("sort_order").drop(columns=["sort_order"])

results = []
for (dataset, algorithm), group in df[df["algorithm"] != "all"].groupby(["dataset", "algorithm"]):
    apf = df[(df["algorithm"] == "all") & (df["dataset"] == dataset)]
    threshold = apf.iloc[0]["r2"]

    surpass = group[group["r2"] >= threshold]
    min_target_size_surpassing = surpass["target_size"].min() if not surpass.empty else "-"

    max_row = group.loc[group["r2"].idxmax()]
    max_r2 = max_row["r2"]
    max_r2_ts = max_row["target_size"]

    diff_percent = ((max_r2 - threshold) / threshold) * 100
    diff_percent = f"{'+' if diff_percent >= 0 else ''}{diff_percent:.2f}%"

    results.append((dataset, ALGS[algorithm], min_target_size_surpassing,
                    round(max_r2,2), max_r2_ts, diff_percent))

result_df = pd.DataFrame(results, columns=[
    "dataset",
    "algorithm",
    "count_surpassing_apf",
    "max_r2",
    "target_size_at_max_r2",
    "r2_diff_percent"
])

result_df.to_csv("apf_comparison.csv", index=False)
