import pandas as pd


DSS = {
    "indian_pines" : "Indian Pines",
    "paviaU" : "paviaU",
    "salinas" : "salinas",
}

def get_metrics(df2, target_size=None):
    if target_size is not None:
        df3 = df2[df2["target_size"] == target_size]
    else:
        df3 = df2
    if len(df3) == 0:
        return None, None, None

    oa_mean = df3["oa"].mean()
    aa_mean = df3["aa"].mean()
    k_mean = df3["k"].mean()

    oa_std = df3["oa"].std()
    aa_std = df3["aa"].std()
    k_std = df3["k"].std()

    oa_str = f"{oa_mean:.2f}"
    aa_str = f"{aa_mean:.2f}"
    k_str = f"{k_mean:.2f}"

    if target_size is not None and len(df3)>1:
        oa_str = f"{oa_str}±{oa_std:.2f}"
        aa_str = f"{aa_str}±{aa_std:.2f}"
        k_str = f"{k_str}±{k_std:.2f}"

    return oa_str, aa_str, k_str


def get_metrics_for_6_targets(df, dataset, algorithm):
    df = df[(df["algorithm"] == algorithm) & (df["dataset"] == dataset)]
    if len(df) == 0:
        print(f"Algorithm {algorithm} not found for {dataset}")
        return None, None, None
    oa_strs = []
    aa_strs = []
    k_strs = []

    if algorithm == "all":
        oa_str, aa_str, k_str = get_metrics(df)
        oa_strs = oa_strs + ([oa_str] * 6)
        aa_strs = aa_strs + ([aa_str] * 6)
        k_strs = k_strs + ([k_str] * 6)
        return oa_strs, aa_strs, k_strs

    for target_size in [5, 10, 15, 20, 25, 30]:
        oa_str, aa_str, k_str = get_metrics(df, target_size)
        oa_strs.append(oa_str)
        aa_strs.append(aa_str)
        k_strs.append(k_str)

    return oa_strs, aa_strs, k_strs


def create_table():
    df = pd.read_csv("saved_figs/source.split.csv")
    algorithms = df["algorithm"].unique()

    oa_df = pd.DataFrame(columns=["algorithm", "oa_5", "oa_10", "oa_15", "oa_15", "oa_20", "oa_25", "oa_30"])
    aa_df = pd.DataFrame(columns=["algorithm", "aa_5", "aa_10", "aa_15", "aa_15", "aa_20", "aa_25", "aa_30"])
    k_df = pd.DataFrame(columns=["algorithm", "k_5", "k_10", "k_15", "k_15", "k_20", "k_25", "k_30"])

    for algorithm in algorithms:
        oa_strs, aa_strs, k_strs = get_metrics_for_6_targets(df, dataset, algorithm)
        if oa_strs is None:
            continue
        print(f"Algorithm: {algorithm}; {oa_strs}")
        oa_df.loc[len(oa_df)] = {"algorithm": algorithm, "oa_5": oa_strs[0], "oa_10": oa_strs[1], "oa_15": oa_strs[2], "oa_20": oa_strs[3], "oa_25": oa_strs[4], "oa_30": oa_strs[5]}
        aa_df.loc[len(aa_df)] = {"algorithm": algorithm, "aa_5": aa_strs[0], "aa_10": aa_strs[1], "aa_15": aa_strs[2], "aa_20": aa_strs[3], "aa_25": aa_strs[4], "aa_30": aa_strs[5]}
        k_df.loc[len(k_df)] = {"algorithm": algorithm, "k_5": k_strs[0], "k_10": k_strs[1], "k_15": k_strs[2], "k_20": k_strs[3], "k_25": k_strs[4], "k_30": k_strs[5]}

    oa_df.to_csv(f"../final_results/{dataset}_oa.csv", index=False)
    aa_df.to_csv(f"../final_results/{dataset}_oa.csv", index=False)
    k_df.to_csv(f"../final_results/{dataset}_kappa.csv", index=False)


create_table()
