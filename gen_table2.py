import pandas as pd
import numpy as np
import os

def get_summaries_rec(d):
    files = os.listdir(d)
    paths = [os.path.join(d, f) for f in files if f.endswith("_summary.csv")]
    paths = [p for p in paths if not os.path.isdir(p)]

    children = [os.path.join(d, f) for f in files if os.path.isdir(os.path.join(d, f))]
    for child in children:
        cpaths = get_summaries_rec(child)
        paths = paths + cpaths

    return paths

def sanitize_df(df):
    if "algorithm" not in df.columns:
        df['target_size'] = 0
        df['algorithm'] = 'all'
        df['time'] = 0
        df['selected_features'] = ''
    return df

source = get_summaries_rec("11_7")
df = [sanitize_df(pd.read_csv(loc)) for loc in source]
df = [d for d in df if len(d) != 0]
df = pd.concat(df, axis=0, ignore_index=True)

datasets = ["indian_pines","paviaU","salinas"]
algorithm = ["bsnet","v0","v9"]

for d in datasets:
    for alg in algorithm:
        tdf = df[df["dataset"] == d]
        adf = tdf[tdf["algorithm"] == "all"]
        algdf = tdf[tdf["algorithm"] == alg]
        all_oa = adf["oa"].max()
        filtered_df = algdf[(algdf['oa'] >= all_oa)]
        t = 0
        oa = 0
        if len(filtered_df) > 0:
            min_target_size_row = filtered_df.loc[filtered_df['target_size'].idxmin()]
            t = min_target_size_row['target_size']
            oa = min_target_size_row['oa']
        #print(t, d,alg,oa,all_oa)
    #print("-------------------")

for d in datasets:
    for alg in algorithm:
        tdf = df[df["dataset"] == d]
        adf = tdf[tdf["algorithm"] == "all"]
        algdf = tdf[tdf["algorithm"] == alg]
        all_aa = adf["aa"].max()
        filtered_df = algdf[(algdf['aa'] >= all_aa)]
        t = 0
        aa = 0
        if len(filtered_df) > 0:
            min_target_size_row = filtered_df.loc[filtered_df['target_size'].idxmin()]
            t = min_target_size_row['target_size']
            aa = min_target_size_row['aa']
        #print(t, d,alg,aa,all_aa)
    #print("-------------------")

for d in datasets:
    for alg in algorithm:
        tdf = df[df["dataset"] == d]
        algdf = tdf[tdf["algorithm"] == alg]

        max_oa_row = algdf.loc[algdf['oa'].idxmax()]
        max_target_size = max_oa_row['target_size']
        max_oa = max_oa_row['oa']

        max_aa_row = algdf.loc[algdf['aa'].idxmax()]
        max_aa_target_size = max_aa_row['target_size']
        max_aa = max_aa_row['aa']

        print(d, alg, max_target_size, max_oa, max_aa_target_size, max_aa)
    print()
