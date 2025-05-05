import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def sanitize_df(df):
    if "algorithm" not in df.columns:
        df['target_size'] = 0
        df['algorithm'] = 'all'
        df['time'] = 0
        df['selected_features'] = ''
    return df

def get_summaries_rec(d):
    files = os.listdir(d)
    paths = [os.path.join(d, f) for f in files if f.endswith("_summary.csv")
             and "all_features_summary" not in f]
    paths = [p for p in paths if not os.path.isdir(p)]

    children = [os.path.join(d, f) for f in files if os.path.isdir(os.path.join(d, f))]
    for child in children:
        cpaths = get_summaries_rec(child)
        paths = paths + cpaths

    return paths

def combine_summary(loc):
    source = get_summaries_rec(loc)
    df = [sanitize_df(pd.read_csv(loc)) for loc in source]
    df = [d for d in df if len(d) != 0]
    df = pd.concat(df, axis=0, ignore_index=True)
    df.to_csv(f"{loc}/loc_combined.csv", index=False)


if __name__ == "__main__":
    combine_summary("lucas_results2")
