import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

ALGS = {
    "v0": "BS-Net-Classifier",
    "v9": "Proposed SABS",
    "all": "All Bands",
    "mcuve": "MCUVE",
    "spa": "SPA",
    "bsnet": "BS-Net-FC",
    "pcal": "PCAL",
}

DSS = {
    "lucas": "LUCAS",
}

COLORS = {
    "v0": "#1f77b4",
    "all": "#2ca02c",
    "mcuve": "#ff7f0e",
    "spa": "#00CED1",
    "bsnet": "#008000",
    "pcal": "#9467bd",
    "v9": "#d62728",
}

def sanitize_df(df):
    if "algorithm" not in df.columns:
        df['target_size'] = 0
        df['algorithm'] = 'all'
        df['time'] = 0
        df['selected_features'] = ''
    return df

def get_summaries_rec(d):
    files = os.listdir(d)
    paths = [os.path.join(d, f) for f in files if f.endswith("_summary.csv")]
    paths = [p for p in paths if not os.path.isdir(p)]

    children = [os.path.join(d, f) for f in files if os.path.isdir(os.path.join(d, f))]
    for child in children:
        cpaths = get_summaries_rec(child)
        paths = paths + cpaths

    return paths

def plot_separately(source, exclude=None, include=None, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    for dataset_key, dataset_label in DSS.items():
        if exclude is None:
            exclude = []
        if isinstance(source, str):
            df = sanitize_df(pd.read_csv(source))
        else:
            df = [sanitize_df(pd.read_csv(loc)) for loc in source]
            df = [d for d in df if len(d) != 0]
            df = pd.concat(df, axis=0, ignore_index=True)

        df = df[df["dataset"] == dataset_key]
        if len(df) == 0:
            continue

        colors = list(COLORS.values())
        markers = ['s', 'P', 'D', '^', 'o', '*', '.', 's', 'P', 'D', '^', 'o', '*', '.']
        labels = ["$R^2$", "RMSE", "RPD"]
        order = ["all", "pcal", "mcuve", "spa","bsnet", "v0", "v9"]

        df["sort_order"] = df["algorithm"].apply(lambda x: order.index(x) if x in order else len(order) + ord(x[0]))
        df = df.sort_values("sort_order").drop(columns=["sort_order"])

        algorithms = df["algorithm"].unique()
        if include is None:
            include = algorithms
        include = [x for x in include if x not in exclude]
        if len(include) == 0:
            include = df["algorithm"].unique()
        else:
            df = df[df["algorithm"].isin(include)]

        min_lim = min(df["r2"].min(), df["rmse"].min(), df["rpd"].min()) - 0.02
        max_lim = max(df["r2"].max(), df["rmse"].max(), df["rpd"].max()) + 0.02

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for metric_index, metric in enumerate(["r2", "rmse", "rpd"]):
            algorithm_counter = 0
            for algorithm_index, algorithm in enumerate(include):
                algorithm_label = ALGS.get(algorithm, algorithm)
                alg_df = df[df["algorithm"] == algorithm].sort_values(by='target_size')

                linestyle = "-"
                marker = markers[algorithm_counter]
                color = COLORS.get(algorithm, colors[algorithm_counter])

                if algorithm == "all":
                    r2 = alg_df.iloc[0]["r2"]
                    rmse = alg_df.iloc[0]["rmse"]
                    k = alg_df.iloc[0]["rpd"]
                    alg_df = pd.DataFrame(
                        {'target_size': range(5, 31), 'r2': [r2] * 26, 'rmse': [rmse] * 26, 'rpd': [k] * 26})
                    linestyle = "--"
                    color = "#000000"
                    marker = None
                else:
                    algorithm_counter += 1

                axes[metric_index].plot(alg_df['target_size'], alg_df[metric],
                                        label=algorithm_label,
                                        color=color,
                                        fillstyle='none', markersize=7, linewidth=2, linestyle=linestyle)

            axes[metric_index].set_xscale("log", base=2)
            axes[metric_index].set_xticks([8, 16, 32, 64, 128, 256, 512])

            axes[metric_index].set_xlabel('Target size')
            axes[metric_index].set_ylabel(labels[metric_index])
            axes[metric_index].set_ylim(min_lim, max_lim)
            axes[metric_index].tick_params(axis='both', which='major')
            axes[metric_index].grid(True, linestyle='-', alpha=0.6)

        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.15), frameon=True)

        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, f"r3_{dataset_key}.png"), bbox_inches='tight')#, pad_inches=0.05)
        plt.close(fig)

if __name__ == "__main__":
    plot_separately(
        get_summaries_rec("lucas_results"),
        include=["pcal", "mcuve", "spa", "bsnet", "v0", "v9", "all"]
    )
