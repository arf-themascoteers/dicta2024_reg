import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

ALGS = {
    "v0": "BS-Net-Classifier [12]",
    "all": "All Bands",
    "v1": "V1: BS-Net-Classifier [12] + FCNN",
    "v2": "V2: V1 + improved aggregation",
    "v6": "V3: V2 + absolute value activation (no sparse constraint)",
    "v9": "V4: V3 + dynamic sparse constraint (proposed algorithm)",
}

COLORS = {
    "v0": "#1f77b4",
    "v4": "#d62728",
    "all": "#2ca02c",
    "mcuve": "#ff7f0e",
    "bsnet": "#008000",
    "pcal": "#9467bd",
    "v1": "#7FFF00",
    "v2": "#FF00FF",
    "v6": "#9467bd",
    "v9": "#d62728",

}


def sanitize_df(df):
    if "algorithm" not in df.columns:
        df['target_size'] = 0
        df['algorithm'] = 'all'
        df['time'] = 0
        df['selected_features'] = ''
    return df


def plot_ablation_oak(source, exclude=None, include=None, out_file="ab.png"):
    if exclude is None:
        exclude = []
    os.makedirs("saved_figs", exist_ok=True)
    if isinstance(source, str):
        df = sanitize_df(pd.read_csv(source))
    else:
        df = [sanitize_df(pd.read_csv(loc)) for loc in source]
        df = [d for d in df if len(d)!=0]
        df = pd.concat(df, axis=0, ignore_index=True)


    df.to_csv(os.path.join("saved_figs", "source.split.csv"), index=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
              "#17becf"]
    markers = ['s', 'P', 'D', '^', 'o', '*', '.','s', 'P', 'D', '^', 'o', '*', '.']
    labels = ["$R^2$", "RMSE", r"RPD"]
    titles = ["(a)", "(b)", "(c)"]

    min_lim = 0.3
    max_lim = 1

    order = ["all", "pcal", "mcuve", "bsnet", "v0", "v1", "v2", "v3", "v35", "v4"]
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
    print(min_lim, max_lim)
    dest = os.path.join("saved_figs", f"lucas.png")
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
    for metric_index, metric in enumerate(["r2", "rmse", "rpd"]):
        algorithm_counter = 0
        for algorithm_index, algorithm in enumerate(include):
            algorithm_label = algorithm
            if algorithm in ALGS:
                algorithm_label = ALGS[algorithm]
            alg_df = df[df["algorithm"] == algorithm]
            alg_df = alg_df.sort_values(by='target_size')
            linestyle = "-"
            if algorithm in COLORS:
                color = COLORS[algorithm]
            else:
                color = colors[algorithm_counter]

            marker = markers[algorithm_counter]
            if algorithm == "all":
                oa = alg_df.iloc[0]["r2"]
                aa = alg_df.iloc[0]["rmse"]
                k = alg_df.iloc[0]["rpd"]
                alg_df = pd.DataFrame(
                    {'target_size': [int(2**i) for i in range(3, 10)], 'r2': [oa] * 7, 'rmse': [aa] * 7, 'rpd': [k] * 7})
                linestyle = "--"
                color = "#000000"
                marker = None
            else:
                algorithm_counter = algorithm_counter + 1

            axes[metric_index].plot(alg_df['target_size'], alg_df[metric],
                                    color=color,
                                    fillstyle='none', markersize=7,
                                    linewidth=2, linestyle=linestyle,
                                    label=algorithm_label
                                    )
            #axes[metric_index].legend()

        axes[metric_index].set_xlabel('Target size', fontsize=18)
        axes[metric_index].set_ylabel(labels[metric_index], fontsize=18)
        #axes[metric_index].set_ylim(min_lim, max_lim)
        axes[metric_index].tick_params(axis='both', which='major', labelsize=14)
        axes[metric_index].text(0.5, -0.3, titles[metric_index],
                                         transform=axes[metric_index].transAxes,
                                         fontsize=18, ha='center')
        axes[metric_index].set_xscale("log", base=2)
        axes[metric_index].set_xticks([8, 16, 32, 64, 128, 256, 512])
        axes[metric_index].get_xaxis().set_major_formatter(plt.ScalarFormatter())


        if metric_index == 0:
            legend = axes[metric_index].legend(loc='upper left', fontsize=18, ncols=2,
                                               bbox_to_anchor=(0, 1.5),
                                               columnspacing=1.0, frameon=True
                                               )
        legend.get_title().set_fontsize('18')
        legend.get_title().set_fontweight('bold')


        axes[metric_index].grid(True, linestyle='-', alpha=0.6)

    fig.subplots_adjust(wspace=0.4, top=0.7, bottom=0.2)
    plt.savefig(dest, bbox_inches='tight', pad_inches=0.05)
    plt.show()



def plot_ablation(source, include = None):
    if include is None:
        include = []
    plot_ablation_oak(source,
         out_file = "ablation.png",
         include=include
    )


def get_summaries(d):
    files = os.listdir(d)
    paths = [os.path.join(d, f) for f in files if f.endswith("_summary.csv")]
    return paths

def get_summaries_rec(d):
    files = os.listdir(d)
    paths = [os.path.join(d, f) for f in files if f.endswith("_summary.csv")]
    paths = [p for p in paths if not os.path.isdir(p)]

    children = [os.path.join(d, f) for f in files if os.path.isdir(os.path.join(d, f))]
    for child in children:
        cpaths = get_summaries_rec(child)
        paths = paths + cpaths

    return paths


if __name__ == "__main__":
    plot_ablation(
        get_summaries_rec("summary")
        ,
        include=["v0","v1","v2","v6","all"]
    )
