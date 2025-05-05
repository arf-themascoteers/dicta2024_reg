import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22

order = [
    "all",
    "pcal",
    "mcuve",
    "spa2",
    "bsnet",
    "v0",
    "v9_dummy",#V1
    "v4_dummy",#V2
    "v1",#V3
    "v2_dummy",#SABS
    "v9"#BSDR
]
ALGS = {
    "all": "All Bands",
    "pcal": "PCAL",
    "mcuve": "MCUVE",
    "spa2": "SPA",
    "bsnet": "BS_Net-FC",
    "v0": "BS-Net-Regressor",
    "v9_dummy": "V1: BS-Net-Regressor + FCNN",
    "v4_dummy": "V2: V1 + improved aggregation",
    "v1": "V3: V2 + absolute value activation",
    "v2_dummy": "Proposed SABS: V3 + dynamic regulation",
    "v9": "Proposed BSDR",
}

COLORS = {
    "all": "black",
    "pcal": "#008080",
    "mcuve": "orange",
    "spa2": "green",
    "bsnet": "#8B4513",
    "v0": "cyan",
    "v9_dummy": "#008080",
    "v4_dummy": "orange",
    "v1": "green",
    "v2_dummy": "red",
    "v9": "purple",
}


def sanitize_df(df):
    if "algorithm" not in df.columns:
        df['target_size'] = 0
        df['algorithm'] = 'all'
        df['time'] = 0
        df['selected_features'] = ''
    return df


def plot_ablation_oak(source, plot_type, exclude=None, include=None):
    loc = "plots2"
    os.makedirs(loc, exist_ok=True)
    dest = os.path.join(loc, f"{plot_type}.png")

    if exclude is None:
        exclude = []
    if isinstance(source, str):
        df = sanitize_df(pd.read_csv(source))
    else:
        df = [sanitize_df(pd.read_csv(loc)) for loc in source]
        df = [d for d in df if len(d)!=0]
        df = pd.concat(df, axis=0, ignore_index=True)
    df.to_csv(os.path.join(loc,"source.split.csv"), index=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
              "#17becf"]
    markers = ['s', 'P', 'D', '^', 'o', '*', '.','s', 'P', 'D', '^', 'o', '*', '.']
    labels = ["$R^2$", "RMSE", r"RPD"]
    titles = ["(a)", "(b)", "(c)"]
    order = [
        "all",
        "pcal","mcuve", "bsnet",
        "v0",
        "v9_dummy",
        "v4_dummy",
        "v1",
        "v2_dummy",
        "v9"
    ]
    df["sort_order"] = df["algorithm"].apply(lambda x: order.index(x) if x in order else len(order) + ord(x[0]))
    df = df.sort_values("sort_order").drop(columns=["sort_order"])

    algorithms = df["algorithm"].unique()
    #print(algorithms)

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
                                    linewidth=2,
                                    label=algorithm_label,
                                    linestyle=linestyle
                                    )
            #axes[metric_index].legend()

        axes[metric_index].set_xlabel('Target size')
        axes[metric_index].set_ylabel(labels[metric_index])
        #axes[metric_index].set_ylim(min_lim, max_lim)
        axes[metric_index].tick_params(axis='both', which='major')
        axes[metric_index].text(0.5, -0.3, titles[metric_index],
                                         transform=axes[metric_index].transAxes,
                                         ha='center')
        axes[metric_index].set_xscale("log", base=2)
        axes[metric_index].set_xticks([8, 16, 32, 64, 128, 256, 512])
        axes[metric_index].get_xaxis().set_major_formatter(plt.ScalarFormatter())


        if metric_index == 0:
            legend = axes[metric_index].legend(loc='upper left',
                                               ncols=2,
                                               bbox_to_anchor=(0, 1.6),
                                               columnspacing=1.0, frameon=True
                                               )
        #legend.get_title().set_fontsize('18')
        legend.get_title().set_fontweight('bold')


        axes[metric_index].grid(True, linestyle='-', alpha=0.6)

    fig.subplots_adjust(wspace=0.4, top=0.7, bottom=0.2)
    plt.savefig(dest, bbox_inches='tight', pad_inches=0.05)
    plt.show()



def plot_ablation(source,plot_type,include = None):
    if include is None:
        include = []
    plot_ablation_oak(source,plot_type,include=include)


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

def create_plot(plot_type="ablation"):
    plot_ablation(
        get_summaries_rec("lucas_results3"),
        plot_type,
        include=[
            "all",
            "v0",
            "v9_dummy",
            "v4_dummy",
            "v1",
            "v2_dummy",
            "v9"
        ]
    )


if __name__ == "__main__":
    create_plot("ablation")