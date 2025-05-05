import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

ALGS = {
    "v0": "BS-Net-Regressor",
    "v9": "Proposed SABS",
    "all": "All Bands",
    "mcuve": "MCUVE",
    "spa": "SPA",
    "spa2": "SPA",
    "bsnet": "BS-Net-FC",
    "pcal": "PCAL",
    "v1": "V1: BS-Net-Regressor + FCNN",
    "v2": "V2: V1 + improved aggregation",
    "v6": "V3: V2 + absolute value activation"
}

DSS = {
    "lucas": "LUCAS",
}

COLORS = {
    "v0": "#1f77b4",
    "all": "#2ca02c",
    "mcuve": "#ff7f0e",
    "spa": "#00CED1",
    "spa2": "#00CED1",
    "bsnet": "#008000",
    "pcal": "#9467bd",
    "v9": "#d62728",
    "v1": "#7FFF00",
    "v2": "#FF00FF",
    "v6": "#9467bd",
}

def sanitize_df(df):
    df['target_size'] = 0
    df['algorithm'] = 'all'
    df['time'] = 0
    df['selected_features'] = ''
    return df

def plot_separately(source,all,exclude=None, include=None,ablation=False):
    if ablation:
        ALGS["v9"] = "Proposed SABS: V3 + dynamic regulation"

    df = pd.read_csv(source)
    all_df = pd.read_csv(all)
    all_df = sanitize_df(all_df)
    df = pd.concat([df,all_df], axis=0, ignore_index=True)

    if exclude is None:
        exclude = []

    colors = list(COLORS.values())
    markers = ['s', 'P', 'D', '^', 'o', '*', '.', 's', 'P', 'D', '^', 'o', '*', '.']
    labels = ["$R^2$", "RMSE", "RPD"]
    order = ["all", "pcal", "mcuve", "spa","bsnet", "v0", "v1","v2","v6","v9"]

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

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for metric_index, metric in enumerate(["r2", "rmse", "rpd"]):
        algorithm_counter = 0
        for algorithm_index, algorithm in enumerate(include):
            algorithm_label = ALGS.get(algorithm, algorithm)
            #print(algorithm_label)
            alg_df = df[df["algorithm"] == algorithm].sort_values(by='target_size')

            linestyle = "-"
            marker = markers[algorithm_counter]
            color = COLORS.get(algorithm, colors[algorithm_counter])

            if algorithm == "all":
                r2 = alg_df.iloc[0]["r2"]
                rmse = alg_df.iloc[0]["rmse"]
                k = alg_df.iloc[0]["rpd"]
                alg_df = pd.DataFrame(
                    {'target_size': [512, 256, 128, 64, 32, 16, 8], 'r2': [r2] * 7, 'rmse': [rmse] * 7, 'rpd': [k] * 7})
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
        axes[metric_index].get_xaxis().set_major_formatter(plt.ScalarFormatter())

        axes[metric_index].set_xlabel('Target size')
        axes[metric_index].set_ylabel(labels[metric_index])
        #axes[metric_index].set_ylim(min_lim, max_lim)
        axes[metric_index].tick_params(axis='both', which='major')
        axes[metric_index].grid(True, linestyle='-', alpha=0.6)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', ncol=3, bbox_to_anchor=(0.45, 1), frameon=True)

    #fig.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.savefig("r3_combined_refined.png", bbox_inches='tight')#, pad_inches=0.05)
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    plot_separately("lucas_results2/loc_combined.csv","lucas_results2/all_w_m_all_features_summary.csv",
                    exclude=["bsnet"]
                    )
