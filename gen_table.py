import pandas as pd
import numpy as np

header = r"""
\begin{table}[ht]
\centering
\caption{Comparison of OA, AA, and $\kappa$ for different datasets and target sizes}
\begin{tabular}{|c|c|c|c|c|c|c|c|}
\hline
\multirow{2}{*}{\parbox{1.5cm}{\centering Dataset}} & \multirow{2}{*}{\parbox{1.5cm}{\centering Target Size}} & \multicolumn{2}{c|}{BS-Net-Classifier \cite{zhang2024hyperspectral}} & \multicolumn{2}{c|}{Proposed Algorithm} \\
\cline{3-6}
 &  & OA & AA & OA & AA  \\
\hline
"""

locations = [
    ["11_7/i/iv0/iv0_details.csv", "11_7/p/p_details.csv", "11_7/s/s_details.csv"],
    ["11_7/i/iv9/iv9_details.csv", "11_7/p/p9/pv9_details.csv", "11_7/s/sv9_details.csv"]
]



names = ["v0","v9"]
targets = [5,10,15,20,25,30]
metrics = ["oa","aa"]

means = {}
stds = {}
datasets = ["indian_pines","paviaU","salinas"]
labels = ["IP","PU","S"]
for a_index, a in enumerate(names):
    if a not in means:
        means[a] = {}
        stds[a] = {}
    for t in targets:
        if t not in means[a]:
            means[a][t] = {}
            stds[a][t] = {}
        for di, d in enumerate(datasets):
            if d not in means[a][t]:
                means[a][t][d] = {}
                stds[a][t][d] = {}
            location = locations[a_index][di]
            df = pd.read_csv(location)
            df = df[ (df["algorithm"] == a)&(df['target_size'] == t)]
            for m in metrics:
                mean_score = df[m].mean()
                std_score = df[m].std()
                means[a][t][d][m] = mean_score
                stds[a][t][d][m] = std_score



body = ""
for di, d in enumerate(datasets):
    body = body + labels[di]
    for t_index, t in enumerate(targets):
        body = body + f"&{t}"
        for a in names:
            for m in metrics:
                mean_score = means[a][t][d][m]
                std_score = stds[a][t][d][m]
                score = f"{mean_score:.2f}±{std_score:.2f}"
                body = body + f"&{score}"
        body = body + r"\\"+"\n"
        body = body + r"\hline"+"\n"



footer = r"""
\end{tabular}
\end{table}
"""

print(header+"\n"+body+"\n"+footer)