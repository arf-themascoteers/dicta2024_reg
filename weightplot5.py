import pandas as pd
import matplotlib.pyplot as plt
import os

paths = [
    '11_7/i/iv9/iv9_v9_indian_pines_30.csv',
    '11_7/p/p9/pv9_v9_paviaU_30.csv',
    '11_7/s/sv9_v9_salinas_30.csv'
]

fig, axes = plt.subplots(1, 3, figsize=(18, 3))
dss = ["Indian Pines", "Pavia University", "Salinas"]
for idx, path in enumerate(paths):
    df = pd.read_csv(path)
    last_row = df.iloc[-1]
    weights = {int(col.split('_')[1]): val for col, val in last_row.items() if col.startswith('weight_')}
    sorted_weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True)[:30])
    axes[idx].bar(range(len(sorted_weights)), sorted_weights.values())
    axes[idx].set_xticks(range(len(sorted_weights)))
    axes[idx].set_xticklabels([key + 1 for key in sorted_weights.keys()])
    axes[idx].set_title(dss[idx], fontsize=18)
    axes[idx].set_xlabel('Band Number', fontsize=18)
    axes[idx].set_ylabel('Weight', fontsize=18)
    axes[idx].tick_params(axis='x', rotation=70)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.3)
plt.tight_layout(pad=2)
plt.savefig('bar3.png')
#plt.show()
