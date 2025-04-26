import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('saved_plots', exist_ok=True)

data = pd.read_csv('8_7_weights/v0_weight_all/v0_weight_all_v0_weight_all_indian_pines_5_weights_all.csv')
data = data.iloc[0:4, :]
batches = data.iloc[:, 0]
weights = data.iloc[:, 1:]

fig, ax = plt.subplots(figsize=(10, 3))

bar_width = 0.2
index = range(len(batches))

for i, weight in enumerate(weights.columns):
    ax.bar([x + i * bar_width for x in index], weights[weight], bar_width, label=f'Mean weight for band {i+1}')

ax.set_xlabel('Batch',fontsize=16)
ax.set_ylabel('Mean weight',fontsize=16)
ax.set_xticks([x + bar_width for x in index])
ax.set_xticklabels(batches)
ax.legend()

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('saved_plots/bars.png', bbox_inches='tight')
plt.close()
