import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = '8_7_weights/v0_weight/v0_weight_v0_weight_indian_pines_5_weights.csv'

df = pd.read_csv(path)
data = df.values[:, 1:]
means = np.mean(data, axis=1)
stds = np.std(data, axis=1)

indices = list(range(0, len(data), 50))
means_50 = means[indices]
stds_50 = stds[indices]

fig, ax = plt.subplots(figsize=(12, 4))

bar_width = 0.4
x = np.arange(len(indices))

ax.bar(x - bar_width / 2, means_50, bar_width, label='Mean weight', color='b')
ax.bar(x + bar_width / 2, stds_50, bar_width, label='Standard deviation of the weights', color='r')

ax.set_xticks(x)
ax.set_xticklabels(indices)
ax.set_xlabel('Training Iteration', fontsize=14)
ax.set_ylabel('')
fig.subplots_adjust(bottom=0.2)
#fig.subplots_adjust(left=0, right=1, top=1, bottom=0.3)
ax.legend(loc='upper center', bbox_to_anchor=(0.335, 1.18), ncol=2, frameon=True, fontsize=14)

os.makedirs("stored_figs", exist_ok=True)
plt.savefig('stored_figs/weightplot_m_s.png')
plt.show()
