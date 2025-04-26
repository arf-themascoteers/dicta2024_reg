import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

os.makedirs('saved_plots', exist_ok=True)


v0 = pd.read_csv('v0_grad_norm.csv').to_numpy()
v11 = pd.read_csv('v11_grad_norm.csv').to_numpy()

result = np.concatenate([v0, v11], axis=1)
result = result[::25,:]

result = result[0:10,:]

fig, ax = plt.subplots(figsize=(12, 4))

bar_width = 0.4
x = np.arange(result.shape[0])

ax.bar(x - bar_width / 2, result[:,0], bar_width, label='Sigmoid activation', color='b')
ax.bar(x + bar_width / 2, result[:,1], bar_width, label='Absolute value activation', color='r')

indices = x*50

ax.set_xticks(x)
ax.set_xticklabels(indices)
ax.set_xlabel('Training Iteration', fontsize=14)
ax.set_ylabel('')
fig.subplots_adjust(bottom=0.2)
#fig.subplots_adjust(left=0, right=1, top=1, bottom=0.3)
ax.legend(loc='upper center', bbox_to_anchor=(0.335, 1.18), ncol=2, frameon=True, fontsize=14)

os.makedirs("stored_figs", exist_ok=True)
plt.savefig('stored_figs/grad_bars.png')
plt.show()
