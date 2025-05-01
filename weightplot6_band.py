import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import seaborn as sns

def band_to_wl(band):
    wl = 400 + (band * 0.5)
    return wl

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22


path = 'lucas_results/v9_lucas_v9_lucas_512.csv'
#path = 'saved_results/v10/v10_v10_lucas_30.csv'

df = pd.read_csv(path)
last_row = df.iloc[-1]
weights = {int(col.split('_')[1]): val*2 for col, val in last_row.items() if col.startswith('weight_')}
arr = sorted(weights.items(), key=lambda item: item[1], reverse=True)

counter = 0
new_array = []
for i in range(len(arr)):
    band_number = arr[i][0]
    if 3000 <= band_number <= 3999:
        if counter > 8:
            continue
        else:
            counter = counter + 1
            new_array.append( (band_to_wl(arr[i][0]), arr[i][1]) )
    else:
        new_array.append( (band_to_wl(arr[i][0]), arr[i][1]) )

sorted_weights = dict(new_array[:512])

bands = list(sorted_weights.keys())
bands = sorted(bands)

top_bands = bands

print(top_bands)

bins = np.arange(400, 2501, 300)
labels = [f"{start}-{start+300}" for start in bins[:-1]]
counts = np.histogram(top_bands, bins=bins)[0]

plt.figure(figsize=(10, 1.5))
sns.heatmap([counts], annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=["Band count"])
plt.xlabel("Wavelength Range (nm)")
plt.title("Distribution of Top 512 Bands Across 300 nm Intervals")
plt.tight_layout()
plt.show()

counter = 0
for b in top_bands:
    if 400 <= b <= 700:
        counter = counter + 1
print(counter)