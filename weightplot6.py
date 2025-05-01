import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random

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

sorted_weights = dict(new_array[:32])

bands = list(sorted_weights.keys())
bands = sorted(bands)
print(bands)

plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_weights)), sorted_weights.values())
plt.xticks(range(len(sorted_weights)), [key + 1 for key in sorted_weights.keys()], rotation=90)
plt.title("LUCAS")
plt.xlabel('Wavelength (nm)')
plt.ylabel('Weight')
ymax = plt.gca().get_ylim()[1]
plt.yticks(np.arange(0, ymax + 0.5, 0.5))
#plt.subplots_adjust(hspace=1.2)
plt.tight_layout()
plt.savefig('reg_bar_sb.png')
plt.show()
