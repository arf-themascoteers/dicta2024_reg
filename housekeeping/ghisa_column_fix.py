import pandas as pd

df = pd.read_csv('../data/ghisaconus.csv')
cols = list(df.columns)
df = df[cols[1:] + [cols[0]]]
df.to_csv('../data/ghisaconus.csv', index=False)
