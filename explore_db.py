import pandas as pd
from sklearn.model_selection import train_test_split
import os


for f in os.listdir("data"):
    if not f.endswith(".csv"):
        continue
    p = os.path.join("data", f)
    df = pd.read_csv(p)
    print(p)
    unique_classes = df.groupby('class').size()
    print("class",len(unique_classes)-1)
    print("total",len(df))
    print("labelled",len(df[df['class'] == 0]))
    print("unlabelled",len(df[df['class'] != 0]))
    print("bands",len(df.columns)-1)
