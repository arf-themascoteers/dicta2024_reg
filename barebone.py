import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/indian_pines.csv")
df = df[df.iloc[:, -1] != 0]
df.iloc[:, :-1] = minmax_scale(df.iloc[:, :-1])
df.iloc[:, -1] = df.iloc[:, -1] + 10
df = df.to_numpy()
train, test = train_test_split(df, test_size=0.95, stratify=df[:, -1])
svc = SVC(C=1e5, kernel='rbf', gamma=1.)
svc.fit(train[:,0:-1], train[:,-1])
y_pred = svc.predict(test[:,0:-1])
acc = accuracy_score(test[:,-1], y_pred)
print(acc)

