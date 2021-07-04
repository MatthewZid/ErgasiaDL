import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_csv('../csv/irmas_train.csv', header=None)
X = dataset[0]
y = dataset[1]

print(y.value_counts())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42, shuffle=True, stratify=y)

train = pd.concat([X_train, y_train], axis=1, ignore_index=True)
val = pd.concat([X_val, y_val], axis=1, ignore_index=True)

train.to_csv('../csv/train_dataset.csv', header=False, index=False)
val.to_csv('../csv/val_dataset.csv', header=False, index=False) 
