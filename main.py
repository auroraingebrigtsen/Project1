import numpy as np
import pandas as pd
from sklearn import datasets, model_selection, metrics


#learn(X, y, impurity_measure='entropy')
#predict(x, tree)


df = pd.read_csv('project1\wine_dataset.csv')
print(df.head(10))
X = df.data[:, :6] # Store the first two features (sepal length in cm and sepal width in cm)
print(X)
y = df[:, 7]   # Store the labels (0, 1 or 2) coding the Iris species of each sample
print(y)