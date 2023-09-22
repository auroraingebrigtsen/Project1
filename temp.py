import pandas as pd

df = pd.read_csv('data\wine_dataset.csv')

test = True if df.duplicated(keep=False).all() else False
print(test)
