import pandas as pd

# Create a sample DataFrame
data = {'feature': [106, 260, 360, 406, 50, 60, 70, 80, 90, 100], 'values': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
df = pd.DataFrame(data)

# Split the DataFrame into 10 bins
bins = pd.qcut(df['values'], q=5, labels=False)

# Add the bin labels to the DataFrame
df['bins'] = bins

# Display the result
print(df)
