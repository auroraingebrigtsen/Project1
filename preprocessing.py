import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_dataset(path, drop_nan=True, visualize_df=True):
    """Load dataset, drop rows with NaN values, and visualize data"""
    df = pd.read_csv(path, delimiter=",")
    
    # Check for NaN values in the DataFrame
    nan_rows = df[df.isna().any(axis=1)]
    
    if not nan_rows.empty and drop_nan:
        # Drop rows with NaN values
        df.dropna(inplace=True)
        print(f"Dropped rows with NaN values.")
    
    # Separate features (X) and target (y)
    X = df.drop(columns=['type'])
    y = df['type']
    if visualize_df:
        _visualize(df)
    return X, y
    
def _visualize(df):
    # Pairplot for visualizing relationships between numerical features
    sns.pairplot(df, hue='type', markers=["o", "s"], palette="husl")
    plt.show()
