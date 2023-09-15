import numpy as np
import pandas as pd
from decision_tree.decisionTree import DecisionTree

def prepare_dataset(path):
    """Load dataset"""
    df = pd.read_csv(path)
    print(df.head(10))
    X = df.drop(columns=['type'])
    y = df['type']
    # check for empty columns and NAn values
    
    return X, y

def main():
    X, y = prepare_dataset('data\wine_dataset.csv')
    decision_tree = DecisionTree()
    decision_tree.learn(X, y, impurity_measure='entropy')

if __name__ == '__main__':
    main()