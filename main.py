import numpy as np
import pandas as pd
from decision_tree.decisionTree import DecisionTree

def prepare_dataset(path):
    """Load dataset"""
    df = pd.read_csv(path)
    X = df.drop(columns=['type'])
    y = df['type']
    # check for empty columns and NAn values
    
    return X, y

def main():
    X, y = prepare_dataset('data\wine_dataset.csv')
    #decision_tree = DecisionTree()
    decision_tree_pruned = DecisionTree()
    #decision_tree.learn(X, y, impurity_measure='entropy', prune=False)
    #decision_tree.print_tree()
    decision_tree_pruned.learn(X, y, impurity_measure='entropy', prune=True)
    decision_tree_pruned.print_tree()
    #print(decision_tree.predict(pd.Series([0.48,1.7,3.19,0.62,9.5])))

if __name__ == '__main__':
    main()