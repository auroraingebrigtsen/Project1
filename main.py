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
    decision_tree = DecisionTree()
    decision_tree.learn(X, y, impurity_measure='gini')
    errors = 0
    correct = 0
    for index in range(len(X)):
        y_hat = decision_tree.predict(X.loc[index])
        if y_hat == y[index]:
            correct += 1
        else:
            errors +=1
    print("Errors: ", errors, "\nCorrect: ", correct)

if __name__ == '__main__':
    main()