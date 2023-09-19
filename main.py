import numpy as np
import pandas as pd
from decision_tree.decisionTree import DecisionTree
from sklearn import model_selection
import time

def prepare_dataset(path):
    """Load dataset"""
    df = pd.read_csv(path)
    X = df.drop(columns=['type'])
    y = df['type']    
    return X, y


def main():
    X, y = prepare_dataset('data\wine_dataset.csv')

    # split to training and test data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.1)

    for impurity_measure in ['entropy', 'gini']:
        for prune in [False, True]:
            print(f'Start build of tree with impurity measure = {impurity_measure} and pruning = {prune}')
            start = time.time()
            decision_tree = DecisionTree()
            decision_tree.learn(X_train, y_train, impurity_measure=impurity_measure, prune=prune)
            end = time.time()
            print(f'Learning took {end-start} seconds')
            total_predicted = X_test.shape[0]
            errors = decision_tree._predict_df(X_test, y_test)
            print(f'Model accuracy {(total_predicted-errors)/total_predicted}')

if __name__ == '__main__':
    main()