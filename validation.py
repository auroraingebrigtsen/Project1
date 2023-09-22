import numpy as np
from sklearn.model_selection import KFold
from decision_tree.decisionTree import DecisionTree
import pandas as pd
import time

def k_fold_cross_validation(X, y, k: int=10) -> DecisionTree:
    """Seperates the data into k folds. Iterates over k to find accuracy with each bin as test data. K cannot be 1"""
    features = X.columns
    accuracy_scores = {}
    kf = KFold(n_splits=k)
    average_time={}

    for impurity_measure in ['entropy', 'gini']:
        for prune in [False, True]:
            combo_scores = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Create and train your model using X_train and y_train
                start = time.time()
                model = DecisionTree()
                model.learn(pd.DataFrame(X_train, columns=features), pd.Series(y_train), impurity_measure=impurity_measure, prune=prune)
                stop = time.time()

                # Evaluate your model on the test set
                score = model.evaluate(X_test, y_test)

                # Store the evaluation score for this fold
                combo_scores.append(score)
                average_time[impurity_measure, prune] = stop - start

            # Calculate and return the mean and standard deviation of the fold scores
            accuracy_scores[model] = np.mean(combo_scores)
    best_model = max(accuracy_scores, key=accuracy_scores.get)
    #print(average_time)
    return best_model, best_model.impurity_measure, best_model.pruned

