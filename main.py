import numpy as np
import pandas as pd
from decision_tree.decisionTree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from validation import k_fold_cross_validation
from preprocessing import load_dataset
from evaluation import visualize_cm,performance_metrics
import time

def main():
    X, y = load_dataset('data\wine_dataset.csv', visualize_df=False)

    # split to training and test data
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=123)

    # fit a simple decision tree
    tree = DecisionTree()
    tree.learn(X_train, y_train)
    tree.print_tree()

    # run k fold cross validation to find optimal hyperparameters
    my_model, impurity_measure, pruned = k_fold_cross_validation(X_train, y_train, k=10)
    print(f'best params: impurity measure = {impurity_measure}, prune = {pruned}')

    # train my model with the entire training set.
    my_model.learn(X_train, y_train, impurity_measure=impurity_measure, prune=pruned)

    # get model's prediction on test data
    predictions = my_model.evaluate(X_test, y_test, get_pred_series=True)

    # visualize confusion matrix for the results
    visualize_cm(y_test, predictions)

    # print performance metrics 
    metrics = performance_metrics(y_test, predictions)
    for key, value in metrics.items():
        print(key, value)

    # comparison to SKlearn decision tree
    start_clf = time.time()
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, y_train)
    stop_clf = time.time()
    start_clf_predict = time.time()
    y_pred = clf.predict(X_test)
    stop_clf_predict = time.time()
    print(f'Sklearn model build time: {stop_clf - start_clf}, predict time {(stop_clf_predict - start_clf_predict)/len(X_test)}')
    
    # print performance metrics for sklearn model
    metrics = performance_metrics(y_test, y_pred)
    for key, value in metrics.items():
        print(key, value)

    """
    # used to measure prediction time
    start = time.time()
    for index in range(len(X_test)):
            tree.predict(X_test.iloc[index])
    stop = time.time()
    print((stop/start)/len(X_test))
    """

    
if __name__ == '__main__':
    main()