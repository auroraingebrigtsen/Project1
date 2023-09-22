import numpy as np
import pandas as pd
from decision_tree.decisionTree import DecisionTree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from validation import k_fold_cross_validation
from preprocessing import load_dataset
from evaluation import visualize_cm,performance_metrics

def main():
    X, y = load_dataset('data\wine_dataset.csv', visualize_df=False)

    # split to training and test data
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=123)

    tre = DecisionTree()
    tre.learn(X_train, y_train)
    tre.print_tree()

    # run k fold cross validation to find optimal hyperparameters
    my_model, impurity_measure, pruned = k_fold_cross_validation(X_train, y_train, k=10)
    print(impurity_measure, pruned)

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
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, y_pred)
    print("Sklearn model accuracy: ", sklearn_accuracy)
    

if __name__ == '__main__':
    main()