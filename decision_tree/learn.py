import numpy as np
from sklearn import model_selection
from .decisionTree import TreeNode, DecisionTree
from .impurity_calculator import ImpurityCalculator

def learn(X, y, impurity_measure='entropy') -> DecisionTree:

    seed = 3

    # split the data into training data (60%) and pruning + test data (40%)
    X_train, X_prun_test, y_train, y_prun_test = model_selection.train_test_split(
    X, y, test_size=0.4, random_state=seed)

    # Shuffle and split the data into pruning and test sets with a ratio of 0.5/0.5:
    X_prun, X_test, y_prun, y_test = model_selection.train_test_split(
    X_prun_test, y_prun_test, random_state=seed, test_size=0.5)

    impurity_calculator = ImpurityCalculator(impurity_measure=impurity_measure)
    tree_model = DecisionTree(impurity_calculator)
    tree_model.root = tree_model.build(X_train, y_train)




    """flyttes til decisionTree
    impurity_calculator = ImpurityCalculator(impurity_measure=impurity_measure)

    if np.all(np.unique(y) == y[0]):  # If all data points have the same label
        print("all data points have the same label")
        return TreeNode(value=y[0])
    elif 1!=1: # Else, if all data points have identical feature values:
        print ("all data points have identical feature values")
        #return a leaf with the most common label.
    else:  # choose a feature that maximizes the information gain,
        impurity = impurity_calculator.calculate_impurity(X, y)
        feature = max(zip(impurity.values(), impurity.keys()))[1] # find key name of max value in dict
        node = TreeNode(feature=feature)
        if tree.root is None:
            tree.root = node
        else:
            pass
        

        #split df[df['max.key'].mean() > 0] for å splitte df i to

        print(impurity)
    

        #for cols in dataset: measure impurity
        #information_gain = impurity_calculator.calculate_ig(dec_impurity, other_impurity)
        print("choose a feature that maximizes the information gain")

"""