import numpy as np
import pandas as pd
from sklearn import model_selection
from .treeNode import TreeNode

class DecisionTree:
    def __init__(self, impurity_measure= 'entropy', root=None) -> None:
        self.root = root
        self.impurity_measure = impurity_measure
        self.impurity_measure = impurity_measure.lower()

    def learn(self, X, y, impurity_measure='entropy', prune=False, test_size=0.3):
        """Seperates the data, sets the root and builds the tree. Prunes the tree"""
        self.impurity_measure = impurity_measure
        seed = 3

        # split the data into training data (70%) and pruning data (30%)
        X_train, X_prune, y_train, y_prune = model_selection.train_test_split(
        X, y, test_size=test_size, random_state=seed)

        # Start recursion
        self.root = self._build(X_train, y_train)
        self.print_tree()

        if prune:
             self._prune(y_train, X_prune, y_prune, self.root)
        
        self.print_tree()

    def _calculate_impurity(self, y:pd.Series):
        """Chooses which function to call based on impurity measure type"""
        if self.impurity_measure == 'entropy':
            return self._entropy(y)
        elif self.impurity_measure == 'gini':
            return self._gini(y)
        else:
            raise ValueError("Invalid impurity_measure parameter")

    def _build(self, X:pd.DataFrame, y:pd.Series) -> 'TreeNode':
        """Builds the tree recursively"""
        if np.all(np.unique(y) == y.iloc[0]):  # If all data points have the same label, return a leaf with that label
            return TreeNode(value=y.iloc[0])
        elif X.duplicated(keep=False).all(): # Else, if all data points have identical feature values, return a leaf with the most common label.
            return TreeNode(value=y.mode()[0]) 
        else:  # choose a feature that maximizes the information gain,
            best_threshold, best_feature = self._find_threshold(X, y)
            new_node = TreeNode(feature_index=X.columns.get_loc(best_feature), feature=best_feature, threshold=best_threshold)
            left_indices, right_indices = self._split(X[best_feature], best_threshold)
            
            left_X = X.loc[left_indices] #.reset_index(drop=True)
            left_y = y.loc[left_indices] #.reset_index(drop=True)
            
            right_X = X.loc[right_indices] #.reset_index(drop=True)
            right_y = y.loc[right_indices] #.reset_index(drop=True)

            left_child = self._build(left_X, left_y)
            right_child = self._build(right_X, right_y)
            new_node.add_child(left_child, y_indexes=left_y)
            new_node.add_child(right_child, y_indexes=right_y)
            return new_node
        
    def _gain(self, x:pd.Series, y:pd.Series, treshold) -> float:
        """calculates the information gain of a single feature"""
        base_impurity = self._calculate_impurity(y)
        left_indexes, right_indexes = self._split(x, treshold)
        if left_indexes.empty or right_indexes.empty:
                return 0
        left_p= len(left_indexes)/len(y)
        right_p= len(right_indexes)/len(y)
        left_child_impurity = left_p*self._calculate_impurity(y.loc[left_indexes])
        right_child_impurity = right_p*self._calculate_impurity(y.loc[right_indexes])
        return base_impurity - (left_child_impurity + right_child_impurity)
        
    
    def _find_threshold(self, X, y): #best_split
            """finds the threshold that gives the best information gain"""
            best_ig = -1
            best_threshold = 0
            best_feature = None
            for feature in X:
                    unique_vals = pd.unique(X[feature]) # array of unique values
                    for threshold in unique_vals:
                            ig = self._gain(X[feature], y, threshold)
                            if ig > best_ig:
                                    best_ig = ig
                                    best_threshold = threshold
                                    best_feature = feature
            return best_threshold, best_feature

    def _split(self, x:pd.Series, threshold) -> 'pd.core.indexes.base.Index':
        """
        returns two lists of split indexes, left and right [0,1], [2,3]
        """
        left_indices = x.index[x < threshold]
        right_indices = x.index[x >= threshold]
        return left_indices, right_indices
        
    def _entropy(self, y:pd.Series):
        """calculates entropy"""
        prob = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in prob if p > 0])
    
    def _gini(self, y:pd.Series):
         """calculates gini index"""
         prob = np.bincount(y) / len(y)
         return 1 - np.sum(prob**2)
        
    def print_tree(self):
        self._print_tree_recursive(self.root)

    def _print_tree_recursive(self, node):
        if node is None:
            return
        if node.value is not None:
            print("Predict:", node.value)
        else:
            print(f"Feature {node.feature} <= {node.threshold}")
            print("Left:")
            self._print_tree_recursive(node.children[0])
            print("Right:")
            self._print_tree_recursive(node.children[1])

    def predict(self, x:pd.Series):
        node = self.root
        while node.value is None:
            if x.iloc[node.feature_index] < node.threshold:
                node = node.children[0]
            else:
                node = node.children[1]
        return node.value

    def _prune(self, y_train, X_prune, y_prune, node: 'TreeNode') -> None:
        """
        1. check if node has two leaf children
        2. if yes, find y data most common label without the split
        3. set a decnode_asleaf = most common label
        4. check this how the tree does with this new leaf in training data == df_prune.labels, and validation data using predict
        5. find sum of errors in both
        5. if sum on val data with decnode_asleaf <= sum on val data with decnode as decnode
        6. children.remove
        7. else keep children, remove this leaf
        """
        if node.value is None:  # Only consider non-leaf nodes
            for child in node.children:
                self._prune(y_train, X_prune, y_prune, child)
            if node.children[0].value is not None and node.children[1].value is not None:
                    decnode_erros = self._predict_df(X_prune, y_prune)
                    labels = y_train.iloc[node.y_indexes]
                    node.value = labels.mode()[0]
                    leaf_errors = self._predict_df(X_prune, y_prune)
                    print("Leaf error:", leaf_errors, "Decnode error:", decnode_erros)
                    if leaf_errors <= decnode_erros:
                        node.remove_children()
                        print("Removed children")
                    else:
                        node.value = None

    def _predict_df(self, X, y):
        errors = 0
        for index in range(len(X)):
             y_hat = self.predict(X.iloc[index])
             if y_hat != y.iloc[index]:
                  errors += 1
        return errors
            
