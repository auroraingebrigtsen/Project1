import numpy as np
import pandas as pd
from sklearn import model_selection
from .treeNode import TreeNode

class DecisionTree:
    def __init__(self, impurity_measure= 'entropy', root=None) -> None:
        self.root = root
        self.impurity_measure = impurity_measure
        self.impurity_measure = impurity_measure.lower()

    def learn(self, X, y, impurity_measure='entropy'):
        """Seperates the data, sets the root and builds the tree. Prunes the tree"""
        self.impurity_measure = impurity_measure
        """
        seed = 3

        # split the data into training data (60%) and pruning + test data (40%)
        X_train, X_prun_test, y_train, y_prun_test = model_selection.train_test_split(
        X, y, test_size=0.4, random_state=seed)

        # Shuffle and split the data into pruning and test sets with a ratio of 0.5/0.5:
        X_prun, X_test, y_prun, y_test = model_selection.train_test_split(
        X_prun_test, y_prun_test, random_state=seed, test_size=0.5)
        """

        # Start recursion
        #self.root = self._build(X_train, y_train)

        self.root = self._build(X,y)
        
        # TODO: call function to prun tree
      
    def _calculate_impurity(self, y:pd.Series):
        """Chooses which function to call based on impurity measure type"""
        if self.impurity_measure == 'entropy':
            return self._entropy(y)
        elif self.impurity_measure == 'gini':
            return self._gini(y)
        else:
            raise ValueError("Invalid impurity_measure parameter")

    def _build(self, X:pd.DataFrame, y:pd.Series, depth=0) -> 'TreeNode':
        """Builds the tree recursively"""
        if np.all(np.unique(y) == y.iloc[0]):  # If all data points have the same label, return a leaf with that label
            return TreeNode(value=y.iloc[0])
        elif X.duplicated(keep=False).all(): # Else, if all data points have identical feature values, return a leaf with the most common label.
            return TreeNode(value=y.mode()[0]) 
        else:  # choose a feature that maximizes the information gain,
            best_threshold, best_feature = self._find_threshold(X, y)
            new_node = TreeNode(feature_index=X.columns.get_loc(best_feature), feature=best_feature, threshold=best_threshold)
            left_indices, right_indices = self._split(X[best_feature], best_threshold)
            
            left_X = X.loc[left_indices].reset_index(drop=True)
            left_y = y.loc[left_indices].reset_index(drop=True)
            
            right_X = X.loc[right_indices].reset_index(drop=True)
            right_y = y.loc[right_indices].reset_index(drop=True)

            left_child = self._build(left_X, left_y, depth=depth+1)
            right_child = self._build(right_X, right_y, depth=depth+1)
            new_node.add_child(left_child)
            new_node.add_child(right_child)
            return new_node
        
    def _information_gain(self, x:pd.Series, y:pd.Series, treshold) -> float:
        """calculates the information gain of a single feature"""
        base_impurity = self._calculate_impurity(y)
        left_indexes, right_indexes = self._split(x, treshold)
        if left_indexes.empty or right_indexes.empty:
                return 0
        left_frac= len(left_indexes)/len(y)
        right_frac= len(right_indexes)/len(y)
        left_child_entropy = left_frac*self._calculate_impurity(y.loc[left_indexes])
        right_child_entropy = right_frac*self._calculate_impurity(y.loc[right_indexes])
        return base_impurity - (left_child_entropy + right_child_entropy)
        
    
    def _find_threshold(self, X, y): #best_split
            """finds the threshold that gives the best information gain"""
            best_ig = -1
            best_threshold = 0
            best_feature = None
            for feature in X:
                    unique_vals = pd.unique(X[feature]) # array of unique values
                    for threshold in unique_vals:
                            ig = self._information_gain(X[feature], y, threshold)
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
        pass

    def predict(self, x:pd.Series):
        if len(self.root.children) == 0:
             return self.root.value
        else:
             return self._predict(x, self.root)
        
    def _predict(self, x:pd.Series, node:'TreeNode'):
        if len(node.children) == 0:
            return node.value
        elif x.iloc[node.feature_index] < node.threshold:
            return self._predict(x, node.children[0])
        else:
            return self._predict(x, node.children[1])
