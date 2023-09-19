import numpy as np
import pandas as pd
from sklearn import model_selection
from .treeNode import TreeNode
import time

class DecisionTree:
    def __init__(self) -> None:
        self.root = None
        self.impurity_measure = None

    def learn(self, X:pd.DataFrame, y:pd.Series, impurity_measure: str='entropy', prune: bool=False, test_size: float=0.3):
        """starts the recursion to build a model. Prunes the tree if parameter prune=True"""
        self.impurity_measure = impurity_measure.lower()
        
        if prune:
            # split the data into training data and pruning data, 70-30 ratio by default
            X_train, X_prune, y_train, y_prune = model_selection.train_test_split(
            X, y, test_size=test_size)
            # start recursion on training data
            self.root = self._build(X_train, y_train)
            self._prune(y_train, X_prune, y_prune, self.root)
        else:
            # start recursion on training data
            self.root = self._build(X, y)

    def predict(self, x:pd.Series) -> int:
        """Predicts the y label given a series of features x"""
        node = self.root
        while node.value is None:  # while the node is a decision node
            if x.iloc[node.feature_index] < node.threshold:  # check if the node feature value is below threshold
                node = node.children[0]  # go into left child
            else:
                node = node.children[1]  # else, go into right child
        return node.value

    def _calculate_impurity(self, y:pd.Series):
        """Chooses which function to call based on impurity measure type"""
        if self.impurity_measure == 'entropy':
            return self._entropy(y)
        elif self.impurity_measure == 'gini':
            return self._gini(y)
        else:
            raise ValueError("Invalid impurity_measure parameter")

    def _build(self, X:pd.DataFrame, y:pd.Series) -> 'TreeNode':
        """Builds the tree recursively. Base cases: If all data points have the same label, return a leaf with that label. 
          Else, if all data points have identical feature values, return a leaf with the most common label (first if several) 
          Else, find feature that maximizes the information gain, find indices of split this split, use indices to split the data into two children. Recursively call 
          the method on the children. Connect the children to the current node and return the current node."""
        if np.all(np.unique(y) == y.iloc[0]):
            return TreeNode(value=y.iloc[0])
        elif X.duplicated(keep=False).all():
            return TreeNode(value=y.mode()[0])
        else:  
            best_threshold, best_feature = self._find_threshold(X, y)
            new_node = TreeNode(feature_index=X.columns.get_loc(best_feature), feature=best_feature, threshold=best_threshold)
            left_indices, right_indices = self._split(X[best_feature], best_threshold)
            
            left_X = X.loc[left_indices]
            left_y = y.loc[left_indices]  
            
            right_X = X.loc[right_indices] 
            right_y = y.loc[right_indices] 

            left_child = self._build(left_X, left_y)  # recursively call children
            right_child = self._build(right_X, right_y)

            new_node.add_child(left_child, y_indexes=left_y)
            new_node.add_child(right_child, y_indexes=right_y)

            return new_node
        
    def _gain(self, x:pd.Series, y:pd.Series, threshold) -> float:
        """Calculates the information gain of a single feature. Takes impurity for whole feature - impurity for each child"""
        base_impurity = self._calculate_impurity(y)  # calculate impurity for the whole feature
        left_indexes, right_indexes = self._split(x, threshold) 
        if left_indexes.empty or right_indexes.empty:  # check that none of the df will be emtpy
                return 0
        p_left= len(left_indexes)/len(y)  # find fraction of rows below and over/equal to threshold
        p_right= len(right_indexes)/len(y)
        left_child_impurity = p_left*self._calculate_impurity(y.loc[left_indexes])  # calculate impurity for children
        right_child_impurity = p_right*self._calculate_impurity(y.loc[right_indexes])
        return base_impurity - (left_child_impurity + right_child_impurity)  # calculate gain
        
    def _find_threshold(self, X:pd.DataFrame, y:pd.Series):
            """Finds the threshold that gives the best information gain. Loops through each unique value in each row in a df.
            Checks which gain the unique value gives and compares to the best value encountred yet"""
            best_ig = -1
            best_threshold = 0
            best_feature = None
            for feature in X:
                    unique_vals = pd.unique(X[feature]) # array of unique values
                    # Calculate the minimum and maximum values in the unique values
                    #min_val, max_val = min(unique_vals), max(unique_vals)
        
                    # Generate thresholds within the specified range,
                    #thresholds = np.linspace(min_val, max_val, 10)
                    for threshold in unique_vals:
                            ig = self._gain(X[feature], y, threshold)
                            if ig > best_ig:
                                    best_ig = ig
                                    best_threshold = threshold
                                    best_feature = feature
            return best_threshold, best_feature

    def _split(self, x:pd.Series, threshold) -> 'pd.core.indexes.base.Index':
        """Returns the indexes of rows in a series above a treshold and below or equals to the threshold"""
        left_indices = x.index[x < threshold] 
        right_indices = x.index[x >= threshold]
        return left_indices, right_indices
        
    def _entropy(self, y:pd.Series) -> float:
        """Calculates entropy"""
        prob = np.bincount(y) / len(y)
        entropy =  -np.sum([p * np.log2(p) for p in prob if p > 0])
        return entropy
    
    def _gini(self, y:pd.Series) -> float:
         """Calculates gini index"""
         prob = np.bincount(y) / len(y)
         gini = 1 - np.sum(p**2 for p in prob if p > 0)
         return gini

    def print_tree(self, node: 'TreeNode', direction="", depth=0) -> None:
        """Prints the tree recursively"""
        if node is None:
            return
        
        if node.value is not None:
            print(f"{'|   ' * depth}{direction} Leaf node value: {node.value}")  # Print leaf node
        else:
            print(f"{'|   ' * depth}{direction} Decision node on: {node.feature}, threshold {node.threshold}")  # Print decision node

        if node.children:
            for child, child_direction in zip(node.children, ["Left: ", "Right: "]):
                self.print_tree(child, child_direction, depth + 1)

    def _prune(self, y_train:pd.Series, X_prune:pd.DataFrame, y_prune:pd.Series, node: 'TreeNode') -> None:
        """Performs reduced error pruning on the Tree. Calls the method recursively for each child. If both children is leaf nodes
        seperate the part of y that meets the node's conditions, make node a leaf node with mode as value and
        use validation data to test accuracy. If the node performs better as a leaf node, make it a leaf node, 
        by keeping the value and removing the children. Else, go back to being a decision node"""
        if node.value is None:  # only consider decision nodes
            for child in node.children: 
                self._prune(y_train, X_prune, y_prune, child)
            if node.children[0].value is not None and node.children[1].value is not None:
                    dec_node_errors = self._predict_df(X_prune, y_prune)
                    labels = y_train.iloc[node.y_indexes]
                    node.value = labels.mode()[0]
                    leaf_errors = self._predict_df(X_prune, y_prune)  # TODO find better name
                    if leaf_errors <= dec_node_errors:
                        node.remove_children()
                    else:
                        node.value = None

    def _predict_df(self, X:pd.DataFrame, y:pd.Series) -> int:
        """Iterates over the rows in a df and predicts the label. If label does not match true label, count 1 error"""
        errors = 0
        for index in range(len(X)):
             y_hat = self.predict(X.iloc[index])
             if y_hat != y.iloc[index]:
                  errors += 1
        return errors
            
