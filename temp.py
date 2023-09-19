import numpy as np
import pandas as pd
from sklearn import model_selection
from decision_tree.treeNode import TreeNode
import time

class DecisionTree:
    def __init__(self) -> None:
        self.root = None
        self.impurity_measure = None

    def learn(self, X:pd.DataFrame, y:pd.Series, impurity_measure: str='entropy', prune: bool=False, test_size: float=0.3):
        """starts the recursion to build a model. Prunes the tree if parameter prune=True"""
        self.impurity_measure = impurity_measure.lower()
        
        if prune:
            seed = 321
            # split the data into training data and pruning data, 70-30 ratio by default
            X_train, X_prune, y_train, y_prune = model_selection.train_test_split(
            X, y, test_size=test_size, random_state=seed)
            print(X_prune)
            # start recursion on training data
            self.root = self._build(X_train, y_train)
            self._prune(y_train, X_prune, y_prune, self.root)
        else:
            # start recursion on training data
            start = time.time()
            self.root = self._build(X, y)
            end = time.time()
            print("Tree build took ", end-start)

    def predict(self, x:pd.Series) -> int:
        """predicts the y label given a set of features x"""
        node = self.root  # start at the root node
        while node.value is None:  # while the node is a decision node
            if x.iloc[node.feature_index] < node.threshold:  # check if the node feature value is below threshold
                node = node.children[0]  # go into left node
            else:
                node = node.children[1]  # else, node feature value is higher or equal than threshold, go into right child
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
        """Builds the tree recursively"""
        if np.all(np.unique(y) == y.iloc[0]):  # if all data points have the same label
            return TreeNode(value=y.iloc[0])  # return a leaf with that label
        elif X.duplicated(keep=False).all(): # else, if all data points have identical feature values
            return TreeNode(value=y.mode()[0])  # return a leaf with the most common label, first if several
        else:  # else, choose a feature that maximizes the information gain
            print("Entered else")
            best_threshold, best_feature = self._find_threshold(X, y)
            print("First output", best_feature, best_threshold, X.loc[:, best_feature])
            new_node = TreeNode(feature_index=X.columns.get_loc(best_feature), feature=best_feature, threshold=best_threshold)
            left_indices, right_indices = self._split(X.loc[:,best_feature], best_threshold)
            
            left_X = X.loc[left_indices] 
            left_y = y.loc[left_indices] 
            
            right_X = X.loc[right_indices] 
            right_y = y.loc[right_indices] 

            print("INDEX",right_indices, left_indices)

            left_child = self._build(left_X, left_y)
            right_child = self._build(right_X, right_y)

            new_node.add_child(left_child, y_indexes=left_y)
            new_node.add_child(right_child, y_indexes=right_y)

            return new_node
        
    def _gain(self, x:pd.Series, y:pd.Series, treshold) -> float:
        """calculates the information gain of a single feature"""
        base_impurity = self._calculate_impurity(y)  # calculate gain 
        left_indexes, right_indexes = self._split(x, treshold)
        if left_indexes.empty or right_indexes.empty:
                return 0
        p_left= len(left_indexes)/len(y)
        p_right= len(right_indexes)/len(y)
        left_child_impurity = p_left*self._calculate_impurity(y.loc[left_indexes])
        right_child_impurity = p_right*self._calculate_impurity(y.loc[right_indexes])
        return base_impurity - (left_child_impurity + right_child_impurity)
        
    def _find_threshold(self, X, y):
            """finds the threshold that gives the best information gain"""
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
        """returns the indices of elements in a series above a treshold and below or equals to the threshold"""
        left_indices = x.index[x < threshold]
        right_indices = x.index[x >= threshold]
        return left_indices, right_indices
        
    def _entropy(self, y:pd.Series) -> float:
        """calculates entropy"""
        prob = np.bincount(y) / len(y)
        entropy =  -np.sum([p * np.log2(p) for p in prob if p > 0])
        return entropy
    
    def _gini(self, y:pd.Series):
         """calculates gini index"""
         prob = np.bincount(y) / len(y)
         gini = 1 - np.sum(p**2 for p in prob if p > 0)
         return gini
    
    def print_tree(self, node: 'TreeNode', direction="", depth=0, show_level=True) -> None:
         if node is None:
                return

         if node.value is not None:
                print(f"{'|   ' * depth}{direction} Leaf node value: {node.value}")  # Print leaf node
         else:
                if show_level:
                        print(f"{'|   ' * depth}{direction} Decision node on: {node.feature}, threshold {node.threshold}")  # Print decision node
                else:
                        print(f"{'|   ' * depth}{direction}Decision node on: {node.feature}, threshold {node.threshold}")  # Print decision node

        # Recursively print children
         if node.children:
                child_directions = ["Left: ", "Right: "]
                for child, child_direction in zip(node.children, child_directions):
                        self.print_tree(child, child_direction, depth if show_level else depth + 1, show_level=False)



    def _prune(self, y_train, X_prune, y_prune, node: 'TreeNode') -> None:
        """reduced error pruning"""
        if node.value is None:  # only consider decision nodes
            for child in node.children: 
                self._prune(y_train, X_prune, y_prune, child)  # calls the method recursively for each child
            if node.children[0].value is not None and node.children[1].value is not None:  # if both children is leaf nodes
                    decnode_erros = self._predict_df(X_prune, y_prune)  # use validation data to test accuracy
                    labels = y_train.iloc[node.y_indexes]
                    print(node.feature, node.y_indexes)  # seperate the part of y that meets the node's conditions
                    node.value = labels.mode()[0]  # make node a leaf node with mode as value
                    leaf_errors = self._predict_df(X_prune, y_prune)  # use validation data to test accuracy 
                    if leaf_errors <= decnode_erros:  # if the node performs better as a leaf node
                        node.remove_children()  # make it a leaf node, by keeping the value and removing the children
                    else:
                        node.value = None  # else, go back to being a decision node

    def _predict_df(self, X:pd.DataFrame, y:pd.Series) -> int:
        """iterates over the rows in a df and predicts the label. If label does not match true label, count 1 error"""
        errors = 0
        for index in range(len(X)):
             y_hat = self.predict(X.iloc[index])
             if y_hat != y.iloc[index]:
                  errors += 1
        return errors
            


data = {
    'Outlook': [1, 1, 2, 3, 3, 3, 2, 1, 1, 3, 1, 2, 2, 3],
    'Temp.': [1, 1, 1, 2, 3, 3, 3, 2, 3, 2, 2, 2, 1, 2],
    'Humidity': [1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1],
    'Wind': [1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2],
    'Decision': [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)
# Display the DataFrame

X = df.drop(columns='Decision')
y = df['Decision']

print(X)
print(X.to_numpy())

X_train, X_test, y_train, y_test = model_selection.train_test_split(
X, y, test_size=0.1, random_state=123)

tre = DecisionTree()
#tre.learn(X_train, y_train, prune=True)
#print(tre.root.feature, tre.root.children[0].feature, tre.root.children[1].feature)