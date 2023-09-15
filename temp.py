import numpy as np
import pandas as pd
from sklearn import model_selection


def split(x:pd.Series, threshold) -> 'pd.core.indexes.base.Index':
        """
        returns two lists of split indexes, left and right [0,1], [2,3]
        """
        left_indices = x.index[x < threshold]
        right_indices = x.index[x >= threshold]
        return left_indices, right_indices

feature = pd.Series([0,1,0,0,0,1,1,0,0,0,1,1,0,1])
feature2 = pd.Series([1,1,1,1,0,0,0,1,0,0,0,1,0,1])
df = pd.DataFrame({"wind":feature, "outlook":feature2})
print(df)
y = pd.Series([0,0,1,1,1,0,1,0,1,1,1,1,1,0])

df2 = pd.DataFrame({"wind":[1,2,3,4,5,6,7,8,9,10], "outlook":[1,2,3,4,5,6,7,8,9,10]})
y2 = pd.Series([0,0,0,0,1,0,1,0,1,1,1,1,1,0])

def entropy(y:pd.Series):
        """calculates entropy"""
        prob = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in prob if p > 0])

def _information_gain(x:pd.Series, y:pd.Series, treshold) -> float:
        """TODO"""
        base_impurity = entropy(y)
        left_indexes, right_indexes = split(x, treshold)
        if left_indexes.empty or right_indexes.empty:
                return 0
        left_frac= len(left_indexes)/len(y)
        right_frac= len(right_indexes)/len(y)
        left_child_entropy = left_frac*entropy(y.loc[left_indexes])
        right_child_entropy = right_frac*entropy(y.loc[right_indexes])
        return base_impurity - (left_child_entropy + right_child_entropy)


def find_threshold(X, y): #best_split
        """
        set best_ig = 0
        1. loop trough all columns of features
                2. find all unique values in the column
                3. loop trough the unique values (2)
                        4. calculate impurity
                        5. check if new ig better than best_ig, if true change best_ig
        returns index and treshold for were to split
        """
        best_ig = -1
        best_threshold = 0
        for feature in X:
                unique_vals = pd.unique(X[feature]) # array of unique values
                for threshold in unique_vals:
                        ig = _information_gain(X[feature], y, threshold)
                        if ig > best_ig:
                                best_ig = ig
                                best_threshold = threshold
        return best_threshold



print(find_threshold(df, y))
print(find_threshold(df2, y2))
