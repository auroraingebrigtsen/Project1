import numpy as np
from math import log2
import pandas as pd

class ImpurityCalculator:
    def __init__(self, impurity_measure='entropy', baseevent_impurity=None) -> None:
        self.impurity_measure = impurity_measure.lower()
        self.baseevent = baseevent_impurity
        # make num of classification count 

    def calculate_impurity(self, X, y) -> dict:
        """Chooses which function to call based on impurity measure type"""
        if self.impurity_measure == 'entropy':
            return self.calculate_entropy(X, y)
        elif self.impurity_measure == 'gini':
            return self.calculate_gini(X, y)
        else:
            raise ValueError("Invalid impurity_measure parameter")
        
    def entropy(self, y:pd.Series):
        prob = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in prob if p > 0])
    
    def gini(self, y:pd.Series):
         prob = np.bincount(y) / len(y)
         return 1 - np.sum(prob**2)
    
    def information_gain():
         """
         for 
         """
    
    def find_treshold():
        """
        set best_ig = 0
        1. loop trough all columns of features
            2. find all unique values in the column
                3. loop trough the unique values (2)
                    4. calculate impurity
                    5. check if new ig better than best_ig, if true change best_ig
        """
        
    def bin_series(self, x:pd.Series, num_bins=2):
        """Binning the continuous variable into bins based on column mean"""
        unique_values = x.unique()
        if len(unique_values) != num_bins or not all(val in [0, 1] for val in unique_values):
            mean = x.mean()
            x = x.apply(lambda val: 1 if val > mean else 0)
        return x

    def calculate_entropy(self, X, y:pd.Series)->np.array:
            """Calculates the information gain of a dataframe X and a series y"""
            ig_dict = {}
            #TODO optimize this 
            # Calculate base entropy
            base_0 = y.value_counts()[0]  # TODO consider using get() instead of []
            base_1 = y.value_counts()[1]
            base_sum = len(y)
            if base_sum <= 0:
                raise ValueError("base_sum must be greater than zero. Division by zero error.")
            baseevent_impurity = -(base_0/base_sum*np.log2(base_0/base_sum)+base_1/base_sum*np.log2(base_1/base_sum))
            for column in X: # kan kanskje bruke np apply along axis istedet for for loop
                    x = self.bin_series(X[column])
                    p_0, p_1 = self.p(x)
                    h_0, h_1 = self.h(x, y)
                    entropy = p_0 * h_0 + p_1 * h_1
                    ig = baseevent_impurity - entropy
                    ig_dict[column] = ig
            return ig_dict


    def h(self, x: pd.Series, y:pd.Series):
            temp_df = pd.DataFrame({'F': x, 'L': y})
            feature1_count = len(temp_df[temp_df['F'] == 0])
            feature2_count = len(temp_df[temp_df['F'] == 1])
            feature1 = len(temp_df[(temp_df['F'] == 0) & (temp_df['L'] == 0)])
            frac1 = feature1/feature1_count if feature1_count != 0 else 0
            feature2 = len(temp_df[(temp_df['F'] == 1) & (temp_df['L'] == 0)])
            frac2 = feature2/feature2_count if feature2_count != 0 else 0
            h_0 = -(frac1 * np.log2(frac1) + (1-frac1) * np.log2(1 - frac1))
            h_1 = -(frac2 * np.log2(frac2) + (1-frac2) * np.log2(1 - frac2)) 
            return h_0, h_1



    def p(self, x:pd.Series):
            total_count = len(x)
            if total_count == 0:
                    return 0, 0  # Handle the case of an empty array
            p_0 = (x == 0).sum()/total_count
            p_1 = (x == 1).sum()/total_count
            return p_0, p_1


    def calculate_gini(self, y):
        # Calculate Gini impurity
        # Add your Gini impurity calculation code here
        pass

    #def TODO: make string features and labels into numerous values
