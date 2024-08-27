"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from metrics import *


@dataclass
class Next:
    """
    A class representing the next node in the decision tree.
    """
    name: str
    ptr: 'Node' = None

class Node:
    """
    A class representing a node in the decision tree.
    """
    def __init__(self, feature=None, threshold=None, depth=0, decision=None):
        self.feature = feature
        self.threshold = threshold
        self.depth = depth
        self.decision = decision
        self.left_child = None
        self.right_child = None
        self.nexts = []

    def add_in_nexts(self, name, ptr=None):
        self.nexts.append(Next(name=name, ptr=ptr))

class DecisionTree:
    """
    A class representing the Decision Tree model.
    """
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int  

    def __init__(self, criterion='information_gain', max_depth=4):
        self.criterion = criterion
        self.max_depth = max_depth
        self.Root = None    

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree.
        """

        y = pd.Series(y)  # Convert y to Pandas Series
        X = pd.DataFrame(X)  # Convert X to DataFrame

        def _build_tree(X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
            node = Node(depth=depth)

            # Base cases for stopping the recursion
            if y.nunique() == 1:
                node.decision = y.mode()[0]
                return node

            if depth >= self.max_depth or X.empty:
                node.decision = y.mean() if self.check_ifreal(y) else y.mode()[0]
                return node

            first_feature = X.columns[0]
            if not self.check_ifreal(X[first_feature]):
                # Discrete feature
                best_feature = self.opt_split_attribute(X, y, self.criterion)
                node.feature = best_feature
                unique_values = X[best_feature].unique() #here taking unique values for example outlook has three unique values
                                                             # sunny,overcast,rain
                for val in unique_values:
                    subset_X, subset_y = self.split_data(X, y, best_feature, val) #subsets created for different features of outlook
                    child_node = _build_tree(subset_X, subset_y, depth + 1)
                    node.add_in_nexts(name=val, ptr=child_node)

            else:
                # Continuous feature
                if self.check_ifreal(y):
                    # Real Input, Real Output
                    best_feature, split_val = self.opt_split_real_mse(X, y)
                else:
                    # Real Input, Discrete Output
                    best_feature, split_val = self.opt_split_real(X, y) 
                    
                node.feature = best_feature
                node.threshold = split_val

                left_X, left_y, right_X, right_y = self.split_data_continuous(X, y, best_feature, split_val)
                
                node.left_child = _build_tree(left_X, left_y, depth + 1)
                node.right_child = _build_tree(right_X, right_y, depth + 1)

            return node
        
        self.Root = _build_tree(X, y, depth=0)

    def split_data(self, X: pd.DataFrame, y: pd.Series, feature: str, value: Union[str, float]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Splits the data based on a discrete feature and its value.
        """
        subset_X = X[X[feature] == value]
        subset_y = y[X[feature] == value]
        return subset_X, subset_y

    def split_data_continuous(self, X: pd.DataFrame, y: pd.Series, feature: str, threshold: float) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Splits the data based on a continuous feature and a threshold.
        """
        left_X = X[X[feature] <= threshold]
        left_y = y[X[feature] <= threshold]
        right_X = X[X[feature] > threshold]
        right_y = y[X[feature] > threshold]
        return left_X, left_y, right_X, right_y
    
    def opt_split_real(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, float]:
        """
        Function to find the best split for continuous features for classification (Real Input, Discrete Output).
        """
        best_gain = -np.inf
        best_split = None
        best_feature = None

        for feature in X.columns:
            sorted_values = np.sort(X[feature].unique())
            split_candidates = (sorted_values[:-1] + sorted_values[1:]) / 2

            for split_val in split_candidates:
                left_y = y[X[feature] <= split_val]
                right_y = y[X[feature] > split_val]

                if self.criterion == "information_gain":
                    parent_criterion = entropy(y)
                    left_criterion = entropy(left_y)
                    right_criterion = entropy(right_y)
                elif self.criterion == "gini_index":
                    parent_criterion = gini_index(y)
                    left_criterion = gini_index(left_y)
                    right_criterion = gini_index(right_y)
                else:
                    raise ValueError(f"Unknown criterion: {self.criterion}")

                gain = parent_criterion - (
                    (len(left_y) / len(y)) * left_criterion +
                    (len(right_y) / len(y)) * right_criterion
                )

                if gain > best_gain:
                    best_gain = gain
                    best_split = split_val
                    best_feature = feature

        return best_feature, best_split
    
    def opt_split_real_mse(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, float]:
        """
        Function to find the best split for continuous features for regression (Real Input, Real Output).
        """
        best_gain = -np.inf
        best_split = None
        best_feature = None

        for feature in X.columns:
            sorted_values = np.sort(X[feature].unique())
            split_candidates = (sorted_values[:-1] + sorted_values[1:]) / 2

            for split_val in split_candidates:
                left_y = y[X[feature] <= split_val]
                right_y = y[X[feature] > split_val]

                parent_mse = self.mse(y, [y.mean()] * len(y))
                left_mse = self.mse(left_y, [left_y.mean()] * len(left_y))
                right_mse = self.mse(right_y, [right_y.mean()] * len(right_y))
                
                weighted_avg_mse = (len(left_y) / len(y)) * left_mse + (len(right_y) / len(y)) * right_mse
                gain = parent_mse - weighted_avg_mse

                if gain > best_gain:
                    best_gain = gain
                    best_split = split_val
                    best_feature = feature

        return best_feature, best_split
    
    def mse(self, y: pd.Series, y_pred: np.ndarray) -> float:
        """
        Calculates Mean Squared Error (MSE) between actual and predicted values.
        """
        return np.mean((y - y_pred) ** 2)

    
    def opt_split_attribute(self, X: pd.DataFrame, y: pd.Series, criterion='information_gain') -> str:
        """
        Function to find the best attribute to split on for discrete features.
        """
        best_feature = None

        max_gain = -float('inf')
        
        for feature in X.columns:
            gain = information_gain(X[feature], y, criterion)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
        
        return best_feature
    
    def check_ifreal(self, y: pd.Series) -> bool:
        """
        Function to check if the given series has real or discrete values.
        Returns True if the values are real, otherwise False.
        """
        return pd.api.types.is_numeric_dtype(y)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """
        Function to make predictions using the trained decision tree.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        def predict_from_node(X_row: pd.Series, node: Node) -> Union[str, float]:
            if node.decision is not None:
                return node.decision
            
            if node.feature is not None:
                # Continuous feature
                if node.threshold is not None:
                    if X_row[node.feature] <= node.threshold:
                        if node.left_child:
                            return predict_from_node(X_row, node.left_child)
                    else:
                        if node.right_child:
                            return predict_from_node(X_row, node.right_child)
                
                # Discrete feature
                for next_node in node.nexts:
                    if next_node.name == X_row[node.feature]:
                        if next_node.ptr:
                            return predict_from_node(X_row, next_node.ptr)
            
            # If no valid child node is found, fall back to a majority class or mean
            print(f"Warning: No valid child node found for feature '{node.feature}' with value '{X_row[node.feature]}' at depth {node.depth}. Falling back to decision.")
            return node.decision if node.decision is not None else None

        # Apply the function to each row in X
        predictions = X.apply(lambda row: predict_from_node(row, self.Root), axis=1)

        # Ensure predictions are a Pandas Series
        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions)

        if predictions.dtype in ['float64', 'float32']:
            predictions = predictions.round().astype(int)

        return predictions


    def plot(self) -> None:
        """
        Function to plot the tree structure.
        """
        def plot_(node: Node, depth: int) -> None:
            if node.decision is not None:
                print("   " * depth, "Decision:", node.decision)
                return
            if node.threshold is not None:
                print("   " * depth, f"[{node.feature} <= {node.threshold}]")
                plot_(node.left_child, depth + 1)
                print("   " * depth, f"[{node.feature} > {node.threshold}]")
                plot_(node.right_child, depth + 1)
            else:
                print("   " * depth, f"[{node.feature}]")
                for next_node in node.nexts:
                    print("   " * depth, f"- {next_node.name}:")
                    plot_(next_node.ptr, depth + 1)

        plot_(self.Root, 0)