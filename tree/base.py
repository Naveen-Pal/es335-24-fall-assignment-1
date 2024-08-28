from dataclasses import dataclass
from typing import Literal, Union, Tuple
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

@dataclass
class DecisionTree:
    """
    A class representing the Decision Tree model.
    """
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int  

    def __init__(self, criterion='information_gain', max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.Root = None    

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree.
        """

        y = pd.Series(y)  # Convert y to Pandas Series
        X = pd.DataFrame(X)  # Convert X to DataFrame (if needed)

        def _build_tree(X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
            node = Node(depth=depth)

            # Base cases for stopping the recursion
            if y.nunique() == 1:
                node.decision = y.mode()[0]
                return node

            if depth >= self.max_depth or X.empty:
                node.decision = y.mean() if check_ifreal(y) else y.mode()[0]
                return node

            first_feature = X.columns[0]
            if not check_ifreal(X[first_feature]):
                # Discrete feature
                best_feature = opt_split_attribute(X, y, self.criterion)
                node.feature = best_feature
                unique_values = X[best_feature].unique()

                for val in unique_values:
                    subset_X, subset_y = split_data(X, y, best_feature, val)
                    child_node = _build_tree(subset_X, subset_y, depth + 1)
                    node.add_in_nexts(name=val, ptr=child_node)
            else:
                # Continuous feature
                if check_ifreal(y):
                    # Real Input, Real Output
                    best_feature, split_val = self.opt_split_real_mse(X, y)
                else:
                    # Real Input, Discrete Output
                    best_feature, split_val = self.opt_split_real(X, y)
                    
                node.feature = best_feature
                node.threshold = split_val

                left_X, left_y, right_X, right_y = split_data_continuous(X, y, best_feature, split_val)
                node.left_child = _build_tree(left_X, left_y, depth + 1)
                node.right_child = _build_tree(right_X, right_y, depth + 1)

            return node
        
        self.Root = _build_tree(X, y, depth=0)

    def opt_split_real_mse(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, float]:
        """
        Function to find the best split for continuous features for regression (Real Input, Real Output).
        """
        best_gain = -np.inf
        best_split = None
        best_feature = None

        for feature in X.columns:
            sorted_values = np.sort(X[feature].unique())
            split_candidates = (np.array(sorted_values[:-1]) + np.array(sorted_values[1:])) / 2
            for split_val in split_candidates:
                left_y = y[X[feature] <= split_val]
                right_y = y[X[feature] > split_val]

                parent_mse = mse(y)
                left_mse = mse(left_y)
                right_mse = mse(right_y)
                
                weighted_avg_mse = (len(left_y) / len(y)) * left_mse + (len(right_y) / len(y)) * right_mse
                gain = parent_mse - weighted_avg_mse

                if gain > best_gain:
                    best_gain = gain
                    best_split = split_val
                    best_feature = feature

        return best_feature, best_split

    def opt_split_real(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, float]:
        """
        Function to find the best split for continuous features (Real Input, Discrete Output).
        """
        best_gain = -np.inf
        best_split = None
        best_feature = None

        for feature in X.columns:
            sorted_values = np.sort((X[feature].unique()))
            split_candidates = (np.array(sorted_values[:-1]) + np.array(sorted_values[1:])) / 2

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

                gain = parent_criterion - (
                    (len(left_y) / len(y)) * left_criterion +
                    (len(right_y) / len(y)) * right_criterion
                )

                if gain > best_gain:
                    best_gain = gain
                    best_split = split_val
                    best_feature = feature

        return best_feature, best_split

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
                if node.threshold is not None:
                    if (X_row[node.feature]) <= (node.threshold):
                        return predict_from_node(X_row, node.left_child)
                    else:
                        return predict_from_node(X_row, node.right_child)
                else:
                    for next_node in node.nexts:
                        if next_node.name == X_row[node.feature]:
                            return predict_from_node(X_row, next_node.ptr)
            return np.nan  # Return NaN if no prediction can be made

        predictions = X.apply(lambda row: predict_from_node(row, self.Root), axis=1)
        
        if check_ifreal(predictions):  # If predictions are continuous
            predictions = (predictions >= 0.5).astype(int)  # For binary classification
        
        return pd.Series(predictions)

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