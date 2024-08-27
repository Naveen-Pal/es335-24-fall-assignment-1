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

import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self, criterion='information_gain'):
        self.criterion = criterion
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth=0):
        # Check if all labels are the same
        if y.nunique() == 1:
            return y.iloc[0]

        # If there are no features left to split, return the most common label
        if X.empty:
            return y.mode()[0]

        # Find the best feature to split on
        if X.shape[1] == 0:
            return y.mode()[0]

        best_feature, split_val = self.opt_split_real(X, y)
        tree = {best_feature: {}}

        for value in X[best_feature].unique():
            subset_X, subset_y = self._split_data(X, y, best_feature, value)
            tree[best_feature][value] = self._build_tree(subset_X, subset_y, depth + 1)

        return tree

    def opt_split_real(self, X: pd.DataFrame, y: pd.Series):
        # Assuming we're working with numerical features
        best_feature = None
        best_split_value = None
        # Simplified: select the first feature as the best split for demonstration
        best_feature = X.columns[0]
        return best_feature, None

    def _split_data(self, X: pd.DataFrame, y: pd.Series, feature: str, value):
        subset_X = X[X[feature] == value]
        subset_y = y[X[feature] == value]
        return subset_X, subset_y

    def predict(self, X: pd.DataFrame):
        # Example of prediction (very basic)
        predictions = [self._predict_single(row) for _, row in X.iterrows()]
        return np.array(predictions)

    def _predict_single(self, row):
        # Example of prediction (very basic)
        current_node = self.tree
        while isinstance(current_node, dict):
            feature = list(current_node.keys())[0]
            value = row[feature]
            current_node = current_node[feature].get(value, None)
            if current_node is None:
                return None
        return current_node

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
