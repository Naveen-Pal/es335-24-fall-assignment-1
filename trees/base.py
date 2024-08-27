from dataclasses import dataclass
from typing import Literal, List, Optional, Union, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import opt_split_attribute, split_data, check_ifreal, opt_attr, find_optimal_attribute, df_to_array

class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, value=None, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value = value
        self.depth = depth

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None   

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        def _build_tree(X: pd.DataFrame, y: pd.Series, features: List[str], criterion: str, max_depth: int, depth: int) -> Node:
            # Base case: If the dataset is empty or all data belong to the same class, return the class label.
            if y.nunique() == 1:
                return Node(value=y.mode()[0], depth=depth)
            
            if depth >= max_depth:
                # For regression or classification at max depth, return the mean or mode
                if check_ifreal(y):
                    return Node(value=y.mean(), depth=depth)
                else:
                    return Node(value=y.mode()[0], depth=depth)

            # If no more features to split
            if len(features) == 0:
                if check_ifreal(y):
                    return Node(value=y.mean(), depth=depth)
                else:
                    return Node(value=y.mode()[0], depth=depth)

            # Determine the best attribute to split
            if not check_ifreal(X.iloc[:, 0]):
                opt_attr_ = opt_split_attribute(X, y, criterion, pd.Series(features))
                node = Node(feature=opt_attr_, depth=depth)
                for value, (Xi, yi) in split_data(X, y, opt_attr_, None):
                    new_features = [f for f in features if f != opt_attr_]
                    child_node = _build_tree(Xi, yi, new_features, criterion, max_depth, depth + 1)
                    if node.left_child is None:
                        node.left_child = child_node
                    else:
                        node.right_child = child_node
                return node

            else:
                if check_ifreal(y):
                    opt_attribute, split_val = opt_attr(X, y)
                else:
                    optimal_attribute = find_optimal_attribute(df_to_array(X), np.array(y))
                    opt_attribute = X.columns[optimal_attribute.feature_index]
                    split_val = optimal_attribute.split_value
                
                node = Node(feature=opt_attribute, threshold=split_val, depth=depth)
                left_data, right_data = split_data(X, y, opt_attribute, split_val)
                
                left_X, left_y = dfconvertor(left_data)
                right_X, right_y = dfconvertor(right_data)
                
                node.left_child = _build_tree(left_X, left_y, features, criterion, max_depth, depth + 1)
                node.right_child = _build_tree(right_X, right_y, features, criterion, max_depth, depth + 1)
                return node

        def dfconvertor(data):
            X = pd.DataFrame(data[0])
            X[data[1].name] = data[1]
            X = X.reset_index(drop=True)
            return X.iloc[:, :-1], X.iloc[:, -1]

        self.root = _build_tree(X, y, list(X.columns), self.criterion, self.max_depth, 0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """
        def _predict(node: Node, sample: pd.Series) -> Union[int, float]:
            if node.value is not None:
                return node.value
            
            if check_ifreal(sample[node.feature]):
                if sample[node.feature] <= node.threshold:
                    return _predict(node.left_child, sample)
                else:
                    return _predict(node.right_child, sample)
            else:
                if sample[node.feature] == node.threshold:
                    return _predict(node.left_child, sample)
                else:
                    return _predict(node.right_child, sample)

        predictions = X.apply(lambda row: _predict(self.root, row), axis=1)
        return predictions

    def plot(self) -> None:
        """
        Function to plot the tree
        """
        def _plot_tree(node: Node, depth: int, pos: str, parent_name: str, graph: Dict[str, str]) -> None:
            if node is None:
                return
            
            node_name = f"Node_{depth}_{id(node)}"
            if node.value is not None:
                graph[node_name] = f"Value: {node.value}"
            else:
                if check_ifreal(node.feature):
                    graph[node_name] = f"{node.feature} <= {node.threshold}"
                else:
                    graph[node_name] = f"{node.feature} == {node.threshold}"
            
            if parent_name:
                plt.plot([pos[0], pos[1]], [depth, depth], 'k-')
                plt.text((pos[0] + pos[1]) / 2, depth, parent_name, ha='center')

            if node.left_child:
                _plot_tree(node.left_child, depth + 1, (pos[0] - 0.1, pos[0]), node_name, graph)
            if node.right_child:
                _plot_tree(node.right_child, depth + 1, (pos[1], pos[1] + 0.1), node_name, graph)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        graph = {}
        _plot_tree(self.root, 0, (0.5, 0.5), '', graph)
        plt.show()