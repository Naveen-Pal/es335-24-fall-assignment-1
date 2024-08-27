import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, criterion="entropy"):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features) 
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        # Select random features
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        
        # Find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, threshold):
        # Parent entropy/gini/mse
        parent_loss = self._calculate_loss(y)

        # Create children
        left_idx, right_idx = self._split(X_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._calculate_loss(y[left_idx]), self._calculate_loss(y[right_idx])

        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        # Calculate information gain
        information_gain = parent_loss - child_loss 
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _calculate_loss(self, y):
        if self.criterion == "entropy":
            return self._entropy(y)
        elif self.criterion == "gini":
            return self._gini(y)
        elif self.criterion == "mse":
            return self._mse(y)
        else:
            raise ValueError("Invalid criterion")

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum([p**2 for p in ps])

    def _mse(self, y):
        return np.mean((y - np.mean(y))**2)

    def _calculate_leaf_value(self, y):
        if self.criterion in ['entropy', 'gini']:
            return Counter(y).most_common(1)[0][0]
        elif self.criterion == 'mse':
            return np.mean(y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left_child)
        return self._traverse_tree(x, node.right_child)

    def plot_tree(self, node=None, depth=0, ax=None, pos=0.5, parent_pos=None, parent_value=None):
        if node is None:
            node = self.root
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))  

        if node.is_leaf_node():
            ax.text(pos, -depth, f'Leaf\nValue: {node.value}', ha='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))
            if parent_pos is not None:
                ax.plot([parent_pos, pos], [-depth + 1, -depth], color='black')
        else:
            ax.text(pos, -depth, f'Feature: {node.feature}\nThreshold: {node.threshold:.2f}', ha='center', bbox=dict(facecolor='lightblue', edgecolor='black'))
            if parent_pos is not None:
                ax.plot([parent_pos, pos], [-depth + 1, -depth], color='black')

            shift = 0.5 ** (depth + 1)
            self.plot_tree(node.left_child, depth + 1, ax, pos - shift, pos, node.feature)
            self.plot_tree(node.right_child, depth + 1, ax, pos + shift, pos, node.feature)

        return ax