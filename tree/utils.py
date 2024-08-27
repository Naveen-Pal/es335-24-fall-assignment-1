"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Union

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one-hot encoding on the input data.
    Converts categorical variables into a form that could be provided to ML algorithms to do a better job in prediction.
    """
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values.
    Returns True if the values are real, otherwise False.
    """
    return pd.api.types.is_numeric_dtype(y)

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy.
    """
    p = Y.value_counts() / len(Y)
    entropy_value = -np.sum(p * np.log2(p + 1e-9))  # Adding a small epsilon to avoid log(0)
    return entropy_value

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the Gini index.
    """
    p = Y.value_counts() / len(Y)
    gini_value = 1 - np.sum(p ** 2)
    return gini_value

def information_gain(X_feature: pd.Series, y: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain for a feature.
    """
    if criterion == 'information_gain':
        base_entropy = entropy(y)
        values, counts = np.unique(X_feature, return_counts=True)
        weighted_entropy = np.sum((counts[i] / np.sum(counts)) * entropy(y[X_feature == v]) for i, v in enumerate(values))
        gain = base_entropy - weighted_entropy
    elif criterion == 'gini_index':
        base_gini = gini_index(y)
        values, counts = np.unique(X_feature, return_counts=True)
        weighted_gini = np.sum((counts[i] / np.sum(counts)) * gini_index(y[X_feature == v]) for i, v in enumerate(values))
        gain = base_gini - weighted_gini
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    return gain

def mse(y: pd.Series) -> float:
    """
    Function to calculate the Mean Squared Error (MSE).
    Used as a criterion for continuous outputs.
    """
    mean_y = np.mean(y)
    return np.mean((y - mean_y) ** 2)

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion='information_gain') -> str:
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

def opt_split_real(X: pd.DataFrame, y: pd.Series, criterion: str) -> Tuple[str, float]:
    """
    Function to find the best split for continuous features.
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

            if criterion == "information_gain":
                parent_criterion = entropy(y)
                left_criterion = entropy(left_y)
                right_criterion = entropy(right_y)
            elif criterion == "gini_index":
                parent_criterion = gini_index(y)
                left_criterion = gini_index(left_y)
                right_criterion = gini_index(right_y)
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            gain = parent_criterion - (
                (len(left_y) / len(y)) * left_criterion +
                (len(right_y) / len(y)) * right_criterion
            )

            if gain > best_gain:
                best_gain = gain
                best_split = split_val
                best_feature = feature

    return best_feature, best_split

def split_data(X: pd.DataFrame, y: pd.Series, feature: str, value: Union[str, float]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the data based on a discrete feature and its value.
    """
    subset_X = X[X[feature] == value]
    subset_y = y[X[feature] == value]
    return subset_X, subset_y

def split_data_continuous(X: pd.DataFrame, y: pd.Series, feature: str, threshold: float) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the data based on a continuous feature and a threshold.
    """
    left_X = X[X[feature] <= threshold]
    left_y = y[X[feature] <= threshold]
    right_X = X[X[feature] > threshold]
    right_y = y[X[feature] > threshold]
    return left_X, left_y, right_X, right_y