import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_numeric_dtype(y)

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    value_counts = Y.value_counts(normalize=True)
    return -np.sum(value_counts * np.log2(value_counts + 1e-9))  # Adding a small value to avoid log(0)

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    value_counts = Y.value_counts(normalize=True)
    return 1 - np.sum(value_counts ** 2)

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion == 'information_gain':
        entropy_before = entropy(Y)
        value_counts = attr.value_counts(normalize=True)
        weighted_entropy = np.sum([
            (value_counts[val] * entropy(Y[attr == val]))
            for val in value_counts.index
        ])
        return entropy_before - weighted_entropy

    elif criterion == 'gini_index':
        gini_before = gini_index(Y)
        value_counts = attr.value_counts(normalize=True)
        weighted_gini = np.sum([
            (value_counts[val] * gini_index(Y[attr == val]))
            for val in value_counts.index
        ])
        return gini_before - weighted_gini

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    """
    best_attr = None
    best_gain = -float('inf')
    
    for attr in features:
        if check_ifreal(X[attr]):
            # For real valued attributes
            unique_values = X[attr].unique()
            best_split = None
            best_gain_attr = -float('inf')

            for val in unique_values:
                gain = information_gain(y, X[attr] <= val, criterion)
                if gain > best_gain_attr:
                    best_gain_attr = gain
                    best_split = val

            if best_gain_attr > best_gain:
                best_gain = best_gain_attr
                best_attr = attr
                best_value = best_split

        else:
            # For discrete attributes
            gain = information_gain(y, X[attr], criterion)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                best_value = None

    return best_attr, best_value

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Function to split the data according to an attribute.
    """
    if value is None:
        # Discrete attribute
        mask = X[attribute] == attribute
    else:
        # Real attribute
        mask = X[attribute] <= value

    X_left = X[mask]
    y_left = y[mask]
    X_right = X[~mask]
    y_right = y[~mask]

    return (X_left, y_left), (X_right, y_right)