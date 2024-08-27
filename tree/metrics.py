from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy.
    Accuracy is defined as the number of correct predictions divided by the total number of predictions.
    """
    # Ensure that the predicted and actual values have the same size
    assert y_hat.size == y.size, "Predicted and actual values must have the same size."
    
    # Calculate accuracy
    correct_predictions = (y_hat == y).sum()
    accuracy_score = correct_predictions / y.size
    
    return accuracy_score


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision.
    Precision is defined as the number of true positives divided by the number of true positives plus false positives.
    """
    # Ensure that the predicted and actual values have the same size
    assert y_hat.size == y.size, "Predicted and actual values must have the same size."
    
    # Calculate true positives and false positives
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_positives = ((y_hat == cls) & (y != cls)).sum()
    
    # Handle case where true positives + false positives = 0 to avoid division by zero
    if true_positives + false_positives == 0:
        return 0.0
    
    precision_score = true_positives / (true_positives + false_positives)
    
    return precision_score


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall.
    Recall is defined as the number of true positives divided by the number of true positives plus false negatives.
    """
    # Ensure that the predicted and actual values have the same size
    assert y_hat.size == y.size, "Predicted and actual values must have the same size."
    
    # Calculate true positives and false negatives
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_negatives = ((y_hat != cls) & (y == cls)).sum()
    
    # Handle case where true positives + false negatives = 0 to avoid division by zero
    if true_positives + false_negatives == 0:
        return 0.0
    
    recall_score = true_positives / (true_positives + false_negatives)
    
    return recall_score


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error (RMSE).
    RMSE is a measure of the differences between predicted values and actual values.
    """
    # Ensure that the predicted and actual values have the same size
    assert y_hat.size == y.size, "Predicted and actual values must have the same size."
    
    # Calculate RMSE
    mse = np.mean((y_hat - y) ** 2)
    rmse_score = np.sqrt(mse)
    
    return rmse_score


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error (MAE).
    MAE is a measure of the average magnitude of the errors in a set of predictions, without considering their direction.
    """
    # Ensure that the predicted and actual values have the same size
    assert y_hat.size == y.size, "Predicted and actual values must have the same size."
    
    # Calculate MAE
    mae_score = np.mean(np.abs(y_hat - y))
    
    return mae_score