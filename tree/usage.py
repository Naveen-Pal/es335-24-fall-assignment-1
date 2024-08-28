"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *

np.random.seed(42)

# Test case 1: Real Input and Real Output
print("### Test case 1: Real Input and Real Output ###\n")
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    print(f"Running Decision Tree with {criteria} criterion...\n")
    tree = DecisionTree()
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print(f"Criteria: {criteria}")
    print("RMSE:", rmse(y_hat, y))
    print("MAE:", mae(y_hat, y))
    print("\n" + "#" * 50 + "\n")

# # Test case 2: Real Input and Discrete Output
print("### Test case 2: Real Input and Discrete Output ###\n")
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    print(f"Running Decision Tree with {criteria} criterion...\n")
    tree = DecisionTree(criterion=criteria)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print(f"Criteria: {criteria}")
    print("Accuracy:", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Class {cls} - Precision:", precision(y_hat, y, cls))
        print(f"Class {cls} - Recall:", recall(y_hat, y, cls))
    print("\n" + "#" * 50 + "\n")

# Test case 3: Discrete Input and Discrete Output
print("### Test case 3: Discrete Input and Discrete Output ###\n")
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    print(f"Running Decision Tree with {criteria} criterion...\n")
    
    # Initialize the tree with the chosen criterion
    tree = DecisionTree(criterion=criteria)
    
    # Fit the tree (no need to pass criterion here again)
    tree.fit(X, y)
    
    # Predict and evaluate
    y_hat = tree.predict(X)
    
    # Plot the tree
    tree.plot()
    
    # Print evaluation metrics
    print(f"Criteria: {criteria}")
    print("Accuracy:", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Class {cls} - Precision:", precision(y_hat, y, cls))
        print(f"Class {cls} - Recall:", recall(y_hat, y, cls))
    print("\n" + "#" * 50 + "\n")


# # # Test case 4: Discrete Input and Real Output
print("### Test case 4: Discrete Input and Real Output ###\n")
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    print(f"Running Decision Tree with {criteria} criterion...\n")
    tree = DecisionTree(criterion=criteria)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print(f"Criteria: {criteria}")
    print("RMSE:", rmse(y_hat, y))
    print("MAE:", mae(y_hat, y))
    print("\n" + "#" * 50 + "\n")