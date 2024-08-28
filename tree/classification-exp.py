import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from base import *

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTree(criterion='information_gain', max_depth=4)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
accuracy_ = accuracy(y_test, y_pred)
print("Accuracy of the model:", accuracy_, "\n")
print("Precision for Class 0:", precision(y_pred, y_test, 0))
print("Recall for Class 0:", recall(y_pred, y_test, 0),"\n")
print("Precision for Class 1:", precision(y_pred, y_test, 1))
print("Recall for Class 1:", recall(y_pred, y_test, 1),"\n")


from sklearn.model_selection import KFold

# Define cross-validation parameters
max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
k = 5  # Number of folds
accuracies = []

# Function to compute accuracy score
def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Initialize KFold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Cross-validation loop
for depth in max_depth:
    fold_accuracies = []

    for train_index, val_index in kf.split(X):
        # Split data into training and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Convert to DataFrame and Series
        X_train = pd.DataFrame(X_train)
        y_train = pd.Series(y_train)
        X_val = pd.DataFrame(X_val)
        y_val = pd.Series(y_val)

        # Train and predict with DecisionTree
        tree = DecisionTree(criterion='gini_index', max_depth=depth)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_val)
        
        # Compute accuracy
        accuracy_ = compute_accuracy(y_val, y_pred)
        fold_accuracies.append(accuracy_)

    # Calculate average accuracy for this depth
    mean_accuracy = np.mean(fold_accuracies)
    accuracies.append((depth, mean_accuracy))

# Find the best depth
best_depth = None
best_accuracy = -np.inf

for depth, accuracy in accuracies:
    if accuracy > best_accuracy:
        best_depth = depth
        best_accuracy = accuracy

print(f"Best max_depth: {best_depth}")
print(f"Best cross-validation accuracy: {best_accuracy:.2f}")

# Plot the results
depths, scores = zip(*accuracies)
plt.plot(depths, scores, marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Decision Tree Depth vs Accuracy')
plt.show()