import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from base import DecisionTree

# Generate the dataset
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, 
                           random_state=1, n_clusters_per_class=2, class_sep=0.5)

# Plotting the data (Q1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Generated Data Distribution")

plt.show()

# Split the data: 70% for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the decision tree model with a maximum depth of 4
tree = DecisionTree(criterion='information_gain', max_depth=4)

# Fit the model on the training data
tree.fit(X_train, y_train)

# Predict using the decision tree on the test set
y_pred = tree.predict(X_test)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 1: Generate a synthetic dataset
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

# Step 2: Nested cross-validation function to find the optimal tree depth
def evaluate_tree_depth(X, y, max_depths):
    """
    Perform nested cross-validation to find the optimal depth of the decision tree.
    
    Parameters:
    - X: Features dataframe
    - y: Target series
    - max_depths: List of maximum depths to evaluate
    
    Returns:
    - best_depth: Optimal depth of the decision tree
    - best_score: Best accuracy score achieved
    """

    # Convert numpy arrays to pandas DataFrame/Series if necessary
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    # Outer cross-validation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

best_depth = None
best_score = -np.inf

for depth in range(1, 11):  # Check depths from 1 to 10
    scores = []
    for train_index, test_index in outer_cv.split(X):
        X_train, X_test = X[train_index], X[test_index]  # Use array slicing
        y_train, y_test = y[train_index], y[test_index]  # Use array slicing

        # Train the model with the current depth
        tree = DecisionTree(criterion='information_gain', max_depth=depth)
        tree.fit(X_train, y_train)

        # Make predictions
        y_pred = tree.predict(X_test)

        # Ensure predictions are integers (0 or 1)
        y_pred = np.round(y_pred).astype(int)
        y_test = y_test.astype(int)

        # Calculate accuracy
        try:
            accuracy = accuracy_score(y_test, y_pred)
            scores.append(accuracy)
        except ValueError as e:
            print(f"Error: {e}")
            print(f"y_test: {y_test[:10].tolist()}")
            print(f"y_pred: {y_pred[:10].tolist()}")
            continue

    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_depth = depth

print(f'Best Depth: {best_depth}')
print(f'Best Accuracy Score: {best_score}')