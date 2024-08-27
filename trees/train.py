from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from dt import DecisionTree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=2, class_sep=0.5
)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Scatter plot of generated data")
plt.show()

#  Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree
tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)        

# Predict and Evaluate
y_pred = tree.predict(X_test)    

accuracy = accuracy_score(y_test, y_pred) 
precision = precision_score(y_test, y_pred, average=None) 
recall = recall_score(y_test, y_pred, average=None) 
print(f"Accuracy: {accuracy:.4f}")
print(f"Per-class Precision: {precision}")
print(f"Per-class Recall: {recall}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)