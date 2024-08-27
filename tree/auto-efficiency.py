import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree  # Assuming this is your custom decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                   names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                          "acceleration", "model year", "origin", "car name"])

# Clean the data
data.replace('?', np.nan, inplace=True)  # Replace '?' with NaN
data.dropna(inplace=True)  # Drop rows with NaN values

# Convert data types
data['horsepower'] = data['horsepower'].astype(float)  # Convert horsepower to float
data['origin'] = data['origin'].astype(int)  # Convert origin to int

# Drop the 'car name' column as it is not useful for prediction
data.drop(columns=['car name'], inplace=True)

# Prepare features and target
X = data.drop('mpg', axis=1)  # Features
y = data['mpg']  # Target

# Convert 'mpg' to a categorical target (e.g., 'low', 'high') for classification
# Let's use the median mpg value as the threshold for categorization
median_mpg = y.median()
y = y.apply(lambda mpg: 'high' if mpg >= median_mpg else 'low')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate your custom Decision Tree
tree_custom = DecisionTree(criterion='information_gain', max_depth=5)
tree_custom.fit(X_train, y_train)
y_pred_custom = tree_custom.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)

# Train and evaluate Scikit-Learn's Decision Tree
tree_sklearn = DecisionTreeClassifier(criterion='entropy', max_depth=5)
tree_sklearn.fit(X_train, y_train)
y_pred_sklearn = tree_sklearn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

# Print results
print(f"Custom Decision Tree Accuracy: {accuracy_custom:.2f}")
print(f"Scikit-Learn Decision Tree Accuracy: {accuracy_sklearn:.2f}")

# Optionally, plot comparison if needed
# plt.bar(['Custom Decision Tree', 'Scikit-Learn Decision Tree'], [accuracy_custom, accuracy_sklearn])
# plt.ylabel('Accuracy')
# plt.title('Model Performance Comparison')
# plt.show()
