import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from base import DecisionTree
from typing import Tuple

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values

def create_fake_data(n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic binary data for experiments.
    """
    X = np.random.randint(0, 2, size=(n_samples, n_features))  # Binary features
    y = np.random.randint(0, 2, size=n_samples)  # Binary target
    return X, y

def measure_time(n_samples: int, n_features: int, num_repeats: int = num_average_time) -> Tuple[float, float]:
    """
    Measure the average time taken for fitting and predicting.
    """
    fit_times = []
    predict_times = []
    
    for _ in range(num_repeats):
        X, y = create_fake_data(n_samples, n_features)
        tree = DecisionTree(criterion='information_gain', max_depth=4)  # Adjust parameters as needed
        
        # Measure fitting time
        start = time.time()
        tree.fit(pd.DataFrame(X), pd.Series(y))
        end = time.time()
        fit_times.append(end - start)
        
        # Measure predicting time
        start = time.time()
        tree.predict(pd.DataFrame(X))
        end = time.time()
        predict_times.append(end - start)
    
    avg_fit_time = np.mean(fit_times)
    avg_predict_time = np.mean(predict_times)
    
    return avg_fit_time, avg_predict_time

def plot_results(n_samples_list: list, n_features_list: list, fit_times: list, predict_times: list) -> None:
    """
    Plot the runtime results.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_features_list, fit_times, marker='o')
    plt.xlabel('Number of Features (M)')
    plt.ylabel('Average Fit Time (s)')
    plt.title('Fit Time vs. Number of Features')
    
    plt.subplot(1, 2, 2)
    plt.plot(n_features_list, predict_times, marker='o')
    plt.xlabel('Number of Features (M)')
    plt.ylabel('Average Predict Time (s)')
    plt.title('Predict Time vs. Number of Features')
    
    plt.tight_layout()
    plt.show()

# Define ranges for N and M
n_samples_list = [100, 500, 1000, 5000]  # Example sizes
n_features_list = [5, 10, 20, 50]  # Example feature counts

# Collect results
fit_times = []
predict_times = []

for n_samples in n_samples_list:
    for n_features in n_features_list:
        avg_fit_time, avg_predict_time = measure_time(n_samples, n_features)
        fit_times.append(avg_fit_time)
        predict_times.append(avg_predict_time)
        print(f"Samples: {n_samples}, Features: {n_features}, Fit Time: {avg_fit_time:.4f}s, Predict Time: {avg_predict_time:.4f}s")

# Plot results
plot_results(n_samples_list, n_features_list, fit_times, predict_times)


# Function to create fake data (take inspiration from usage.py)
# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
