import numpy as np
from typing import Optional

class Perceptron:
    def __init__(self, alpha: float = 0.1, epochs: int = 100, gamma: float = 0.01, weights: Optional[np.ndarray] = None) -> None:
        """
        Initialize the Perceptron.

        Parameters:
        - alpha (float): Learning rate.
        - epochs (int): Number of training iterations.
        - gamma (float): Threshold for early stopping based on iteration error.
        - weights (Optional[np.ndarray]): Initial weights. If None, initialized to zeros.
        """
        self.alpha = alpha
        self.epochs = epochs
        self.gamma = gamma
        self.weights = weights

    def __activation(self, x: float) -> int:
        """
        Activation function using the Heaviside step function.

        Parameters:
        - x (float): Weighted sum input.

        Returns:
        - int: 1 if x >= 0 else 0
        """
        return 1 if x >= 0 else 0  # Using 0 and 1 for binary classification

    def predict(self, X: np.ndarray) -> int:
        """
        Predict the class label for a single input sample.

        Parameters:
        - X (np.ndarray): Input feature vector.

        Returns:
        - int: Predicted class label (1 or 0).
        """
        X_augmented = np.insert(X, 0, 1)  # Add bias term
        weighted_sum = np.dot(self.weights, X_augmented)
        return self.__activation(weighted_sum)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Perceptron model on the provided dataset.

        Parameters:
        - X (np.ndarray): Training feature matrix.
        - y (np.ndarray): Training labels.
        """
        self.weights = np.zeros(X.shape[1] + 1)  # Initialize weights with bias

        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Add bias term
                weighted_sum = np.dot(self.weights, x_i)
                prediction = self.__activation(weighted_sum)
                error = y[i] - prediction
                total_error += abs(error)
                self.weights += self.alpha * error * x_i

            iteration_error = total_error / len(X)
            print(f"Epoch {epoch + 1}/{self.epochs}, Iteration Error: {iteration_error:.4f}")
            if iteration_error < self.gamma:
                print(f"Converged at epoch {epoch + 1} with iteration error {iteration_error:.4f}")
                break

import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Number of samples per class
n_samples = 50

# Generate class 1 data (e.g., centered at (2, 2))
mean1 = np.array([2, 2])
cov1 = np.array([[0.5, 0], [0, 0.5]])  # Covariance matrix
X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
y1 = np.ones(n_samples)  # Label: 1

# Generate class 0 data (e.g., centered at (-2, -2))
mean2 = np.array([-2, -2])
cov2 = np.array([[0.5, 0], [0, 0.5]])
X2 = np.random.multivariate_normal(mean2, cov2, n_samples)
y2 = np.zeros(n_samples)  # Label: 0

# Combine the data
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# Shuffle the dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Visualize the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='Class 1')
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', label='Class 0')
plt.title('Synthetic Linearly Separable Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Instantiate the Perceptron
perceptron = Perceptron(alpha=0.1, epochs=100, gamma=0.01)

# Train the Perceptron
perceptron.fit(X, y)

# Predictions on training data
predictions = np.array([perceptron.predict(x) for x in X])

# Calculate accuracy
accuracy = np.mean(predictions == y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# Function to plot decision boundary
def plot_decision_boundary(perceptron, X, y):
    # Define the range for the plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Flatten the grid to pass into the model
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([perceptron.predict(x) for x in grid])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.2, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='Class 1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', label='Class 0')
    plt.title('Perceptron Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the decision boundary
plot_decision_boundary(perceptron, X, y)
