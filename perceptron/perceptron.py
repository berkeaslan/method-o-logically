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