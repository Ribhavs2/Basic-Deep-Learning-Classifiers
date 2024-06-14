"""Logistic regression model."""

import numpy as np

class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def standardize(self, X, training=False):
        """Standardize the data."""
        if training:  # Calculate mean and std for training data
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        # Apply standardization using stored mean and std
        return (X - self.mean) / self.std

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """ 
        return (1 / (1+np.exp(-z)))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        X_train = self.standardize(X_train, training=True)
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)
        
        y_train = np.where(y_train == -1, 0, y_train)

        for i in range(self.epochs):
            z = np.dot(X_train, self.w)
            predictions = self.sigmoid(z)
            self.w += self.lr / n_samples * np.dot(X_train.T, (y_train - predictions))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        X_test = self.standardize(X_test)
        y_array = np.array([])
        for i in range(X_test.shape[0]):
            x_i = X_test[i]
            if self.sigmoid(np.dot(x_i, self.w)) < self.threshold:
                y_array = np.append(y_array, 0)
            else: 
                y_array = np.append(y_array, 1)
        return y_array
