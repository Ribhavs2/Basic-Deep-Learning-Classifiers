"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None 
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        n_samples, n_features = X_train.shape
        np.random.seed(0)
        self.w = np.random.rand(self.n_class, n_features)
        max_acc = 0 
        for epoch in range(self.epochs):
            for i in range(n_samples):
                x, y = X_train[i] , y_train[i]
                scores = np.dot(self.w, x)
                for c in range(self.n_class):
                    if c == y:
                        continue
                    if scores[c] > scores[y]:
                        self.w[y] += self.lr * x
                        self.w[c] -= self.lr * x
            self.lr = (1/(1+20*epoch)) * self.lr
            
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
        # X_test = self.standardize(X_test)
        scores = np.dot(X_test, self.w.T)
        predictions = np.argmax(scores, axis=1)
        return predictions
