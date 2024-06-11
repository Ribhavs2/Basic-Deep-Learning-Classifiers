"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None 
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def standardize(self, X, training=False):
        if training: 
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        else:
            print(self.mean, self.std)
        return (X - self.mean) / self.std

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        n_samples, n_features = X_train.shape
        gradient = np.zeros_like(self.w)

        for i in range(n_samples):
            scores = np.dot(self.w, X_train[i])
            yi = y_train[i]
            for c in range(self.n_class):
                # if c != yi:
                if np.dot(self.w[yi],X_train[i]) - np.dot(self.w[c],X_train[i]) < 1:
                    gradient[yi] -= X_train[i]
                    gradient[yi] += self.reg_const * self.w[yi]/n_samples
                    #incorrect class
                    gradient[c] +=  X_train[i]
                    gradient[c] += self.reg_const * self.w[c]/n_samples
                    
                else:
                    gradient[yi] += self.reg_const * self.w[yi]/n_samples
                    #incorrect class
                    gradient[c] += self.reg_const * self.w[c]/n_samples

        return gradient/n_samples

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        X_train = self.standardize(X_train, training=True)
        np.random.seed(0)
        n_samples, n_features = X_train.shape
        self.w = np.random.rand(self.n_class, n_features)
        batch_size = 5000 # maybe 5000
        max_acc = 0
        for epoch in range(self.epochs):
            for batch in range(0, n_samples, batch_size):
                end = min(batch + batch_size, n_samples)
                X_batch = X_train[batch:end]
                y_batch = y_train[batch:end]

                gradient = self.calc_gradient(X_batch, y_batch)
                self.w -= self.lr * gradient
        

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
        scores = np.dot(X_test, self.w.T)
        y_pred = np.argmax(scores, axis=1)
        return y_pred
    