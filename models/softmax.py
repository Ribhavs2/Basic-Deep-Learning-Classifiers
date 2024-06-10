"""Softmax model."""

import numpy as np
import math


class Softmax:
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
        # self.accuracy = []
    
    def standardize(self, X, training=False):
        """Standardize the data."""
        if training:  # Calculate mean and std for training data
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        # Apply standardization using stored mean and std
        return (X - self.mean) / self.std

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c smean that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        n = X_train.shape[0]
        gradients = np.zeros((self.n_class, X_train.shape[1]))
        
        for i in range(0, n):
            y_i = y_train[i]
            x_i = X_train[i]
            fcs = np.dot(self.w, x_i)
            fcs = fcs/30 # Temp scaling
            k = max(fcs) # To avoid overflow
            # fcs = fcs - sum(fcs)/self.n_class # To avpod overflow
            exp_fcs = np.exp(fcs - k)

            # fcs = fcs/50
            # exp_fcs = np.exp(fcs)
            sum_exp = sum(exp_fcs)
            for c in range(0, self.n_class):
                if c == y_i:
                    gradients[c] += (exp_fcs[c]*x_i/sum_exp) - x_i
                else:
                    gradients[c] += exp_fcs[c]*x_i/sum_exp
            
        return gradients/n


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        X_train = self.standardize(X_train, training=True)
        n = X_train.shape[0]

        np.random.seed(0)
        self.w = np.random.rand(self.n_class, X_train.shape[1])
        
        batch_size = 5000
        for epoch in range(0,self.epochs):
            if epoch > 50:
                self.lr = self.lr*0.8

            if epoch == 60:
                self.lr = 1
                print(self.w)
            for batch in range(0, int(n/batch_size)):
                grads = self.calc_gradient(X_train[batch*batch_size:(batch+1)*batch_size], y_train[batch*batch_size:(batch+1)*batch_size])
                self.w -= self.lr*grads


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
        X_test = self.standardize(X_test, training=False)

        result = np.zeros(X_test.shape[0], dtype=np.uint32)
        for i in range(X_test.shape[0]):
            x_i = X_test[i]
            preds = np.dot(self.w, x_i)
            
            max_val = preds[0]
            max_cls = 0
            for c in range(1, preds.shape[0]):
                if preds[c] > max_val:
                    max_val = preds[c]
                    max_cls = c
            
            result[i] = max_cls
        return result
    
def get_acc(pred, y_test):
    return np.sum(y_test == pred) / len(y_test) * 100
