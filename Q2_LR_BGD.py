# -*- coding: utf-8 -*-
"""
C11 A2 Programming Logistic Regression with Batch Gradient Descent
"""

import numpy as np
import pandas as pd


class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, max_iter=10000):
        """
        Initialize the model with learning rate and  max number of iterations.
        """
        pass

    def sigmoid(self, z):
        """
        Compute the sigmoid function.
        """
        pass

    def fit(self, X, y):
        """
        Train the model using batch gradient descent. 
        Dataset: N data points, d features
        
        Parameters:
        X: numpy array, shape (N, d)
           Features
        y: numpy array, shape (N,)
           Binary target values (0 or 1).
        """
        pass

    def predict_prob(self, X):
        """
        Predict probability estimates for input data X.
        """
        pass

    def predict(self, X, threshold=0.5):
        """
        Predict binary labels (0 or 1) for input data X using a threshold.
        """
        pass


# Example usage:
if __name__ == "__main__":
    
    # load dataframe using pandas from the .csv file
    df = pd.read_csv('loan_application.csv')

    X = df[['annual_income', 'credit_score']]
    y = y = df['loan_approved']
    
    # turn into numpy arrays
    X = X.values
    y = y.values
    
    # Feature scaling
    pass

    # Create the model and train it
    model = LogisticRegressionClassifier(learning_rate=0.1, max_iter=10000)
    model.fit(X, y)
    
    # Predict using the training set and get the training accuracy 
    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print("Training accuracy:", accuracy)
