import pandas as pd
import numpy as np


# Using Gradient Descent
class LinearRegression:
    def __init__(self, step_size=0.2, max_steps=100):
        self.step_size = step_size
        self.max_steps = max_steps

    def sum_of_squared_error(self, X, y, preds):
        return np.sum((preds - y)**2)
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        
        # Initialize our weights to either zero, mean or random.
        self.weights = np.zeros(X.shape[1])

        # Make prediction using current weights
        preds = self.predict(X)

        # Compute the loss with the initialized weights. 
        # You should expect the loss to be high. 
        current_loss = self.sum_of_squared_error(X,y,preds)
        
        # Running Gradient Descent
        for _ in range(self.max_steps):

            # The partial derivative of loss with respect to weights
            # is the following equation
            dw = (1/num_samples) * np.dot(X.T, (preds - y))

            # Update the weights with the step size * gradient
            self.weights -= self.step_size * dw

            preds = self.predict(X)

            # Compute new loss with new weights
            new_loss = self.sum_of_squared_error(X,y,preds)

            # We want the loss to **increase** with each iteration. 
            # This is a Maximum Likelihood Estimation, we want to 
            # maximize our likelihood function.
            if current_loss < new_loss:
                break

            # Replace the loss
            current_loss = new_loss

    
    def predict(self, X):
        preds = np.dot(X, self.weights)
        return preds


# Using Linear Algebra
class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        X = np.concatenate((np.ones((num_samples, 1)), X), axis=1)
        A = np.dot(X.T, X)
        b = np.dot(X.T, y)
        self.weights = np.linalg.solve(A, b)
        self.bias = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        num_samples, num_features = X.shape
        X = np.concatenate((np.ones((num_samples, 1)), X), axis=1)
        return np.dot(X, self.weights) + self.bias






X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 3, 5])

print(X)

# Create an instance of the LinearRegression class
model = LinearRegressionLinearAlgebra()

# Fit the model to the training data
model.fit(X, y)

# Make predictions on new data
X_new = np.array([[7, 8]])
y_pred = model.predict(X_new)
print(y_pred)  
