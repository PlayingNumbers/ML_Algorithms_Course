import numpy as np

class LogisticRegression:
    def __init__(self, step_size=0.2, max_steps=100, reg_lambda = 0):
        self.step_size = step_size
        self.max_steps = max_steps
        self.reg_lambda = reg_lambda
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def log_likelihood(self, X, y, preds, ):
        # Compute the regularization term
        reg_term = self.reg_lambda/(2*len(y)) * np.sum(self.weights ** 2)

        # Compute the loss with the regularization term
        return (np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds)))/(len(y)) + reg_term
    
    def fit(self, X, y):
        
        # Initialize our weights to either zero, mean or random.
        self.weights = np.zeros(X.shape[1])

        # Compute the loss with the initialized weights. 
        # You should expect the loss to be high. 
        preds = self.predict(X)
        current_loss = self.log_likelihood(X, y, preds)
        
        # Running Gradient Descent
        for _ in range(self.max_steps):

            # Calculate the gradient for regularization
            reg_gradient = self.regularization_param * self.weights
            
            # The partial derivative of loss with respect to weights
            # The derivative of the regularization
            gradient = np.dot(X.T, (preds - y)) / y.size + self.reg_term/y.size * self.weights

            # Update the weights with the step size * gradient
            self.weights -= self.step_size * gradient

            # Make prediction using current weights
            preds = self.predict(X)

            # Compute new loss with new weights
            new_loss = self.log_likelihood(X,y,preds)

            # We want the loss to **increase** with each iteration. 
            # This is a Maximum Likelihood Estimation, we want to 
            # maximize our likelihood function.
            if current_loss > new_loss:
                break

            # Replace the loss
            current_loss = new_loss

    
    def predict(self, X):
        # Z = weights * inputs (X)
        z = np.dot(X, self.weights)

        # Apply sigmoid transformation
        preds = self.sigmoid(z)
        return preds