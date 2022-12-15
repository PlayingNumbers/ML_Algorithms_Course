class LogisticRegression:
    def __init__(self, step_size=0.2, max_steps=100):
        self.step_size = step_size
        self.max_steps = max_steps
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def log_likelihood(self, X, y):
        # Z = weights * inputs (X)
        z = np.dot(X, self.weights)
        
        # Apply sigmoid transformation
        # Resulting formula = 1/ 1+e^(-w * x)
        h = self.sigmoid(z)

        # Compute log likelihood
        return 1 * (np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)))
    
    def fit(self, X, y):
        
        # Initialize our weights to either zero, mean or random.
        self.weights = np.zeros(X.shape[1])

        # Compute the loss with the initialized weights. 
        # You should expect the loss to be high. 
        current_loss = self.log_likelihood(X,y)
        
        # Running Gradient Descent
        for _ in range(self.max_steps):

            # This is X * w. No intercept in this example.
            z = np.dot(X, self.weights)
            
            # Using X * w, apply sigmoid transformation
            preds = self.sigmoid(z)
            
            # The partial derivative of loss with respect to weights
            # is the following equation
            gradient = np.dot(X.T, (preds - y)) / y.size

            # Update the weights with the step size * gradient
            self.weights -= self.step_size * gradient

            # Compute new loss with new weights
            new_loss = self.log_likelihood(X,y)

            # We want the loss to **increase** with each iteration. 
            # This is a Maximum Likelihood Estimation, we want to 
            # maximize our likelihood function.
            if current_loss < new_loss:
                break

            # Replace the loss
            current_loss = new_loss

    
    def predict(self, X):
        # Z = weights * inputs (X)
        z = np.dot(X, self.weights)

        # Apply sigmoid transformation
        preds = self.sigmoid(z)
        return preds