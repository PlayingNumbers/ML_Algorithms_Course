import numpy as np

# Define the SVM class
class SVM:
    # Initialize the SVM with the given hyperparameters
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto'):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
    
    # Define the function to fit the training data
    def fit(self, X, y):
        # Initialize the number of support vectors
        self.num_support_vectors = 0
        
        # Calculate the number of training examples
        self.num_examples = X.shape[0]
        
        # Initialize the weights and bias
        self.w = np.zeros(X.shape[1])
        self.b = 0

        # Initialize the Lagrange multipliers
        self.lagrange_multipliers = np.zeros(self.num_examples)
        
        # Set the convergence criteria
        criteria = (self.C * np.eye(self.num_examples)).tolist()
        
        # Optimize the Lagrange multipliers
        while True:
            num_changed_lagrange_multipliers = 0
            for i in range(self.num_examples):
                # Calculate the error
                error = 0
                for j in range(self.num_examples):
                    # Calculate the kernel
                    if self.kernel == 'linear':
                        kernel = np.dot(X[i], X[j])
                    elif self.kernel == 'poly':
                        kernel = (np.dot(X[i], X[j]) + 1) ** self.degree
                    else:
                        kernel = np.exp(-self.gamma *
                            np.sum(np.square(X[i] - X[j])))
                        
                    # Calculate the error
                    error += self.lagrange_multipliers[j] * y[j] * kernel
                
                error -= y[i]
                
                # Check if the Lagrange multiplier is valid
                if ((y[i] * error < -criteria[i][i]) and
                    (self.lagrange_multipliers[i] < self.C)) or \
                    ((y[i] * error > criteria[i][i]) and
                    (self.lagrange_multipliers[i] > 0)):
                    
                    # Select the second Lagrange multiplier randomly
                    j = np.random.randint(0, self.num_examples)
                    while j == i:
                        j = np.random.randint(0, self.num_examples)
                    
                    # Calculate the error
                    error_j = 0
                    for k in range(self.num_examples):
                        # Calculate the kernel
                        if self.kernel == 'linear':
                            kernel = np.dot(X[j], X[k])
                        elif self.kernel == 'poly':
                            kernel = (np.dot(X[j], X[k]) + 1) ** self.degree
                        else:
                            kernel = np.exp(-self.gamma *
                                np.sum(np.square(X[j] - X[k])))
                        
                        # Calculate the error
                        error_j += self.lagrange_multipliers[k] * \
                            y[k] * kernel
                    
                    error_j -= y[j]
                    
                    # Save the Lagrange multipliers
                    lagrange_multipliers_i_old = self.lagrange_multipliers[i]
                    lagrange_multipliers_j_old = self.lagrange_multipliers[j]
                    
                    # Compute the bounds for the Lagrange multipliers
                    if y[i] != y[j]:
                        lower_bound = max(0, self.lagrange_multipliers[j] -
                            self.lagrange_multipliers[i])
                        upper_bound = min(self.C,
                            self.C + self.lagrange_multipliers[j] -
                            self.lagrange_multipliers[i])
                    else:
                        lower_bound = max(0, self.lagrange_multipliers[j] +
                            self.lagrange_multipliers[i] - self.C)
                        upper_bound = min(self.C,
                            self.lagrange_multipliers[j] +
                            self.lagrange_multipliers[i])
                    
                    # Compute the Lagrange multiplier
                    if lower_bound == upper_bound:
                        continue
                    
                    # Calculate the kernel
                    if self.kernel == 'linear':
                        kernel = np.dot(X[i], X[j])
                    elif self.kernel == 'poly':
                        kernel = (np.dot(X[i], X[j]) + 1) ** self.degree
                    else:
                        kernel = np.exp(-self.gamma *
                            np.sum(np.square(X[i] - X[j])))
                    
                    # Compute the Lagrange multiplier
                    lagrange_multipliers_j_new = self.lagrange_multipliers[j] + \
                        y[j] * (error - error_j) / (kernel +
                        self.lagrange_multipliers[i] -
                        self.lagrange_multipliers[j])
                    
                    # Clip the Lagrange multiplier
                    lagrange_multipliers_j_new = min(max(
                        lagrange_multipliers_j_new, lower_bound), upper_bound)
                    
                    # Check if the Lagrange multiplier is valid
                    if abs(lagrange_multipliers_j_new -
                        lagrange_multipliers_j_old) < 1e-5:
                        continue
                    
                    # Compute the Lagrange multiplier
                    lagrange_multipliers_i_new = self.lagrange_multipliers[i] + \
                        y[i] * y[j] * (lagrange_multipliers_j_old -
                        lagrange_multipliers_j_new)
                    
                    # Update the Lagrange multipliers
                    self.lagrange_multipliers[i] = lagrange_multipliers_i_new
                    self.lagrange_multipliers[j] = lagrange_multipliers_j_new
                    
                    # Update the error
                    error += y[i] * y[j] * (lagrange_multipliers_j_old -
                        lagrange_multipliers_j_new) * kernel
                    
                    # Update the weights and bias
                    self.w += (lagrange_multipliers_i_new -
                        lagrange_multipliers_i_old) * y[i] * X[i] + \
                        (lagrange_multipliers_j_new -
                        lagrange_multipliers_j_old) * y[j] * X[j]
                    self.b += (lagrange_multipliers_j_new -
                        lagrange_multipliers_j_old) * y[j]
                    
                    # Increment the number of changed Lagrange multipliers
                    num_changed_lagrange_multipliers += 1
            
            # Break if no Lagrange multiplier has changed
            if num_changed_lagrange_multipliers == 0:
                break
        
        # Compute the support vectors
        self.support_vectors = []
        for i in range(self.num_examples):
            if self.lagrange_multipliers[i] > 0:
                self.support_vectors.append(X[i])
        self.support_vectors = np.array(self.support_vectors)
        
        # Compute the number of support vectors
        self.num_support_vectors = len(self.support_vectors)
        
    # Define the function to predict the labels
    def predict(self, X):
        # Initialize the predictions
        predictions = np.zeros(X.shape[0])
        
        # Compute the predictions
        for i in range(X.shape[0]):
            prediction = 0
            for j in range(self.num_support_vectors):
                # Calculate the kernel
                if self.kernel == 'linear':
                    kernel = np.dot(X[i], self.support_vectors[j])
                elif self.kernel == 'poly':
                    kernel = (np.dot(X[i], self.support_vectors[j]) + 1) ** \
                        self.degree
                else:
                    kernel = np.exp(-self.gamma *
                        np.sum(np.square(X[i] - self.support_vectors[j])))
                
                # Calculate the prediction
                prediction += self.lagrange_multipliers[j] * \
                    y[j] * kernel
            
            prediction += self.b
            predictions[i] = np.sign(prediction)
        
        return predictions