import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NaiveBayes:
    def fit(self, X, y):
        # number of samples and features
        n_samples, n_features = X.shape
        
        # number of unique classes
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # calculate prior probabilities for each class
        self.priors = np.zeros(n_classes)
        for i in range(n_classes):
            self.priors[i] = np.sum(y == self.classes[i]) / float(n_samples)
        
        # calculate mean and variance for each feature and class
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            X_class = X[y == self.classes[i]]
            self.means[i, :] = X_class.mean(axis=0)
            self.variances[i, :] = X_class.var(axis=0)
            
    def predict(self, X):
        # initialize an empty list to store predicted class labels
        y_pred = []
        
        # iterate over each sample in X
        for sample in X:
            # calculate the posterior probability for each class
            posteriors = []
            for i in range(len(self.classes)):
                prior = np.log(self.priors[i])
                posterior = np.sum(np.log(self.calculate_likelihood(sample, self.means[i, :], self.variances[i, :])))
                posterior = prior + posterior
                posteriors.append(posterior)
            # select the class with the highest posterior probability as the predicted label
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return np.array(y_pred)
    
    def calculate_likelihood(self, x, mean, var):
        # calculate the probability of each feature given a class using Gaussian distribution
        exponent = np.exp(-((x - mean) ** 2 / (2 * var)))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

# generate a synthetic dataset with 4 features and 3 classes
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# instantiate the Naive Bayes classifier
nb = NaiveBayes()

# fit the model on the training set
nb.fit(X_train, y_train)

# make predictions on the testing set
y_pred = nb.predict(X_test)

# calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)

# print the accuracy score
print("Accuracy:", accuracy)