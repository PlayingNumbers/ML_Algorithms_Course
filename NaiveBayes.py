import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = []
        self.class_prob = {}
        self.data_mean = {}
        self.data_var = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_prob = {c: np.mean(y == c) for c in self.classes}
        self.data_mean = {c: np.mean(X[y == c], axis=0) for c in self.classes}
        self.data_var = {c: np.var(X[y == c], axis=0) for c in self.classes}

    def predict(self, X):
        scores = []
        for c in self.classes:
            mean = self.data_mean[c]
            var = self.data_var[c]
            log_prob = np.sum(np.log(1 / np.sqrt(2 * np.pi * var)) - (X - mean)**2 / (2 * var))
            scores.append(log_prob + np.log(self.class_prob[c]))
        return self.classes[np.argmax(scores)]