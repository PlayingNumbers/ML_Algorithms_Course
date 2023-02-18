import numpy as np
import pandas as pd

class NMF:
    def __init__(self, n_components, max_iter=200, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize factors with random values
        self.W = np.random.rand(n_samples, self.n_components)
        self.H = np.random.rand(self.n_components, n_features)

        for n_iter in range(self.max_iter):
            # Update H
            # XW = np.dot(X, self.W)
            # HH = np.dot(self.H, self.H.T)
            self.H *= np.dot(self.W.T, X) / (np.dot(np.dot(self.W.T, self.W), self.H) + 1e-10)


            # Update W
            XH = np.dot(self.W, self.H)
            WHH = np.dot(XH, self.H.T) + 1e-10
            self.W *= np.dot(X, self.H.T) / WHH

            # Compute reconstruction error
            err = np.mean((X - np.dot(self.W, self.H)) ** 2)

            # Check for convergence
            if n_iter % 10 == 0:
                print("Iteration {}: error = {:.4f}".format(n_iter, err))
            if err < self.tol:
                print("Converged after {} iterations".format(n_iter))
                break

    def transform(self, X):
        return np.dot(self.W.T, X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

X = np.random.rand(100, 50)

nmf = NMF(n_components=10, max_iter=200, tol=1e-4)
W = nmf.fit_transform(X)