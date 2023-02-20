import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self,
                 num_estimators,
                 learning_rate,
                 max_depth = None):
        self.num_estimators = num_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):

        # Initialize model with the mean
        self.avg_y = np.mean(y)
        self.trees = []

        for i in range(self.num_estimators):

            # Compute residual
            residual = y - self.predict(X)

            # Fit Decision Tree on the residuals
            tree = DecisionTreeRegressor(max_depth = self.max_depth)
            tree.fit(X, residual)
            self.trees.append(tree)

            # Make prediction & recompute residual
            residual_preds = tree.predict(X)

    def predict(self, X):

        # Initial array with training means
        final_prediction = np.full((1,len(X)),self.avg_y)[0]

        # Get residual predictions
        for tree in self.trees:
            # Make Prediction
            resid_pred = tree.predict(X)
            final_prediction += resid_pred * self.learning_rate

        return final_prediction

if __name__=='__main__':

    ## Create dummy dataframe
    df = pd.DataFrame()
    X1 = np.array([1,2,3,4,5,6,7,8,9,10])
    X2 = np.array([4,5,6,4,5,1,7,8,9,10])
    y = np.array([8,7,5,8,2,3,5,7,7,8])
    df['col_1'] = X1
    df['col_2'] = X2
    df['y'] = y

    X = df[['col_1','col_2']]
    y = df['y']

    dt = GradientBoostingRegressor(num_estimators = 100,
                                   learning_rate = 0.02)
    dt.fit(X, y)
    preds = dt.predict(X)

    print(preds)