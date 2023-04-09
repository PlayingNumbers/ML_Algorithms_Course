import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self,
                 n_estimators=100,
                 max_depth = None,
                 min_samples_split=2,
                 max_features = 'auto',
                 max_samples = None):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.feature_importances = None
        self.max_features = max_features
        self.max_samples = max_samples

    def fit(self, X, y):

        ## Set an automatic max features
        if self.max_features == 'auto':
            self.max_features = int(np.sqrt(len(X.columns)))

        # Use the loop to create the number of estimators
        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth = self.max_depth,
                                          min_samples_split = self.min_samples_split,
                                          max_features = self.max_features)

            # Default to 70%
            if self.max_samples is None:
                self.max_samples = int(X.shape[0] * 0.7)

            # Define sub-sample
            indices = np.random.choice(X.shape[0], self.max_samples, replace = True)
            tree.fit(X.iloc[indices], y.iloc[indices])

            self.trees.append(tree)

    def feature_importances_(self):
        self.feature_importances = np.zeros(X.shape[1])
        for tree in self.trees:
            self.feature_importance += tree.feature_importances_

        self.feature_importances /= self.n_estimators

    def predict(self, X):
        all_preds = []
        for tree in self.trees:
            preds = tree.predict(X)
            all_preds.append(preds)
        return pd.DataFrame(all_preds).mean().values


if __name__=='__main__':

    ## Create dummy dataframe
    df = pd.DataFrame()
    X1 = np.array([1,2,3,4,5,6,7,8,9,10])
    X2 = np.array([4,5,6,4,5,1,7,8,9,10])
    y = np.array([0,1,1,0,0,1,0,1,0,1])
    df['col_1'] = X1
    df['col_2'] = X2
    df['y'] = y

    X = df[['col_1','col_2']]
    y = df['y']

    dt = RandomForestClassifier()
    dt.fit(X, y)
    preds = dt.predict(X)

    print(preds)






