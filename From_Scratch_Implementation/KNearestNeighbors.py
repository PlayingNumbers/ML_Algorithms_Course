import numpy as np
import pandas as pd

class KNN:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y

    def euclidean_distance(self, X1, X2):
        return np.linalg.norm(X1 - X2)

    def _select_neighbors(self, all_distances):
        nn_dict = dict()
        for key, distances in all_distances.items():
            sorted_d = sorted(distances.items(),
                              key=lambda item: item[1],
                              reverse=True)

            nearest_neighbors = sorted_d[:self.n_neighbors]
            nn_dict[key] = nearest_neighbors
        return nn_dict

    def _compute_distances(self, X):
        all_distances = {}
        ## Compute distances between X and fitted data
        for i_pred, X_pred in X.iterrows():
            individ_distances = {}
            for i_fit, X_fit in self.X.iterrows():
                distance = self.euclidean_distance(X_pred, X_fit)
                individ_distances[i_fit] = distance
            all_distances[i_pred] = individ_distances
        return all_distances

    def predict(self, X):
        all_distances = self._compute_distances(X)
        nn_dict = self._select_neighbors(all_distances)

        # Compute predictions
        predictions = []
        for key, neighbors in nn_dict.items():
            labels = [self.y[neighbor[0]] for neighbor in neighbors]
            predictions.append(np.mean(labels))
        return predictions

if __name__=='__main__':
    # Create a dummy dataset using pandas
    df = pd.DataFrame()
    X1 = np.array([1,2,3,4,5,6,7,8,9,10])
    X2 = np.array([4,5,6,4,5,1,7,8,9,10])
    y = np.array([8,7,5,8,2,3,5,7,7,8])
    df['col_1'] = X1
    df['col_2'] = X2
    df['y'] = y

    X = df[['col_1','col_2']]
    y = df['y']

    knn = KNN(n_neighbors = 3)
    knn.fit(X, y)
    predictions = knn.predict(X)

    print(predictions)













