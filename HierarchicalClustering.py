import numpy as np
import pandas as pd
from itertools import product


class HierarchicalClustering:
    def __init__(self, n_clusters, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        self.n_samples = X.shape[0]
        self.clusters = [[i] for i in range(self.n_samples)]
        self.distances = self._calculate_distances(X)
        self.history = []

        while len(self.clusters) > self.n_clusters:
            i, j = self._find_closest_pair()
            self._merge_clusters(i, j)
            self.history.append((i, j))

    def _calculate_distances(self, X):
        distances = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(i+1, self.n_samples):
                distances[i, j] = self._calculate_distance(X.iloc[i], X.iloc[j])
        return distances

    def _calculate_distance(self, x, y):
        if self.linkage == 'single':
            return np.min(np.abs(x - y))
        elif self.linkage == 'complete':
            return np.max(np.abs(x - y))
        elif self.linkage == 'average':
            return np.mean(np.abs(x - y))

    def _find_closest_pair(self):
        min_distance = np.inf
        closest_pair = None
        for i in range(len(self.clusters)):
            for j in range(i+1, len(self.clusters)):
                distance = self._calculate_cluster_distance(self.clusters[i], self.clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j)
        return closest_pair

    def _calculate_cluster_distance(self, c1, c2):
        distance = np.inf
        for i in c1:
            for j in c2:
                d = self.distances[min(i, j), max(i, j)]
                if d < distance:
                    distance = d
        return distance

    def _merge_clusters(self, i, j):
        self.clusters[i] = self.clusters[i] + self.clusters[j]
        self.clusters.pop(j)

    def predict(self):
        labels = np.zeros(self.n_samples, dtype=np.int32)
        for i, cluster in enumerate(self.clusters):
            for j in cluster:
                labels[j] = i
        return labels
if __name__=='__main__':

    ## Create dummy dataframe
    df = pd.DataFrame()
    X1 = np.array([1,2,3,4,5,6,7,8,9,10])
    X2 = np.array([4,5,6,4,5,1,7,8,9,10])
    y = np.array([0,1,1,0,0,1,0,1,0,1])
    df['col_1'] = X1
    df['col_2'] = X2

    X = df[['col_1','col_2']]

    hc = HierarchicalClustering(n_clusters=4)
    hc.fit(X)
    print(hc.clusters)
		