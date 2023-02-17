import numpy as np
import pandas as pd
from itertools import product

# def euclidean_distance(x1, x2):
# 	diff = np.array(x1) - np.array(x2)
# 	distance = np.sqrt(np.sum(diff**2))
# 	return distance

# def get_distance(pair_1, pair_2):
# 	min_distance = None
# 	for arr1 in pair_1:
# 		for arr2 in pair_2:
# 			distance = euclidean_distance(arr1, arr2)
# 			if min_distance is None:
# 				min_distance = distance
# 			if distance < min_distance:
# 				min_distance = distance 
# 	return min_distance

# class HierarchicalClustering:
# 	def __init__(self, n_clusters):
# 		self.n_clusters = n_clusters

# 	def _build_initial_clusters(self, X):
# 		# Place every data point in a cluster
# 		clusters = []
# 		for i in range(len(X)):
# 			clusters.append(X.iloc[i].values)
# 		return clusters

# 	def fit(self, X):

# 		self.clusters = self._build_initial_clusters(X)

# 		clusters_created = len(self.clusters)
# 		while clusters_created > self.n_clusters:

# 			closest_distance = None
# 			closest_pair = None
			
# 			# This basically gets us the closest function
# 			pairs_list = list(product(self.clusters, self.clusters))
# 			for pair in pairs_list:
				
# 				# Remove any that are paired with itself
# 				if str(list(pair[0])) == str(list(pair[1])):
# 					continue
				
# 				# Compute distance between pairs
# 				distance = get_distance(pair[0], pair[1])

# 				# Update the closest pair with the closest distance
# 				if closest_distance is None:
# 					closest_distance = distance
# 					closest_pair = pair

# 				if distance < closest_distance:
# 					closest_distance = distance
# 					closest_pair = pair

# 			# Remove closest pair from the overall clusters
# 			# Add it back as its own cluster
# 			# Need t figure this piece out
# 			def readjust_clusters(closest_pair, clusters):
# 				for pair in closest_pair:



# 			new_clusters = []
# 			for pair in self.clusters:
# 				if str(pair) not in str(closest_pair):
# 					new_clusters.append(pair)
# 				else:
# 					continue

# 			new_clusters.append(closest_pair)

# 			print(self.clusters)
# 			print(len(self.clusters))
# 			print(new_clusters)
# 			print(len(new_clusters))
# 			assert self.clusters != new_clusters

# 			self.clusters = new_clusters
			
# 			clusters_created = len(self.clusters)



# 	def predict(self, X):
		
# 		# Store each cluster in a dictionary
# 		cluster_store = dict()
# 		for i, cluster in enumerate(self.clusters):
# 			cluster_store[i] = cluster

# 		# Loop through each data point
# 		assigned_cluster = dict()

# 		for row in X.iterrows():
# 			closest_distance = None
# 			assigned_cluster = None
# 			for key, values in cluster_store.iteritems():				
# 				distance = euclidean_distance(row, values[0]) # change this index
# 				if closest_distance is None or closest_distance > distance:
# 					closest_distance = distance
# 					assigned_cluster = key
# 			assigned_cluster[row] = assigned_cluster
# 		return assigned_cluster


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
        pairs_list = list(product(np.arange(0,len(self.clusters)), np.arange(0,len(self.clusters))))
                distances[i, j] = self._calculate_distance(X[i], X[j])
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

        pairs_list = list(product(np.arange(0,len(self.clusters)), np.arange(0,len(self.clusters))))
        for i, j in pairs_list:
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
		