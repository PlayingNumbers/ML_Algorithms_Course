import numpy as np
import pandas as pd
from itertools import product

def euclidean_distance(x1, x2):
	diff = np.array(x1) - np.array(x2)
	distance = np.sqrt(np.sum(diff**2))
	return distance

class HierarchicalClustering:
	def __init__(self, n_clusters):
		self.n_clusters = n_clusters

	def _build_initial_clusters(self, X):
		# Place every data point in a cluster
		clusters = []
		for i in range(len(X)):
			clusters.append(X.iloc[i].values)
		return clusters

	def fit(self, X):

		self.clusters = self._build_initial_clusters(X)

		clusters_created = len(self.clusters)
		while clusters_created > self.n_clusters:

			closest_distance = None
			closest_pair = None
			
			# Make a combination of all clutsers
			pairs_list = list(product(self.clusters, self.clusters))
			for pair in pairs_list:
				
				# Remove any that are paired with itself
				if str(list(pair[0])) == str(list(pair[1])):
					continue
				
				# Compute distance between pairs
				# TO DO: Figure out how to catch the case of multiple pairs

				distance = euclidean_distance(sorted(pair[0]), sorted(pair[1]))

				# Update the closest pair with the closest distance
				if closest_distance is None:
					closest_distance = distance
					closest_pair = pair

				if distance < closest_distance:
					closest_distance = distance
					closest_pair = pair

			# Remove closest pair from the overall clusters
			# Add it back as its own cluster
			for pair in closest_pair:
				self.clusters.remove(pair)

			self.clusters.append(closest_pair)
			clusters_created = len(self.clusters)

	def predict(self, X):
		
		# Store each cluster in a dictionary
		cluster_store = dict()
		for i, cluster in enumerate(self.clusters):
			cluster_store[i] = cluster

		# Loop through each data point
		assigned_cluster = dict()

		for row in X.iterrows():
			closest_distance = None
			assigned_cluster = None
			for key, values in cluster_store.iteritems():				
				distance = euclidean_distance(row, values[0]) # change this index
				if closest_distance is None or closest_distance > distance:
					closest_distance = distance
					assigned_cluster = key
			assigned_cluster[row] = assigned_cluster
		return assigned_cluster


if __name__=='__main__':

    ## Create dummy dataframe
    df = pd.DataFrame()
    X1 = np.array([1,2,3,4,5,6,7,8,9,10])
    X2 = np.array([4,5,6,4,5,1,7,8,9,10])
    y = np.array([0,1,1,0,0,1,0,1,0,1])
    df['col_1'] = X1
    df['col_2'] = X2

    X = df[['col_1','col_2']]

    hc = HierarchicalClustering(n_clusters=2)
    hc.fit(X)
		