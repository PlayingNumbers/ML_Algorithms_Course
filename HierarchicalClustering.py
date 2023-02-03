import numpy as np
import pandas as pd
from itertools import product

class HierarchicalClustering:
	def __init__(self, n_clusters):
		self.n_clusters = n_clusters
		
	def euclidean_distance(x1, x2):
	    return np.sqrt(np.sum((x1 - x2)**2))

	def _build_initial_clusters(self, X):
		# Place every data point in a cluster
		clusters = []
		for i in range(len(X)):
			clusters.append(X[i])

	def fit(self, X):

		self.clusters = self._build_initial_clusters(self, X)
		clusters_created = len(clusters)

		while clusters_created > self.n_clusters:

			closest_distance = None
			closest_pair = None
			
			# Find the closest pair.
			pairs_list = list(product(self.clusters, self.clusters))
			for pair in pairs:
				if pair[0] == pair[1]:
					continue
				
				# Compute distances
				distance = euclidean_distance(sorted(pair)[0], sorted(pair)[0])

				# Replace closest distance
				if distance < closest_distance or closest_distance is None:
					closest_distance = distance
					closest_pair = pair

			# Remove closest pair from the overall clusters
			# Add it back as its own cluster
			for i in closest_pair:
				closest_cluster = []
				for p in pair:
					self.clusters.remove(p)
					closest_cluster.append(p)

				self.clusters.append(closest_cluster)

			clusters_created = len(clusters)

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
		