import numpy as np
import pandas as pd


class KMeans:
	def __init__(self, k, max_iter):
		self.k = k
		self.max_iter = max_iter
		self.centroids = None

	def _euclidean_distance(self, X1, X2):
		return np.linalg.norm(X1 - X2)

	def _compute_centroid(self,
						  X,
						  assigned_centroid_dict = None):

		# Loop through each key, value in dictionary
		# For each one of these dataframes, take the mean to get the new centroid
		# return the new means
		self.centroids = []
		for centroid, centroid_df in assigned_centroid_dict.items():

			centroid_mean = pd.DataFrame(centroid_df.mean(axis = 0))

			centroid_mean = centroid_mean.T

			self.centroids.append(centroid_mean)

		return pd.concat(self.centroids)


	def predict(self, X):
				
		# Loop through each centroid, get distances
		assigned_centroid_dict = {}

		for row_num, row_X in X.iterrows():

			row_X_df = pd.DataFrame(row_X).T

			assigned_centroid = None
			closest_distance = None

			for centroid_num, row_c in self.centroids.iterrows():

				distance = self._euclidean_distance(row_c, row_X)

				if assigned_centroid is None:
					assigned_centroid = centroid_num
					closest_distance = distance
					continue
				
				# Replace assigned centroid if closer
				elif distance < closest_distance:
					assigned_centroid = centroid_num
					closest_distance = distance

			if assigned_centroid not in assigned_centroid_dict.keys():
				assigned_centroid_dict[assigned_centroid] = row_X_df

			else:
				assigned_centroid_dict[assigned_centroid].append(row_X_df)

		return assigned_centroid_dict

	def fit(self, X):

		# Initialize centroids randomly if first time.
		self.centroids = X.sample(self.k)

		for i in range(self.max_iter):
			self.assigned_centroid_dict = self.predict(X)
			self.centroids = self._compute_centroid(X, self.assigned_centroid_dict)


if __name__=='__main__':

    ## Create dummy dataframe
    df = pd.DataFrame()
    X1 = np.array([1,2,3,4,5,6,7,8,9,10])
    X2 = np.array([4,5,6,4,5,1,7,8,9,10])
    y = np.array([0,1,1,0,0,1,0,1,0,1])
    df['col_1'] = X1
    df['col_2'] = X2

    X = df[['col_1','col_2']]

    km = KMeans(k=2,max_iter = 1)
    km.fit(X)

    print(km.centroids)



