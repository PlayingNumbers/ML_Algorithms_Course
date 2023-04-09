import numpy as np
import pandas as pd
from collections import Counter


class Node:
    def __init__(self,
                 feature = None,
                 feature_value = None,
                 threshold = None,
                 data_left = None,
                 data_right = None,
                 gain = None,
                 value = None):

        self.feature = feature
        self.feature_value = feature_value
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X,y)

    def entropy(self, data):

        class_counts = np.bincount(data)
        class_probs = class_counts/len(data)

        class_entropies = []
        entropy = 0
        for prob in class_probs:
            if prob > 0:
                entropy += prob * np.log(prob)
                class_entropies.append(entropy)
        entropy = np.sum(class_entropies) * -1
        return entropy

    def information_gain(self,
                         parent,
                         left_child,
                         right_child):

        num_left = len(left_child)/len(parent)
        num_right = len(right_child)/len(parent)

        ## Compute entropies
        parent_entropy = self.entropy(parent)
        left_entropy = self.entropy(left_child)
        right_entropy = self.entropy(right_child)

        ## Compute information gain
        info_gain = parent_entropy - (num_left * left_entropy +
                                      num_right * right_entropy)
        return info_gain

    def _best_split(self, X, y):
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape

        feature_set = list(X.columns)

        # Aggregate X and y to create dataframe
        # We need to do this in order to compute information gain
        df = np.concatenate((X, np.array(y).reshape(1, -1).T), axis=1)
        df = pd.DataFrame(df)

        df.columns = feature_set + ['y']

        # Loop through each dataset feature
        for i, feature in enumerate(feature_set):

            # Store the feature data
            feature_data = sorted(np.unique(X[feature]))

            # Loop through each value and store the left and right.
            for feature_val in feature_data:

                # Store the left and right values
                df_left = df[df[feature] <= feature_val].copy()
                df_right = df[df[feature] > feature_val].copy()

                # Extract target variables
                if len(df_left) > 0 and len(df_right) > 0:
                    y_parent = df['y']
                    y_left = df_left['y']
                    y_right = df_right['y']

                    # Compute information gain
                    info_gain = self.information_gain(y_parent,
                                                      y_left,
                                                      y_right)
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_split = {
                            'feature_col':feature,
                            'split_value':feature_val,
                            'df_left':df_left,
                            'df_right':df_right,
                            'gain':info_gain
                        }
        return best_split

    def _build_tree(self, X, y, depth = 0):
        n_rows, n_cols = X.shape

        # Ensuring this isn't a leaf node. If so, we don't split.
        # This is the base case for the recursion
        if n_rows >= self.min_samples_split and depth <= self.max_depth:

            # Get best split
            best = self._best_split(X,y)

            # If information gain is not 0, possibly, room to split.
            if best['gain'] > 0:
                left = self._build_tree(
                    X=best['df_left'].drop(['y'], axis = 1),
                    y=best['df_left']['y'],
                    depth = depth + 1
                )

                right = self._build_tree(
                    X=best['df_right'].drop(['y'], axis = 1),
                    y=best['df_right']['y'],
                    depth = depth + 1
                )
                return Node(
                    feature=best['feature_col'],
                    threshold=best['split_value'],
                    data_left=left,
                    data_right=right,
                    gain=best['gain']
                )
        return Node(value=Counter(y).most_common(1)[0][0])

    def _traverse_tree(self, x, node):

        # If we hit leaf node, return that value
        if node.value != None:
            return node.value

        # Pull feature column
        feature_value = x[node.feature]

        # Go left if less than threshold
        if feature_value <= node.threshold:
            return self._traverse_tree(x=x, node = node.data_left)

        # Go right if more than threshold
        if feature_value > node.threshold:
            return self._traverse_tree(x=x, node = node.data_right)

    def predict(self, X):
        predictions = []
        for index, x in X.iterrows():
            pred = self._traverse_tree(x,self.root)
            predictions.append(pred)
        return predictions


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

    dt = DecisionTree()
    dt.fit(X, y)
    preds = dt.predict(X)

    print(preds)