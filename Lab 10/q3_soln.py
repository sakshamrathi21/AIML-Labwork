import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """ 
    return np.sqrt(np.sum((point - data)**2, axis=1))

def euclidean_squared(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sum((point - data)**2, axis=1)

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []

    def initialise(self, X_train):
        """
        Initialize the self.centroids class variable, using the "k-means++" method, 
        Pick a random data point as the first centroid,
        Pick the next centroids with probability directly proportional to their distance from the closest centroid
        Function returns self.centroids
        USE np.random for any random number generation that you may require. Do NOT use the random module!
        """
        # Saksham
        self.centroids = [X_train[np.random.randint(0,len(X_train))]]
        # Rathi
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.min([euclidean_squared(centroid, X_train) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            # Saksham
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            # Rathi
            self.centroids += [X_train[new_centroid_idx]]
        return np.array(self.centroids )
    def fit(self, X_train):
        """
        Updates the self.centroids class variable using the two-step iterative algorithm on the X_train dataset.
        X_train has dimensions (N,d) where N is the number of samples and each point belongs to d dimensions
        Ensure that the total number of iterations does not exceed self.max_iter
        Function returns self.centroids
        """
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
        return np.array(self.centroids)
    def evaluate(self, X):
        """
        Given N data samples in X, find the cluster that each point belongs to 
        using the self.centroids class variable as the centroids.
        Return two arrays, the first being self.centroids 
        and the second is an array having length equal to the number of data points 
        and each entry being between 0 and K-1 (both inclusive) where K is number of clusters.
        """
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return np.array(self.centroids), np.array(centroid_idxs)

def evaluate_loss(X, centroids, classification):
    loss = 0
    for idx, point in enumerate(X):
        loss += np.linalg.norm(point - centroids[classification[idx]])

    return loss

