import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

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
        Function returns self.centroids as an np.array
        USE np.random for any random number generation that you may require 
        (Generate no more than K random numbers). 
        Do NOT use the random module at ALL!
        """
        self.centroids = [X_train[np.random.randint(len(X_train))]]
        for _ in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(p - c)**2 for c in self.centroids]) for p in X_train])
            probabilities = distances / distances.sum()
            next_centroid = X_train[np.random.choice(len(X_train), p=probabilities)]
            self.centroids.append(next_centroid)
        self.centroids = np.array(self.centroids)
        return self.centroids

    def fit(self, X_train):
        """
        Updates the self.centroids class variable using the two-step iterative algorithm on the X_train dataset.
        X_train has dimensions (N,d) where N is the number of samples and each point belongs to d dimensions
        Ensure that the total number of iterations does not exceed self.max_iter
        Function returns self.centroids as an np array
        """
        # self.initialise(X_train)
        for _ in range(self.max_iter):
            classifications = np.argmin(np.linalg.norm(X_train[:, np.newaxis] - self.centroids, axis=2), axis=1)

            new_centroids = np.array([X_train[classifications == k].mean(axis=0) for k in range(self.n_clusters)])
            self.centroids = new_centroids

        return self.centroids

    def evaluate(self, X):
        """
        Given N data samples in X, find the cluster that each point belongs to 
        using the self.centroids class variable as the centroids.
        Return two np arrays, the first being self.centroids 
        and the second is an array having length equal to the number of data points 
        and each entry being between 0 and K-1 (both inclusive) where K is number of clusters.
        """
        classifications = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)
        return self.centroids, classifications

def evaluate_loss(X, centroids, classification):
    loss = 0
    for idx, point in enumerate(X):
        loss += np.linalg.norm(point - centroids[classification[idx]])
    return loss
