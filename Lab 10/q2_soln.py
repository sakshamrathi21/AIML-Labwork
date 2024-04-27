import numpy as np
import pickle as pkl

class LDA:
    def __init__(self,k):
        self.n_components = k
        self.linear_discriminants = None

    def fit(self, X, y):
        """#X: (n,d) array consisting of input features
        #y: (n,) array consisting of labels
        #return: Linear Discriminant np.array of size (d,k)"""
        #Flattening the martix
        X = X.reshape(X.shape[0], -1)
        #Extracting target labels
        class_labels = np.unique(y)
        n_features = X.shape[1]
        mean_overall = np.mean(X, axis=0)
        #Initializing S_W and S_B
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        #Constructing S_W and S_B for each target label
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(-1, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)
        #Finding eigenvalues and eigenvectors
        #Note the use of pinv() to find psuodeoinverse
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
        #Sorting eigenvectors in decreasing order based on eigenvalues 
        eigenvectors = eigenvectors[:, np.argsort(abs(eigenvalues))[::-1]]
        #Extracting k (self.n_components) eigenvectors corresponding to largest k eigenvalues
        self.linear_discriminants = eigenvectors[:, :self.n_components]
        return(self.linear_discriminants)
    def transform(self, X, w):
        """#w:Linear Discriminant array of size (d,1)
        #return: the projected features of size (n,1)"""
        #Flattening the matrix
        X = X.reshape(X.shape[0], -1)
        #Returning projections
        return np.matmul(X,w)

