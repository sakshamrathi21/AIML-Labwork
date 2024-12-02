import numpy as np
import pickle as pkl

class LDA:
    def __init__(self,k):
        self.n_components = k
        self.linear_discriminants = None

    def fit(self, X, y):
        """
        X: (n,d,d) array consisting of input features
        y: (n,) array consisting of labels
        return: Linear Discriminant np.array of size (d*d,k)
        """
        # TODO
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        # return np.zeros((X.shape[1]*X.shape[2],self.n_components))
        n_features = X.shape[1]
        class_labels = np.unique(y)
        mean_vectors = []
        for cls in class_labels:
            mean_vectors.append(np.mean(X[y == cls], axis=0))
        overall_mean = np.mean(X, axis=0)
        # print(np.shape(mea))
        S_W = np.zeros((n_features, n_features))
        for cls, mean_vec in zip(class_labels, mean_vectors):
            class_sc_mat = np.zeros((n_features, n_features))
            for row in X[y == cls]:
                # print("hello")
                row, mv = row.reshape(n_features, 1), mean_vec.reshape(n_features, 1)
                class_sc_mat += (row - mv).dot((row - mv).T)
            S_W += class_sc_mat
        # print(S_W)
        S_B = np.zeros((n_features, n_features))
        for mean_vec in mean_vectors:
            mean_vec = mean_vec.reshape(n_features, 1)  
            overall_mean = overall_mean.reshape(n_features, 1)
            S_B += len(X[y == cls]) * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        # print(np.linalg.inv(S_W).dot(S_B))
        # print(np.shape(np.linalg.inv(S_W).dot(S_B)))
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(S_W).dot(S_B))
        # print("hello")
        # eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
        # eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
        # self.linear_discriminants = np.hstack([pair[1].reshape(n_features, 1) for pair in eigen_pairs[:self.n_components]])
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors_sorted = eigenvectors[:, idx]
        basis_vectors = eigenvectors_sorted[:, :self.n_components]
        self.linear_discriminants = basis_vectors
        return self.linear_discriminants
        #END TODO 
    
    def transform(self, X, w):
        """
        w:Linear Discriminant array of size (d*d,1)
        return: np-array of the projected features of size (n,k)
        """
        # TODO
        X_flattened = X.reshape((np.shape(X)[0], np.shape(w)[0]))
        return np.dot(X_flattened,w)                  
        # END TODO
