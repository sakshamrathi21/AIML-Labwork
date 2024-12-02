import pickle as pkl
import numpy as np

def pca(X: np.array, k: int) -> np.array:
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    Return (a*b,k) np array comprising the k normalised basis vectors comprising the k-dimensional subspace for all images
    where the first column must be the most relevant principal component and so on
    """
    # TODO
    # pass
    N,a,b = X.shape
    X_flattened = X.reshape(N, a*b)
    X_mean = np.mean(X_flattened, axis=0)
    X_std = X_flattened - X_mean
    covariance_matrix = np.cov(X_std, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, idx]
    basis_vectors = eigenvectors_sorted[:, :k]
    return basis_vectors
    #END TODO
    

def projection(X: np.array, basis: np.array):
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    basis is an (a*b,k) array comprising of k normalised vectors
    Return (n,k) np array comprising the k dimensional projections of the N images on the normalised basis vectors
    """
    # TODO
    N,a,b = X.shape
    X_flattened = X.reshape(N, a*b)
    projections = np.dot(X_flattened, basis)
    return projections
    # END TODO
    