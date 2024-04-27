import pickle as pkl
import numpy as np

def pca(X: np.array, k: int) -> np.array:
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    Return (a*b,k) array comprising the k normalised basis vectors comprising the k-dimensional subspace for all images
    where the first column must be the most relevant principal component and so on
    """
    target_images = X

    # Flatten the images to 2D arrays
    flattened_images = target_images.reshape(target_images.shape[0], -1)

    # Center the data
    mean_image = np.mean(flattened_images, axis=0)
    centered_images = flattened_images - mean_image

    # Compute the covariance matrix
    # Saksham
    covariance_matrix = np.cov(centered_images, rowvar=False)
    # Rathi
    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Take the first k eigenvectors
    k_eigenvectors = sorted_eigenvectors[:, :k]

    # Normalize the eigenvectors
    # Saksham
    normalized_eigenvectors = k_eigenvectors / np.linalg.norm(k_eigenvectors, axis=0)
    # Rathi
    return normalized_eigenvectors
    

def projection(X: np.array, basis: np.array):
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    basis is an (a*b,k) array comprising of k normalised vectors
    Return (n,k) np array comprising the k dimensional projections of the N images on the normalised basis vectors
    """
    target_images = X
    # Flatten the images into 1D arrays
    target_images_flat = target_images.reshape(target_images.shape[0], -1)
    # Compute projections onto the basis vectors
    projections = np.matmul(target_images_flat, basis)
    return projections
    
