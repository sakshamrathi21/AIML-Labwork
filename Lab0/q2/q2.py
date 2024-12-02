import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA
    mean_row = init_array.mean(axis=1)
    standard_matrix = (init_array - np.mean(init_array, axis=0))
    square_matrix = standard_matrix.iloc[:, :standard_matrix.shape[1]]
    cov_array = square_matrix.cov()
    eigenvalues, eigenvectors = np.linalg.eig(cov_array)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvalues = np.round(sorted_eigenvalues, decimals=4)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    transformation_matrix = sorted_eigenvectors[:, :dimensions]
    final_data = np.dot(standard_matrix, transformation_matrix)

    # END TODO

    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png
    plt.scatter(final_data[:, 0], final_data[:, 1])
    plt.axis('equal')
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    # plt.show()
    plt.savefig("out.png")
    # END TODO
