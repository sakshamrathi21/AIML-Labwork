import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA
    # dimension=init_array.shape
    # print(dimension)
    standardized_matrix = (init_array - np.mean(init_array, axis=0))
    covariance_matrix = np.cov(init_array, rowvar=False)
    # print(covariance_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # print(sorted_eigenvalues)
    # print(sorted_eigenvectors)
    selected_eigenvectors = sorted_eigenvectors[:, :dimensions]

    # Transform the original matrix
    transformed_matrix = np.dot(init_array, selected_eigenvectors)

    # Create a DataFrame for the transformed matrix
    final_data = pd.DataFrame(transformed_matrix)
    sorted_eigenvalues=sorted_eigenvalues.round(4)
    # Save the transformed matrix to a CSV file
   # transformed_df.to_csv("transform.csv", index=False)
    # END TODO

    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png
    plt.scatter(final_data[0],final_data[1])
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.gca().set_aspect('equal')
    plt.savefig("out.png")
    plt.show()
    # END TODO