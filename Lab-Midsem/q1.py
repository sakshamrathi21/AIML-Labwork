import numpy as np

'''DO NOT change anything in this function'''
def load_dataset(filename):
	data = np.loadtxt(filename, delimiter=',', skiprows=1)
	X = data[:, :-1]
	y = data[:, -1]
    
	split_ratio = 0.8
	split_index = int(len(X) * split_ratio)
	X_train, X_test = X[:split_index], X[split_index:]
	y_train, y_test = y[:split_index], y[split_index:]
     
	return X_train, y_train, X_test, y_test
    

def gaussian_kernel(x, xi, bandwidth):
	# TODO : Return Gaussian Kernel K(x,xi) (as described in the QP)
	# print(np.shape(x))
	product = (1/pow((np.sqrt(2*np.pi)*bandwidth),np.shape(x)[0]))*np.exp(-np.sum((x-xi)**2)/(2*bandwidth*bandwidth))
	# product = 1/(np.sqrt(2*np.pi)*bandwidth)
	# product = pow(product, np.shape(x))
	# print(np.sum((x-xi)**2))
	
	# product = product*np.exp(-np.sum((x-xi)**2)/(2*bandwidth*bandwidth))
	return product


def kernel_regression(X, y, x0, bandwidth):
	# TODO : Return predicted value for x0 (as described in the QP)
	# print(np.shape(x0))
	numerator = 0
	denominator = 0
	for i in range(np.shape(X)[0]):
		numerator += gaussian_kernel(x0, X[i], bandwidth)*y[i]
		denominator += gaussian_kernel(x0, X[i], bandwidth)
	return (numerator/denominator)


'''DO NOT change anything in this function'''
def rmse(X_train, y_train, X_test, y_test, bandwidth):
	# Perform kernel regression for each point in the test set
	predictions = [kernel_regression(X_train, y_train, test_point, bandwidth)
							 for test_point in X_test]
	
	# Calculate RMSE for the test set
	rmse = np.sqrt(np.mean((y_test - predictions)**2))
	return rmse


def find_optimal_bandwidth(X_train, y_train, X_test, y_test, bandwidth_values):
	# TODO : Return the value of bandwidth which produces the lowest RMSE on test set
	minimum_band = bandwidth_values[0]
	for band in bandwidth_values:
		if (rmse(X_train, y_train, X_test, y_test, band)) < (rmse(X_train, y_train, X_test, y_test, minimum_band)):
			minimum_band = band
	return minimum_band


'''DO NOT change anything below'''
if __name__ == '__main__':
	dataset = 'dataset.csv'
    
	# Split the data into train and test sets (80% train, 20% test)
	X_train, y_train, X_test, y_test = load_dataset(dataset)

	bandwidth_values = np.arange(0.1, 2.01, 0.05)
	
	print("RMSE for different bandwidths")
	for bandwidth in bandwidth_values:
		error = rmse(X_train, y_train, X_test, y_test, bandwidth)
		print(np.round(bandwidth, 2), ": ", error)

	optimal_bandwidth = find_optimal_bandwidth(X_train, y_train, X_test, y_test, bandwidth_values)
	print("\nOptimal bandwidth: ", np.round(optimal_bandwidth, 2))