import numpy as np

'''DO NOT change anything in this function'''
def load_dataset(filename):
	data = np.loadtxt(filename, delimiter=',', skiprows=1)
	X = data[:, :-1]
	y = data[:, -1]
    
	split_ratio = 0.7
	split_index = int(len(X) * split_ratio)
	X_train, X_test = X[:split_index], X[split_index:]
	y_train, y_test = y[:split_index], y[split_index:]
     
	return X_train, y_train, X_test, y_test
    

def gaussian_kernel(x, xi, bandwidth):
	# TODO : Return Gaussian Kernel K(x,xi) (as described in the QP)
	diff = x - xi
	exponent = -0.5 * np.dot(diff, diff) / (bandwidth**2)
	return np.exp(exponent) / ((2 * np.pi)**(len(x) / 2) * bandwidth**len(x))


def kernel_regression(X, y, x0, bandwidth):
	# TODO : Return predicted value for x0 (as described in the QP)
	weights = np.array([gaussian_kernel(x0, xi, bandwidth) for xi in X])
	weighted_y = weights * y
	return np.sum(weighted_y) / np.sum(weights)


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
	errors = [rmse(X_train, y_train, X_test, y_test, bandwidth)
			  for bandwidth in bandwidth_values]
	errors = np.array(errors)
	return bandwidth_values[np.argmin(errors)]