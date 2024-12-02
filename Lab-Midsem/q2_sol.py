import numpy as np

def data_generate(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    X = data[:, 0].reshape(-1,1)
    y = data[:, 1].astype(int).reshape(-1,1)
    
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, y_train, X_test, y_test

def polynomial_kernel(X1, X2, degree, bias):
    # TODO : Implement the polynomial kernel, and return the computed kernel
    # Here X1 and X2 are Mx1 and Nx1 matrices, and the result should be an MxN matrix
    # The value at (i,j)-th entry of the output should be equal to K(X1_i, X2_j)
    return (np.dot(X1, X2.T) + bias)**degree

def sigmoid(z):
    # TODO : Implement the sigmoid function, and return its value
    # Here, z is an Nx1 vector, and the result should also be an Nx1 vector
    # The ith entry in the output should be the sigmoid of z_i
    return 1 / (1 + np.exp(-z))

def predict(X_poly, weights, bias):
    return sigmoid(np.dot(X_poly, weights) + bias)

def compute_cost(predictions, labels):
    # TODO : Implement the negative log likelihood loss function, and return the cost
    # Here predictions and labels are Nx1 vectors, and the output should be a scalar
    m = len(labels)
    cost = -1/m * np.sum(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
    return cost

def update_parameters(features, predictions, labels, weights, bias, learning_rate):
    # TODO : Update the parameters weights and bias, and return the updated values
    # Here features is the kernel polynomial feature matrix of size NxM (M=N)
    # Also predictions and labels are Nx1 vectors, and weights is an Mx1 vector
    m = len(labels)
    dz = predictions - labels
    dw = (1/m) * np.dot(features.T, dz)
    db = (1/m) * np.sum(dz)
    # print(np.shape(learning_rate))
    dw = np.reshape(dw, (len(dw), 1))
    # print(np.shape(dw))
    weights -= learning_rate * dw
    bias -= learning_rate * db

    return weights, bias

def update_parameters_2(features, predictions, labels, weights, bias, learning_rate):
    # TODO : Update the parameters weights and bias, and return the updated values
    # Here features is the kernel polynomial feature matrix of size NxM (M=N)
    # Also predictions and labels are Nx1 vectors, and weights is an Mx1 vector
    m = len(labels)
    dz = predictions - labels
    dw = (1/m) * np.dot(features.T, dz)
    db = (1/m) * np.sum(dz)
    
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    return weights, bias

def logistic_regression(X_train, y_train, X_test, y_test, degree, learning_rate, num_epochs):
    bias = 1 
    X_train_poly = polynomial_kernel(X_train, X_train, degree, bias)
    X_test_poly = polynomial_kernel(X_test, X_train, degree, bias)

    num_features = X_train_poly.shape[1]
    weights = np.zeros(num_features).reshape(-1,1)
    bias = 0

    for epoch in range(num_epochs):
        predictions = predict(X_train_poly, weights, bias)
        cost = compute_cost(predictions, y_train)
        weights, bias = update_parameters(X_train_poly, predictions, y_train, weights, bias, learning_rate)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Cost: {cost}')

    # Testing
    test_predictions = np.round(predict(X_test_poly, weights, bias))
    accuracy = np.mean(test_predictions == y_test)
    print(f'Test Accuracy: {100.0 * accuracy}%')

    return accuracy
