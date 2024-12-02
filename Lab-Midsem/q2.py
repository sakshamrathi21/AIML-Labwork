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
    M = np.shape(X1)[0]
    N = np.shape(X2)[0]
    # print(M,N)
    output_matrix = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            output_matrix[i][j] = pow((X1[i]*X2[j]+bias), degree)
    return output_matrix

def sigmoid(z):
    # TODO : Implement the sigmoid function, and return its value
    # Here, z is an Nx1 vector, and the result should also be an Nx1 vector
    # The ith entry in the output should be the sigmoid of z_i
    # print(np.shape(z))
    # print(1/(1+np.exp(-z)))
    # print(np.shape(1/(1+np.exp(-z))))
    # print(z)
    y = np.zeros_like(z)
    # if z.any()<0:
    #     print("hello")
    for i in range(np.shape(z)[0]):
        if (z[i][0]>100):
            y[i][0] = 1
        elif (z[i][0]<-100):
            y[i][0] = 0
        else:
            y[i][0] = (1/(1+np.exp(-z[i][0])))
    return y

def predict(X_poly, weights, bias):
    return sigmoid(np.dot(X_poly, weights) + bias)

def compute_cost(predictions, labels):
    # TODO : Implement the negative log likelihood loss function, and return the cost
    # Here predictions and labels are Nx1 vectors, and the output should be a scalar
    # sum = 0
    # for i in range(np.shape(predictions)[0]):
    #     sum += (labels[i][0]*np.log(predictions[i][0]) + (1-labels[i][0])*np.log(1-predictions[i][0]))
    # print(np.dot(np.transpose(np.log(predictions)), labels))
    # print((-1/np.shape(predictions)[0])*sum)
    # return (-1/np.shape(predictions)[0])*sum
    cost = (-1/np.shape(predictions)[0])*((np.dot(np.transpose(np.log(predictions)), labels))+(np.dot(np.transpose(np.log(1-predictions)),(1-labels))))
    if (np.shape(cost) == (1,1)):
        return cost[0][0]
    else:
        return cost

def update_parameters(features, predictions, labels, weights, bias, learning_rate):
    # TODO : Update the parameters weights and bias, and return the updated values
    # Here features is the polynomial kernel feature matrix of size NxM (M=N)
    # Also predictions and labels are Nx1 vectors, and weights is an Mx1 vector
    weights = weights.reshape(np.shape(weights)[0],1)
    predictions = predictions.reshape(np.shape(predictions)[0],1)
    labels = labels.reshape(np.shape(labels)[0],1)
    weights_change = np.zeros_like(weights)
    bias_change = 0
    # print(np.shape(weights), np.shape(weights_change), np.shape(predictions), np.shape(features))
    for i in range(np.shape(predictions)[0]):
        # print(np.shape(weights_change), np.shape(predictions[i]-labels[i]))
        weights_change += (predictions[i]-labels[i])*(features[i].reshape(np.shape(features)[1],1))
        # print(np.shape(features))
        bias_change += (predictions[i][0]-labels[i][0])
    weights -= learning_rate*(1/np.shape(predictions)[0])*weights_change
    bias -= learning_rate*(1/np.shape(predictions)[0])*bias_change
    # weights -= learning_rate*(1/np.shape(predictions)[0])*np.sum(np.dot(np.transpose(features),(predictions-labels)))
    # bias -= learning_rate*(1/np.shape(predictions)[0])*np.sum(predictions-labels)
    return weights,bias

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

if __name__ == '__main__':
    dataset = 'dataset.csv'
    degree = 3
    learning_rate = 0.01
    num_epochs = 1000
    
    X_train, y_train, X_test, y_test = data_generate(dataset)
    
    # Train and evaluate polynomial kernel logistic regression
    logistic_regression(X_train, y_train, X_test, y_test, degree,
                        learning_rate, num_epochs)
