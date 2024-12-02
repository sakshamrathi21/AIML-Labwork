import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.sizes = [input_size]
        self.sizes.extend(hidden_sizes)
        self.sizes.append(output_size)
        # now self.sizes littrally contains every sizes.
        self.weights = [np.random.rand(self.sizes[i+1], self.sizes[i]) for i in range(len(self.sizes)-1)]
        self.biases = [np.random.rand(self.sizes[i+1], 1) for i in range(len(self.sizes)-1)]
        # self.weights[i] is of level i to i+1
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def error(self, target, output):
        return np.sum((target-output)**2)

    def forward(self, inputs):
        inputs = np.array(inputs)
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1).T
        self.activations = [] # would be of size levels+1 with initial activations as inputs, and last as predicted output.
        activation = np.array(inputs).T
        self.activations.append(activation)
        # Forward pass through hidden layers
        for i in range(0, len(self.weights)):
            activation = np.array(self.sigmoid( self.weights[i] @ activation + self.biases[i]))
            self.activations.append(activation)
        return activation
    
    def backward(self, inputs, targets, learning_rate, lam):
        pred_out = self.forward(inputs)
        if lam % 500 == 0:
            print(self.error(targets, pred_out))
        self.derivative = [np.zeros_like(a) for a in self.activations]  # for every activation, there will be some derivative dC/da^l
        # its size is <-- inputlayer + hidden + output -->
        # self.weights is of 1 lenghth lesser actually.

        self.derivative[-1] = 2 * (pred_out - targets.T)/self.sizes[-1]

        for i in range(len(self.weights)-1 , -1, -1):

            self.derivative[i] = self.weights[i].T @ ((self.activations[i+1] * (1 - self.activations[i+1])) * self.derivative[i+1])
 
            gradient = ((self.activations[i+1] * (1 - self.activations[i+1])) * self.derivative[i+1]) @ self.activations[i].T

            self.weights[i] -= learning_rate * gradient/inputs.shape[0]
            self.biases[i] -= learning_rate * np.sum((self.activations[i+1] * (1 - self.activations[i+1])) * self.derivative[i+1], axis=1, keepdims=True) / inputs.shape[0]

    
    def train(self, inputs, targets, epochs, learning_rate):
        for lam in range(epochs):
            self.backward(inputs[:2], targets[:2], learning_rate, lam)
            self.backward(inputs[1:3], targets[1:3], learning_rate, lam)
            self.backward(inputs[2:4], targets[2:4], learning_rate, lam)
            self.backward(inputs[[0,3]], targets[[0,3]], learning_rate, lam)