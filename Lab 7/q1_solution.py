import numpy as np
import random
import logging
import pickle

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


class NeuralNetwork:

    def __init__(self, num_layers, num_neurons_per_layer_list, seed=400):
        logging.info("Creating a NN with {} layers having {} neurons respectively".format(num_layers,
                                                                                          num_neurons_per_layer_list))
        seed_everything(seed)
        self.seed = seed
        self.num_layers = num_layers
        self.num_neurons_per_layer_list = num_neurons_per_layer_list
        # The following self.weights and self.biases list contains layer-by-layer weights and biases as numpy arrays
        # For example, if we have an input layer with 784 neurons, two hidden layer with 8 neurons, and an output layer
        # with 10 neurons:
        # self.biases = [numpy array (8, 1), numpy array (8, 1), numpy array (10, 1)]
        # self.weights = [numpy array (784, 8), numpy array (8, 8), numpy array (8, 10)]
        self.biases = [np.random.randn(i, 1) for i in num_neurons_per_layer_list[1:]]
        self.weights = [np.random.randn(i, j)
                        for i, j in zip(num_neurons_per_layer_list[:-1], num_neurons_per_layer_list[1:])]
        pass

    @staticmethod
    def sigmoid(z):
        """
        :param z: numpy array (number of neurons, 1)
        :return: numpy array (number of neurons, 1)
        """
        sigmoid_z = np.zeros(z.shape)
        sigmoid_z = 1.0 / (1.0 + np.exp(-z))
        return sigmoid_z

    @staticmethod
    def cost_derivative(output_activations, y):
        """
        Derivative of cost w.r.t final activations
        :param output_activations: numpy array (number of neurons in output layer, 1)
        :param y: numpy array (number of neurons in output layer, 1)
        :return: numpy array (number of neurons in output layer, 1)
        """
        derivative_cost = np.zeros(output_activations.shape)
        derivative_cost = output_activations - y
        return derivative_cost

    def cost(self, training_data):
        """ Calculate cost (sum of squared differences) over training data for the current weights and biases
        :param training_data:
        list of 50000 samples, each item of the list is a tuple (x, y)
        x - input numpy array (784, 1)  y - labelled output numpy array (10, 1)
        :return: cost - float
        """
        logging.info("Calcuting costs...")
        cost = 0
        for data in training_data:
            nn_output = self.forward_pass(data[0])
            y = data[1]
            cost += np.sum(0.5 * (nn_output - y) ** 2)
        logging.info("Calcuting costs complete...")
        return cost

    def sigmoid_derivative(self, z):
        """Derivative of the sigmoid function."""
        derivative_sigmoid = np.zeros(z.shape)
        derivative_sigmoid = self.sigmoid(z) * (1 - self.sigmoid(z))
        return derivative_sigmoid

    def forward_pass(self, x):
        """
        Perform forward pass and return the output of the NN.
        :param x: numpy array (784, 1)
        :return: numpy array (10, 1)
        """
        nn_output = np.zeros((self.num_neurons_per_layer_list[-1], 1))
        nn_output = x
        for b, w in zip(self.biases, self.weights):
            nn_output = self.sigmoid(np.dot(w.T, nn_output) + b)
        return nn_output

    def mini_batch_GD(self, training_data, epochs, mini_batch_size, eta, test_data):
        """ Train the neural network using mini batch gradient descent
        :param training_data: list of tuples, where each tuple is a single labelled data point as follows.
         "(x - numpy array (784, 1), y (10, 1))"
        :param epochs: int
        :param mini_batch_size: int
        :param eta: learning rate float
        :param test_data: test data
        :return: cost_history - list (append cost after every epoch to this list, this will used to reward marks for
        gradient descent update step)
        """
        logging.info("Running mini_batch_GD")
        num_samples = len(training_data)
        costs_history = []
        for j in range(epochs):
            random.Random(self.seed).shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, num_samples, mini_batch_size)]
            for mini_batch in mini_batches:
                sum_dc_db_list = [np.zeros(b.shape) for b in self.biases]
                sum_dc_dw_list = [np.zeros(w.shape) for w in self.weights]
                for x, y in mini_batch:
                    dc_db_list, dc_dw_list = self.back_propagation(x, y)
                    sum_dc_db_list = [dC_db_sum + dC_db for dC_db_sum, dC_db in zip(sum_dc_db_list, dc_db_list)]
                    sum_dc_dw_list = [dC_dw_sum + dC_dw for dC_dw_sum, dC_dw in zip(sum_dc_dw_list, dc_dw_list)]

                self.weights = [w - (eta / len(mini_batch)) * dC_dw_sum for w, dC_dw_sum in
                                zip(self.weights, sum_dc_dw_list)]
                self.biases = [b - (eta / len(mini_batch)) * dC_db_sum for b, dC_db_sum in
                               zip(self.biases, sum_dc_db_list)]

            logging.info("After Epoch {}".format(j + 1))
            self.test_accuracy(test_data)
            cost_curr = self.cost(training_data)
            costs_history.append(cost_curr)
            logging.info("Cost: {}".format(cost_curr))
        return costs_history

    def back_propagation(self, x, y):
        """Compute gradients
        :param x: Input to the NN - numpy array (784, 1)
        :param y: Labelled output data - numpy array (10, 1)
        :return: tuple (dC_db, dC_dw) - list of numpy arrays, similar to self.biases and self.weights
        """
        # These list contain layer-by-layer gradients as numpy arrays, similar to self.biases and self.weights
        # For example, if we have an input layer with 784 neurons, two hidden layer with 8 neurons, and an output layer
        # with 10 neurons:
        # dC_db = [numpy array (8, 1), numpy array (8, 1), numpy array (10, 1)]
        # dC_dw = [numpy array (784, 8), numpy array (8, 8), numpy array (8, 10)]
        dC_db = [np.zeros(b.shape) for b in self.biases]
        dC_dw = [np.zeros(w.shape) for w in self.weights]

        # forward pass
        activation = x
        activations = [x]
        z_list = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w.T, activation) + b
            activation = self.sigmoid(z)
            z_list.append(z)
            activations.append(activation)
        # backward pass
        temp = self.cost_derivative(activations[-1], y) * self.sigmoid_derivative(z_list[-1])
        dC_db[-1] = temp
        dC_dw[-1] = np.dot(temp, activations[-2].transpose()).T

        for i in range(2, self.num_layers):
            z = z_list[-i]
            sigmoid_derivative = self.sigmoid_derivative(z)
            temp = np.dot(self.weights[-i + 1], temp) * sigmoid_derivative
            dC_db[-i] = temp
            dC_dw[-i] = np.dot(temp, activations[-i - 1].transpose()).T
        return dC_db, dC_dw

    def test_accuracy(self, test_data):
        acc_val = 0
        for x, y in test_data:
            if y == np.argmax(self.forward_pass(x)):
                acc_val += 1
        logging.info("Test accuracy {}".format(round(acc_val / len(test_data) * 100, 2)))


def load_data(file="data.pkl"):
    def vectorize(j):
        y = np.zeros((10, 1))
        y[j] = 1.0
        return y

    logging.info("loading data...")
    with open(file, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
        training_outputs = [vectorize(y) for y in training_data[1]]
        training_data = list(zip(training_inputs, training_outputs))
        test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
        test_data = list(zip(test_inputs, test_data[1]))
    logging.info("loaded data...")
    return training_data, test_data


def seed_everything(seed=0):
    np.random.seed(seed)


if __name__ == "__main__":
    data_training, data_test = load_data()
    net = NeuralNetwork(num_layers=4, num_neurons_per_layer_list=[784, 50, 50, 10])
    net.mini_batch_GD(data_training, 15, 10, 3, data_test)
