import q1, q2, q3
import q1_sol, q2_sol, q3_sol
import numpy as np
import pickle as pkl
import sys
import unittest

testing_seed = 12345

def check_output(output, expected):
    if type(output) == str:
        output = float(output)

    if abs(output - expected) < 1e-4:
        return True
    return False

class TestFunctions(unittest.TestCase):

    def __init__(self):
        super().__init__()
        self.total_marks = 0

    def run_test(self, test_function, marks):
        try:
            test_function()
            print("Passed:", test_function.__name__)
            print("Marks:", marks)
            self.total_marks += marks
            print("=========================================")
        except AssertionError as e:
            print("Failed:", e)
            print("Marks: 0")
            print("=========================================")

    def test_q1_gaussian_kernel_1(self):
        """Evaluate test case to check gaussian kernel on random input for Q1"""

        np.random.seed(testing_seed)
        x = np.random.rand(4)
        xi = np.random.rand(4)
        bandwidth = 0.5

        output = q1.gaussian_kernel(x, xi, bandwidth)
        gold_output = q1_sol.gaussian_kernel(x, xi, bandwidth)

        self.assertEqual(check_output(gold_output, output),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on x {} xi {} and bandwidth {}'.format(
                             gold_output, output, x, xi, bandwidth))


    def test_q1_gaussian_kernel_2(self):
        """Evaluate test case to check gaussian kernel on random input for Q1"""

        np.random.seed(testing_seed)
        x = np.random.rand(6)
        xi = np.random.rand(6)
        bandwidth = 0.5

        output = q1.gaussian_kernel(x, xi, bandwidth)
        gold_output = q1_sol.gaussian_kernel(x, xi, bandwidth)

        self.assertEqual(check_output(gold_output, output),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on x {} xi {} and bandwidth {}'.format(
                             gold_output, output, x, xi, bandwidth))
        

    def test_q1_kernel_regression_1(self):
        """Evaluate test case to check kernel regression on random input for Q1"""

        np.random.seed(testing_seed)
        X = np.random.rand(7, 4)
        y = np.random.rand(7)
        x0 = np.random.rand(4)
        bandwidth = 0.5

        output = q1.kernel_regression(X, y, x0, bandwidth)
        gold_output = q1_sol.kernel_regression(X, y, x0, bandwidth)

        self.assertEqual(check_output(gold_output, output),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on X {} y {} x0 {} and bandwidth {}'.format(
                             gold_output, output, X, y, x0, bandwidth))
        

    def test_q1_kernel_regression_2(self):
        """Evaluate test case to check kernel regression on random input for Q1"""

        np.random.seed(testing_seed)
        X = np.random.rand(9, 7)
        y = np.random.rand(9)
        x0 = np.random.rand(7)
        bandwidth = 0.9

        output = q1.kernel_regression(X, y, x0, bandwidth)
        gold_output = q1_sol.kernel_regression(X, y, x0, bandwidth)

        self.assertEqual(check_output(gold_output, output),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on X {} y {} x0 {} and bandwidth {}'.format(
                             gold_output, output, X, y, x0, bandwidth))
        

    def test_q1_optimal_bandwidth_1(self):
        """Evaluate test case to check optimal bandwidth on random input for Q1"""

        np.random.seed(testing_seed)
        X_train, y_train, X_test, y_test = q1_sol.load_dataset('q1_hidden_dataset.csv')
        bandwidth_values = np.arange(0.15, 2.01, 0.05)

        bandwidth = q1.find_optimal_bandwidth(X_train, y_train, X_test, y_test, bandwidth_values)
        gold_bandwidth = q1_sol.find_optimal_bandwidth(X_train, y_train, X_test, y_test, bandwidth_values)

        self.assertEqual(check_output(gold_bandwidth, bandwidth),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on X_train {} y_train {} X_test {} y_test {} and bandwidth_values {}'.format(
                             gold_bandwidth, bandwidth, X_train, y_train, X_test, y_test, bandwidth_values))
        

    def test_q1_optimal_bandwidth_2(self):
        """Evaluate test case to check optimal bandwidth on random input for Q1"""

        np.random.seed(testing_seed)
        X_train, y_train, X_test, y_test = q1_sol.load_dataset('q1_hidden_dataset.csv')
        bandwidth_values = np.arange(0.15, 3.01, 0.09)

        bandwidth = q1.find_optimal_bandwidth(X_train, y_train, X_test, y_test, bandwidth_values)
        gold_bandwidth = q1_sol.find_optimal_bandwidth(X_train, y_train, X_test, y_test, bandwidth_values)

        self.assertEqual(check_output(gold_bandwidth, bandwidth),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on X_train {} y_train {} X_test {} y_test {} and bandwidth_values {}'.format(
                             gold_bandwidth, bandwidth, X_train, y_train, X_test, y_test, bandwidth_values))
        

    def test_q2_polynomial_kernel_1(self):
        """Evaluate test case to check polynomial kernel on random input for Q2"""

        np.random.seed(testing_seed)
        X1 = np.random.rand(5, 1)
        X2 = np.random.rand(6, 1)
        degree = np.random.randint(1, 10)
        c = np.random.rand(1)

        output = q2.polynomial_kernel(X1, X2, degree, c)
        gold_output = q2_sol.polynomial_kernel(X1, X2, degree, c)

        for i in range(len(gold_output)):
            for j in range(len(gold_output[i])):
                self.assertEqual(check_output(gold_output[i][j], output[i][j]),
                                 True,
                                 msg='Your output is incorrect:\n Expected {} \n Yours {} on X1 {} X2 {} degree {} and c {}'.format(
                                     gold_output, output, X1, X2, degree, c))
                

    def test_q2_polynomial_kernel_2(self):
        """Evaluate test case to check polynomial kernel on random input for Q2"""

        np.random.seed(testing_seed)
        X1 = np.random.rand(9, 1)
        X2 = np.random.rand(10, 1)
        degree = np.random.randint(1, 5)
        c = np.random.rand(1)

        output = q2.polynomial_kernel(X1, X2, degree, c)
        gold_output = q2_sol.polynomial_kernel(X1, X2, degree, c)

        for i in range(len(gold_output)):
            for j in range(len(gold_output[i])):
                self.assertEqual(check_output(gold_output[i][j], output[i][j]),
                                 True,
                                 msg='Your output is incorrect:\n Expected {} \n Yours {} on X1 {} X2 {} degree {} and c {}'.format(
                                     gold_output, output, X1, X2, degree, c))
        

    def test_q2_sigmoid_1(self):
        """Evaluate test case to check sigmoid on random input for Q2"""

        np.random.seed(testing_seed)
        x = np.random.rand(5, 1)

        output = q2.sigmoid(x)
        gold_output = q2_sol.sigmoid(x)

        for i in range(len(gold_output)):
            self.assertEqual(check_output(gold_output[i], output[i]),
                    True,
                    msg='Your output is incorrect:\n Expected {} \n Yours {} on x {}'.format(
                        gold_output, output, x))
            

    def test_q2_sigmoid_2(self):
        """Evaluate test case to check sigmoid on random input for Q2"""

        np.random.seed(testing_seed)
        x = np.random.rand(10, 1)

        output = q2.sigmoid(x)
        gold_output = q2_sol.sigmoid(x)

        for i in range(len(gold_output)):
            self.assertEqual(check_output(gold_output[i], output[i]),
                    True,
                    msg='Your output is incorrect:\n Expected {} \n Yours {} on x {}'.format(
                        gold_output, output, x))
        

    def test_q2_cost_function_1(self):
        """Evaluate test case to check cost function on random input for Q2"""

        np.random.seed(testing_seed)
        predictions = np.random.rand(5)
        labels = np.random.randint(0, 2, 5)

        output = q2.compute_cost(predictions, labels)
        gold_output = q2_sol.compute_cost(predictions, labels)

        self.assertEqual(check_output(gold_output, output),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on predictions {} and labels {}'.format(
                             gold_output, output, predictions, labels))
        

    def test_q2_cost_function_2(self):
        """Evaluate test case to check cost function on random input for Q2"""

        np.random.seed(testing_seed)
        predictions = np.random.rand(9)
        labels = np.random.randint(0, 2, 9)

        output = q2.compute_cost(predictions, labels)
        gold_output = q2_sol.compute_cost(predictions, labels)

        self.assertEqual(check_output(gold_output, output),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on predictions {} and labels {}'.format(
                             gold_output, output, predictions, labels))
        

    def test_q2_update_parameters_1(self):
        """Evaluate test case to check update parameters on random input for Q2"""
        # print("hello")
        np.random.seed(testing_seed)
        features = np.random.rand(5, 3)
        predictions = np.random.rand(5)
        labels = np.random.randint(0, 2, 5)
        weights = np.random.rand(3)
        bias = np.random.rand(1)
        learning_rate = 10
        
        weights, bias = q2.update_parameters(features, predictions, labels, weights, bias, learning_rate)
        
        # print(np.shape(weights))
        gold_weights, gold_bias = q2_sol.update_parameters(features, predictions, labels, weights, bias, learning_rate)
        gold_weights_2, gold_bias_2 = q2_sol.update_parameters_2(features, predictions, labels, weights, bias, learning_rate)
        # print(np.shape(gold_weights))
        for i in range(len(gold_weights)):
            self.assertEqual(check_output(gold_weights[i], weights[i]) or check_output(gold_weights_2[i], weights[i]),
                True,
                msg='Your output is incorrect:\n Expected {} \n Yours {} on features {} predictions {} labels {} weights {} bias {} and learning_rate {}'.format(
                    gold_weights, weights, features, predictions, labels, weights, bias, learning_rate))
            
        self.assertEqual(check_output(gold_bias[0], bias[0]) or check_output(gold_bias_2[0], bias[0]),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on features {} predictions {} labels {} weights {} bias {} and learning_rate {}'.format(
                             gold_bias, bias, features, predictions, labels, weights, bias, learning_rate))
        

    def test_q2_update_parameters_2(self):
        """Evaluate test case to check update parameters on random input for Q2"""

        np.random.seed(testing_seed)
        features = np.random.rand(8, 6)
        predictions = np.random.rand(8)
        labels = np.random.randint(0, 2, 8)
        weights = np.random.rand(6)
        bias = np.random.rand(1)
        learning_rate = 10

        weights, bias = q2.update_parameters(features, predictions, labels, weights, bias, learning_rate)
        gold_weights, gold_bias = q2_sol.update_parameters(features, predictions, labels, weights, bias, learning_rate)
        gold_weights_2, gold_bias_2 = q2_sol.update_parameters_2(features, predictions, labels, weights, bias, learning_rate)

        for i in range(len(gold_weights)):
            self.assertEqual(check_output(gold_weights[i], weights[i]) or check_output(gold_weights_2[i], weights[i]),
                True,
                msg='Your output is incorrect:\n Expected {} \n Yours {} on features {} predictions {} labels {} weights {} bias {} and learning_rate {}'.format(
                    gold_weights, weights, features, predictions, labels, weights, bias, learning_rate))
            
        self.assertEqual(check_output(gold_bias, bias) or check_output(gold_bias_2, bias),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on features {} predictions {} labels {} weights {} bias {} and learning_rate {}'.format(
                             gold_bias, bias, features, predictions, labels, weights, bias, learning_rate))
        

    def test_q2_accuracy_on_test_set(self):
        """Evaluate test case to check accuracy on test set on random input for Q2"""

        X_train, y_train, X_test, y_test = q2_sol.data_generate('q2_hidden_dataset.csv')
        degree = 4
        learning_rate = 0.0001
        num_epochs = 1000

        test_accuracy = q2.logistic_regression(X_train, y_train, X_test, y_test, degree, learning_rate, num_epochs)

        if test_accuracy < 0.5:
            self.assertEqual(False, True, msg='Your accuracy is too less:\n Expected > 0.5 \n Yours {}'.format(
                test_accuracy))
            
    
    def test_q3_fully_connected_layer_forward_1(self):
        """Evaluate test case to check fully connected layer forward on random input for Q3"""

        np.random.seed(testing_seed)
        input = np.random.rand(4, 1)

        np.random.seed(0)
        layer = q3.FCLayer(4, 5)
        
        np.random.seed(0)
        gold_layer = q3_sol.FCLayer(4, 5)

        gold_output = gold_layer.forward(input)
        try:
            output = layer.forward(input)
        except Exception as e:
            self.assertEqual(False, True, msg='An error occurred with your FullyConnectedLayer.forward() function: {}'.format(e))

        if output.shape != (5, 1):
            self.assertEqual(False, True, msg='Your output shape is incorrect:\n Expected (5, 1) \n Yours {} on input {}'.format(
                output.shape, input))
            
        for i in range(len(gold_output)):
            for j in range(len(gold_output[i])):
                self.assertEqual(check_output(gold_output[i][j], output[i][j]),
                                 True,
                                 msg='Your output is incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_output, output, input))


    def test_q3_fully_connected_layer_forward_2(self):
        """Evaluate test case to check fully connected layer forward on random input for Q3"""

        np.random.seed(testing_seed)
        input = np.random.rand(10, 1)

        np.random.seed(0)
        layer = q3.FCLayer(10, 8)
        
        np.random.seed(0)
        gold_layer = q3_sol.FCLayer(10, 8)

        gold_output = gold_layer.forward(input)
        try:
            output = layer.forward(input)
        except Exception as e:
            self.assertEqual(False, True, msg='An error occurred with your FullyConnectedLayer.forward() function: {}'.format(e))

        if output.shape != (8, 1):
            self.assertEqual(False, True, msg='Your output shape is incorrect:\n Expected (8, 1) \n Yours {} on input {}'.format(
                output.shape, input))
            
        for i in range(len(gold_output)):
            for j in range(len(gold_output[i])):
                self.assertEqual(check_output(gold_output[i][j], output[i][j]),
                                 True,
                                 msg='Your output is incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_output, output, input))


    def test_q3_fully_connected_layer_backward_1(self):
        """Evaluate test case to check fully connected layer backward on random input for Q3"""

        np.random.seed(testing_seed)
        input = np.random.rand(4, 1)
        output_error = np.random.rand(5, 1)

        np.random.seed(0)
        layer = q3.FCLayer(4, 5)

        np.random.seed(0)
        gold_layer = q3_sol.FCLayer(4, 5)
        gold_layer_2 = q3_sol.FCLayer(4, 5)

        learning_rate = 10

        gold_layer.forward(input)
        gold_input_error = gold_layer.backward(output_error, learning_rate)
        
        gold_layer_2.forward(input)
        gold_input_error_2 = gold_layer_2.backward2(output_error, learning_rate)

        weights = layer.weights
        gold_weights = gold_layer.weights
        gold_weights_2 = gold_layer_2.weights

        bias = layer.bias   
        gold_bias = gold_layer.bias
        gold_bias_2 = gold_layer_2.bias

        try:
            layer.forward(input)
            input_error = layer.backward(output_error, learning_rate)
        except Exception as e:
            self.assertEqual(False, True, msg='An error occurred with your FullyConnectedLayer.backward() function: {}'.format(e))

        if input_error.shape != (4, 1):
            self.assertEqual(False, True, msg='Your input error shape is incorrect:\n Expected (4, 1) \n Yours {} on input {}'.format(
                input_error.shape, input))
            
        for i in range(len(gold_input_error)):
            for j in range(len(gold_input_error[i])):
                self.assertEqual(check_output(gold_input_error[i][j], input_error[i][j]) or check_output(gold_input_error_2[i][j], input_error[i][j]),
                                 True,
                                 msg='Your input_error is incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_input_error, input_error, input))
                
        for i in range(len(gold_weights)):
            for j in range(len(gold_weights[i])):
                self.assertEqual(check_output(gold_weights[i][j], weights[i][j]) or check_output(gold_weights_2[i][j], weights[i][j]),
                                 True,
                                 msg='Your weights are incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_weights, weights, input))
                
        for i in range(len(gold_bias)):
            for j in range(len(gold_bias[i])):
                self.assertEqual(check_output(gold_bias[i][j], bias[i][j]) or check_output(gold_bias_2[i][j], bias[i][j]),
                                 True,
                                 msg='Your bias is incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_bias, bias, input))


    def test_q3_fully_connected_layer_backward_2(self):
        """Evaluate test case to check fully connected layer backward on random input for Q3"""

        np.random.seed(testing_seed)
        input = np.random.rand(8, 1)
        output_error = np.random.rand(10, 1)

        np.random.seed(0)
        layer = q3.FCLayer(8, 10)

        np.random.seed(0)
        gold_layer = q3_sol.FCLayer(8, 10)
        gold_layer_2 = q3_sol.FCLayer(8, 10)

        learning_rate = 1

        gold_layer.forward(input)
        gold_input_error = gold_layer.backward(output_error, learning_rate)
        
        gold_layer_2.forward(input)
        gold_input_error_2 = gold_layer_2.backward2(output_error, learning_rate)

        weights = layer.weights
        gold_weights = gold_layer.weights
        gold_weights_2 = gold_layer_2.weights

        bias = layer.bias   
        gold_bias = gold_layer.bias
        gold_bias_2 = gold_layer_2.bias

        try:
            layer.forward(input)
            input_error = layer.backward(output_error, learning_rate)
        except Exception as e:
            self.assertEqual(False, True, msg='An error occurred with your FullyConnectedLayer.backward() function: {}'.format(e))

        if input_error.shape != (8, 1):
            self.assertEqual(False, True, msg='Your input error shape is incorrect:\n Expected (8, 1) \n Yours {} on input {}'.format(
                input_error.shape, input))
            
        for i in range(len(gold_input_error)):
            for j in range(len(gold_input_error[i])):
                self.assertEqual(check_output(gold_input_error[i][j], input_error[i][j]) or check_output(gold_input_error_2[i][j], input_error[i][j]),
                                 True,
                                 msg='Your input_error is incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_input_error, input_error, input))
                
        for i in range(len(gold_weights)):
            for j in range(len(gold_weights[i])):
                self.assertEqual(check_output(gold_weights[i][j], weights[i][j]) or check_output(gold_weights_2[i][j], weights[i][j]),
                                 True,
                                 msg='Your weights are incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_weights, weights, input))
                
        for i in range(len(gold_bias)):
            for j in range(len(gold_bias[i])):
                self.assertEqual(check_output(gold_bias[i][j], bias[i][j]) or check_output(gold_bias_2[i][j], bias[i][j]),
                                 True,
                                 msg='Your bias is incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_bias, bias, input))


    def test_q3_softmax_layer_forward_1(self):
        """Evaluate test case to check softmax layer forward on random input for Q3"""

        np.random.seed(testing_seed)
        input = np.random.rand(4, 1)

        layer = q3.SoftmaxLayer(4)
        gold_layer = q3_sol.SoftmaxLayer(4)

        gold_output = gold_layer.forward(input)
        try:
            output = layer.forward(input)
        except Exception as e:
            self.assertEqual(False, True, msg='An error occurred with your SoftmaxLayer.forward() function: {}'.format(e))

        if output.shape != (4, 1):
            self.assertEqual(False, True, msg='Your output shape is incorrect:\n Expected (4, 1) \n Yours {} on input {}'.format(
                output.shape, input))
            
        for i in range(len(gold_output)):
            for j in range(len(gold_output[i])):
                self.assertEqual(check_output(gold_output[i][j], output[i][j]),
                                 True,
                                 msg='Your output is incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_output, output, input))


    def test_q3_softmax_layer_forward_2(self):
        """Evaluate test case to check softmax layer forward on random input for Q3"""

        np.random.seed(testing_seed)
        input = np.random.rand(8, 1)

        layer = q3.SoftmaxLayer(8)
        gold_layer = q3_sol.SoftmaxLayer(8)

        gold_output = gold_layer.forward(input)
        try:
            output = layer.forward(input)
        except Exception as e:
            self.assertEqual(False, True, msg='An error occurred with your SoftmaxLayer.forward() function: {}'.format(e))

        if output.shape != (8, 1):
            self.assertEqual(False, True, msg='Your output shape is incorrect:\n Expected (8, 1) \n Yours {} on input {}'.format(
                output.shape, input))
            
        for i in range(len(gold_output)):
            for j in range(len(gold_output[i])):
                self.assertEqual(check_output(gold_output[i][j], output[i][j]),
                                 True,
                                 msg='Your output is incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_output, output, input))


    def test_q3_softmax_layer_backward_1(self):
        """Evaluate test case to check softmax layer backward on random input for Q3"""

        np.random.seed(testing_seed)
        input = np.random.rand(4, 1)
        output_error = np.random.rand(4, 1)
        learning_rate = 0.1

        layer = q3.SoftmaxLayer(4)
        gold_layer = q3_sol.SoftmaxLayer(4)

        gold_layer.forward(input)
        gold_input_error = gold_layer.backward(output_error, learning_rate)

        try:
            layer.forward(input)
            input_error = layer.backward(output_error, learning_rate)
        except Exception as e:
            self.assertEqual(False, True, msg='An error occurred with your SoftmaxLayer.backward() function: {}'.format(e))

        if input_error.shape != (4, 1):
            self.assertEqual(False, True, msg='Your input error shape is incorrect:\n Expected (4, 1) \n Yours {} on input {}'.format(
                input_error.shape, input))
            
        for i in range(len(gold_input_error)):
            for j in range(len(gold_input_error[i])):
                self.assertEqual(check_output(gold_input_error[i][j], input_error[i][j]),
                                 True,
                                 msg='Your output is incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_input_error, input_error, input))
                

    def test_q3_softmax_layer_backward_2(self):
        """Evaluate test case to check softmax layer backward on random input for Q3"""

        np.random.seed(testing_seed)
        input = np.random.rand(8, 1)
        output_error = np.random.rand(8, 1)
        learning_rate = 0.1

        layer = q3.SoftmaxLayer(8)
        gold_layer = q3_sol.SoftmaxLayer(8)

        gold_layer.forward(input)
        gold_input_error = gold_layer.backward(output_error, learning_rate)

        try:
            layer.forward(input)
            input_error = layer.backward(output_error, learning_rate)
        except Exception as e:
            self.assertEqual(False, True, msg='An error occurred with your SoftmaxLayer.backward() function: {}'.format(e))

        if input_error.shape != (8, 1):
            self.assertEqual(False, True, msg='Your input error shape is incorrect:\n Expected (8, 1) \n Yours {} on input {}'.format(
                input_error.shape, input))
            
        for i in range(len(gold_input_error)):
            for j in range(len(gold_input_error[i])):
                self.assertEqual(check_output(gold_input_error[i][j], input_error[i][j]),
                                 True,
                                 msg='Your output is incorrect:\n Expected {} \n Yours {} on input {}'.format(
                                     gold_input_error, input_error, input))
                

    def test_q3_cross_entropy_loss_1(self):
        """Evaluate test case to check cross entropy loss on random input for Q3"""

        np.random.seed(testing_seed)
        y_pred = np.random.rand(10, 1)
        y_true = np.random.randint(0, 10)

        loss = q3.cross_entropy(y_true, y_pred)
        gold_loss = q3_sol.cross_entropy(y_true, y_pred)

        self.assertEqual(check_output(gold_loss, loss),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on y_true {} and y_pred {}'.format(
                             gold_loss, loss, y_true, y_pred))
        

    def test_q3_cross_entropy_loss_2(self):
        """Evaluate test case to check cross entropy loss on random input for Q3"""

        np.random.seed(testing_seed)
        y_pred = np.random.rand(8, 1)
        y_true = np.random.randint(0, 8)

        loss = q3.cross_entropy(y_true, y_pred)
        gold_loss = q3_sol.cross_entropy(y_true, y_pred)

        self.assertEqual(check_output(gold_loss, loss),
                         True,
                         msg='Your output is incorrect:\n Expected {} \n Yours {} on y_true {} and y_pred {}'.format(
                             gold_loss, loss, y_true, y_pred))
        

    def test_q3_cross_entropy_prime_1(self):
        """Evaluate test case to check cross entropy prime on random input for Q3"""

        np.random.seed(testing_seed)
        y_pred = np.random.rand(10, 1)
        y_true = np.random.randint(0, 10)

        loss = q3.cross_entropy_prime(y_true, y_pred)
        gold_loss = q3_sol.cross_entropy_prime(y_true, y_pred)

        if loss.shape != (10, 1):
            self.assertEqual(False, True, msg='Your output shape is incorrect:\n Expected (10, 1) \n Yours {} on y_true {} and y_pred {}'.format(
                loss.shape, y_true, y_pred))
            
        for i in range(len(gold_loss)):
            self.assertEqual(check_output(gold_loss[i], loss[i]),
                             True,
                             msg='Your output is incorrect:\n Expected {} \n Yours {} on y_true {} and y_pred {}'.format(
                                 gold_loss, loss, y_true, y_pred))
            

    def test_q3_cross_entropy_prime_2(self):
        """Evaluate test case to check cross entropy prime on random input for Q3"""

        np.random.seed(testing_seed)
        y_pred = np.random.rand(12, 1)
        y_true = np.random.randint(0, 12)

        loss = q3.cross_entropy_prime(y_true, y_pred)
        gold_loss = q3_sol.cross_entropy_prime(y_true, y_pred)

        if loss.shape != (12, 1):
            self.assertEqual(False, True, msg='Your output shape is incorrect:\n Expected (12, 1) \n Yours {} on y_true {} and y_pred {}'.format(
                loss.shape, y_true, y_pred))
            
        for i in range(len(gold_loss)):
            self.assertEqual(check_output(gold_loss[i], loss[i]),
                             True,
                             msg='Your output is incorrect:\n Expected {} \n Yours {} on y_true {} and y_pred {}'.format(
                                 gold_loss, loss, y_true, y_pred))
            
            
    def test_q3_mnist_validation_accuracy(self):
        """Evaluate public test case to check mnist validation accuracy for Q3"""

        with open(f"./data/mnist_train.pkl", "rb") as file:
            train_dataset = pkl.load(file)

        # self.q3_accuracy_gold = q3_sol.fit(train_dataset[0],train_dataset[1],'mnist')
        try:
            np.random.seed(0)
            self.q3_accuracy_student = q3.fit(train_dataset[0],train_dataset[1],'mnist')
        except Exception as e:
            self.assertEqual(False, True, msg='An error occurred with your q3.fit() function: {}'.format(e))

        if self.q3_accuracy_student < 87.0:
            self.assertEqual(False, True, msg='Your validation accuracy is too less:\n Expected > 87% \n Yours {}'.format(
                self.q3_accuracy_student))


    def test_q3_flowers_validation_accuracy(self):
        """Evaluate public test case to check flowers validation accuracy for Q3"""

        with open(f"./data/flowers_train.pkl", "rb") as file:
            train_dataset = pkl.load(file)

        # self.q3_accuracy_gold = q3_sol.fit(train_dataset[0],train_dataset[1],'flowers')
        try:
            np.random.seed(0)
            self.q3_accuracy_student = q3.fit(train_dataset[0],train_dataset[1],'flowers')
        except Exception as e:
            self.assertEqual(False, True, msg='An error occurred with your q3.fit() function: {}'.format(e))

        if self.q3_accuracy_student < 70.0:
            self.assertEqual(False, True, msg='Your validation accuracy is too less:\n Expected > 70% \n Yours {}'.format(
                self.q3_accuracy_student))
            
if __name__ == '__main__':
    test = TestFunctions()
    sys.stdout = open('results.txt', 'w', buffering=1)

    # # Q1 Tests
    # print("Q1 Tests:")
    # print("=========================================")
    # test.run_test(test.test_q1_gaussian_kernel_1, 2.5)
    # test.run_test(test.test_q1_gaussian_kernel_2, 2.5)
    # test.run_test(test.test_q1_kernel_regression_1, 5)
    # test.run_test(test.test_q1_kernel_regression_2, 5)
    # test.run_test(test.test_q1_optimal_bandwidth_1, 2.5)
    # test.run_test(test.test_q1_optimal_bandwidth_2, 2.5)

    # Q2 Tests
    print("Q2 Tests:")
    print("=========================================")
    # test.run_test(test.test_q2_polynomial_kernel_1, 2)
    # test.run_test(test.test_q2_polynomial_kernel_2, 2)
    # test.run_test(test.test_q2_sigmoid_1, 1.5)
    # test.run_test(test.test_q2_sigmoid_2, 1.5)
    # test.run_test(test.test_q2_cost_function_1, 5)
    # test.run_test(test.test_q2_cost_function_2, 5)
    test.run_test(test.test_q2_update_parameters_1, 5)
    test.run_test(test.test_q2_update_parameters_2, 5)
    test.run_test(test.test_q2_accuracy_on_test_set, 3)

    # # Q3 Tests
    # print("Q3 Tests:")
    # print("=========================================")
    # test.run_test(test.test_q3_fully_connected_layer_forward_1, 2.5)
    # test.run_test(test.test_q3_fully_connected_layer_forward_2, 2.5)
    # test.run_test(test.test_q3_fully_connected_layer_backward_1, 5)
    # test.run_test(test.test_q3_fully_connected_layer_backward_2, 5)
    # test.run_test(test.test_q3_softmax_layer_forward_1, 2.5)
    # test.run_test(test.test_q3_softmax_layer_forward_2, 2.5)
    # test.run_test(test.test_q3_softmax_layer_backward_1, 5)
    # test.run_test(test.test_q3_softmax_layer_backward_2, 5)
    # test.run_test(test.test_q3_cross_entropy_loss_1, 2.5)
    # test.run_test(test.test_q3_cross_entropy_loss_2, 2.5)
    # test.run_test(test.test_q3_cross_entropy_prime_1, 2.5)
    # test.run_test(test.test_q3_cross_entropy_prime_2, 2.5)

    # Comment the 2 tests below if you don't want to test the accuracy on mnist and flowers dataset
    # test.run_test(test.test_q3_mnist_validation_accuracy, 5)
    # test.run_test(test.test_q3_flowers_validation_accuracy, 5)

    print("Total Marks:", test.total_marks)