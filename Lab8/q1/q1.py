import os
import numpy as np
import scipy as sp

# Set the seed for reproducibility
np.random.seed(0)

class DataGenerator:
    """
    A class for generating input and output examples for a character-level language model.
    """
    
    def __init__(self, path):
        """
        Initializes a DataGenerator object.

        Args:
            path (str): The path to the text file containing the training data.
        """
        self.path = path
        
        # Read in data from file and convert to lowercase
        with open(path) as f:
            data = f.read().lower()
        
        # Create list of unique characters in the data
        self.chars = list(set(data))

        # Create dictionaries mapping characters to and from their index in the list of unique characters
        self.char_to_idx = {ch: i for (i, ch) in enumerate(self.chars)}
        self.idx_to_char = {i: ch for (i, ch) in enumerate(self.chars)}
        
        # Set the size of the vocabulary (i.e. number of unique characters)
        self.vocab_size = len(self.chars)
        
        # Read in examples from file and convert to lowercase, removing leading/trailing white space
        with open(path) as f:
            examples = f.readlines()
        self.examples = [x.lower().strip() for x in examples]

 
    def generate_example(self, idx):
        """
        Generates an input/output example for the language model based on the given index.

        Args:
            idx (int): The index of the example to generate.

        Returns:
            A tuple containing the input and output arrays for the example.
        """
        example_chars = self.examples[idx]
        
        # Convert the characters in the example to their corresponding indices in the list of unique characters
        example_char_idx = [self.char_to_idx[char] for char in example_chars]
        
        # Add newline character as the first character in the input array, and as the last character in the output array
        X = [self.char_to_idx['\n']] + example_char_idx
        Y = example_char_idx + [self.char_to_idx['\n']]
        
        return np.array(X), np.array(Y)


class RNN:
    """
    A class used to represent a Recurrent Neural Network (RNN).

    Attributes
    ----------
    hidden_size : int
        The number of hidden units in the RNN.
    vocab_size : int
        The size of the vocabulary used by the RNN.
    sequence_length : int
        The length of the input sequences fed to the RNN.
    learning_rate : float
        The learning rate used during training.
    is_initialized : bool
        Indicates whether the AdamW parameters has been initialized.

    Methods
    -------
    __init__(hidden_size, vocab_size, sequence_length, learning_rate)
        Initializes an instance of the RNN class.
    
    forward(self, X, a_prev)
     Computes the forward pass of the RNN.
     
    softmax(self, x)
       Computes the softmax activation function for a given input array. 
       
    backward(self,x, a, y_preds, targets)    
        Implements the backward pass of the RNN.
        
    loss(self, y_preds, targets)
     Computes the cross-entropy loss for a given sequence of predicted probabilities and true targets. 
     
    adamw(self, beta1=0.9, beta2=0.999, epsilon=1e-8, L2_reg=1e-4)
       Updates the RNN's parameters using the AdamW optimization algorithm.
       
    train(self, generated_names=5)
       Trains the RNN on a dataset using backpropagation through time (BPTT).   
       
    predict(self, start)
        Generates a sequence of characters using the trained self, starting from the given start sequence.
        The generated sequence may contain a maximum of 50 characters or a newline character.

    """

    def __init__(self, hidden_size, data_generator, sequence_length, learning_rate):
        """
        Initializes an instance of the RNN class.

        Parameters
        ----------
        hidden_size : int
            The number of hidden units in the RNN.
        vocab_size : int
            The size of the vocabulary used by the RNN.
        sequence_length : int
            The length of the input sequences fed to the RNN.
        learning_rate : float
            The learning rate used during training.
        """

        # hyper parameters
        self.hidden_size = hidden_size
        self.data_generator = data_generator
        self.vocab_size = self.data_generator.vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.X = None

        # model parameters
        self.Wax = np.random.uniform(-np.sqrt(1. / self.vocab_size), np.sqrt(1. / self.vocab_size), (hidden_size, self.vocab_size))
        self.Waa = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (hidden_size, hidden_size))
        self.Wya = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (self.vocab_size, hidden_size))
        self.ba = np.zeros((hidden_size, 1))  
        self.by = np.zeros((self.vocab_size, 1))
        
        # Initialize gradients
        self.dWax, self.dWaa, self.dWya = np.zeros_like(self.Wax), np.zeros_like(self.Waa), np.zeros_like(self.Wya)
        self.dba, self.dby = np.zeros_like(self.ba), np.zeros_like(self.by)
        
        # parameter update with AdamW
        self.mWax = np.zeros_like(self.Wax)
        self.vWax = np.zeros_like(self.Wax)
        self.mWaa = np.zeros_like(self.Waa)
        self.vWaa = np.zeros_like(self.Waa)
        self.mWya = np.zeros_like(self.Wya)
        self.vWya = np.zeros_like(self.Wya)
        self.mba = np.zeros_like(self.ba)
        self.vba = np.zeros_like(self.ba)
        self.mby = np.zeros_like(self.by)
        self.vby = np.zeros_like(self.by)

    def softmax(self, x):
        """
        Computes the softmax activation function for a given input array.

        Parameters:
            x (ndarray): Input array.

        Returns:
            ndarray: Array of the same shape as `x`, containing the softmax activation values.
        """
        ### TASK 1.1
        ### STUDENT CODE STARTS HERE
        p = None

        # shift the input to prevent overflow when computing the exponentials
        x -= np.max(x)
        # compute the exponentials of the shifted input
        p = np.exp(x)
        # normalize the exponentials by dividing by their sum
        p /= np.sum(p)
        ### STUDENT CODE ENDS HERE
        return p

    def forward(self, X, a_prev):
        """
        Compute the forward pass of the RNN.

        Parameters:
        X (ndarray): Input data of shape (seq_length, vocab_size)
        a_prev (ndarray): Activation of the previous time step of shape (hidden_size, 1)

        Returns:
        x (dict): Dictionary of input data of shape (seq_length, vocab_size, 1), with keys from 0 to seq_length-1
        a (dict): Dictionary of hidden activations for each time step, with keys from 0 to seq_length-1
        y_pred (dict): Dictionary of output probabilities for each time step, with keys from 0 to seq_length-1
        """
        # Initialize dictionaries to store activations and output probabilities.
        x, a, y_pred = {}, {}, {}

        # Store the input data in the class variable for later use in the backward pass.
        self.X = X

        # Set the initial activation to the previous activation.
        a[-1] = np.copy(a_prev)


        ### TASK 1.3
        ### STUDENT CODE STARTS HERE
        # iterate over each time step in the input sequence
        for i in range(np.shape(X)[0]):
            # get the input at the current time step
            # in_x = X[i]
            xt = np.zeros((self.vocab_size, 1))
            xt[X[i]] = 1
            x[i] = xt
            # compute the hidden activation at the current time step
            intermediate = np.dot(self.Waa, a[i-1]) + np.dot(self.Wax, x[i]) + self.ba
            a[i] = np.tanh(intermediate)
            # compute the output probabilities at the current time step
            y_pred[i] = self.softmax(np.dot(self.Wya, a[i]) + self.by)
            # add an extra dimension to X in axis 1 to make it compatible with the shape of the input to the backward pass
            # x[i] = np.expand_dims(X, axis=1)
            # a[i] = a_prev
            # y_pred[i] = y 
        ### STUDENT CODE ENDS HERE
        self.X = np.expand_dims(X, axis=1)
        # return the input, hidden activations, and output probabilities at each time step
        return x, a, y_pred 
    
    def backward(self,x, a, y_preds, targets):
        """
        Implement the backward pass of the RNN.

        Args:
        x -- (dict) of input characters (as one-hot encoding vectors) for each time-step, shape (vocab_size, sequence_length)
        a -- (dict) of hidden state vectors for each time-step, shape (hidden_size, sequence_length)
        y_preds -- (dict) of output probability vectors (after softmax) for each time-step, shape (vocab_size, sequence_length)
        targets -- (list) of integer target characters (indices of characters in the vocabulary) for each time-step, shape (1, sequence_length)

        Returns:
        None

        """
        # Initialize derivative of hidden state for the last time-step
        da_next = np.zeros_like(a[0])

        ### TASK 1.3
        ### STUDENT CODE STARTS HERE
        # Loop through the input sequence backwards
        for i in reversed(range(len(x))):
            # Calculate derivative of output probability vector
            # self.dy = np.zeros_like(y_preds[0])
            # self.dy[targets[i]] = -1/y_preds[i][targets[i]][0]
            self.dy = np.copy(y_preds[i])
            self.dy[targets[i]] -= 1
            # print(np.shape(self.dy))
            # dL_dyt[targets[i]] = -1/(y_preds[targets[i]][i])
            # Calculate derivative of hidden state, da, using the chain rule
            # print(np.shape(da_next))
            self.da = np.dot(self.Wya.T, self.dy) + np.dot(self.Waa.T, da_next)
            # Calculate derivative of hidden state before activation function
            # print(np.shape((1 - (np.tanh(a[i])*np.tanh(a[i])))*self.da))
            self.dh =(1 - a[i]*a[i])*self.da
            # Calculate gradients self.dba, self.dWax, self.dWaa
            self.dba += self.dh
            self.dWaa += np.dot(self.dh, a[i-1].T)
            # print(np.shape(self.dWya))
            self.dWax += np.dot(self.dh, x[i].T)
            # Update derivative of hidden state for the next iteration with da_unactivated
            da_next = self.dh
            # Calculate gradient for output weight matrix, self.dWya
            self.dWya += np.dot(self.dy, a[i].T)
            self.dby += self.dy
            # clip gradients to avoid exploding gradients, just uncomment the line below
            for grad in [self.dWax, self.dWaa, self.dWya, self.dba, self.dby]:
                np.clip(grad, -1, 1, out=grad)
        
        ### STUDENT CODE ENDS HERE
                
 
    def loss(self, y_preds, targets):
        """
        Computes the cross-entropy loss for a given sequence of predicted probabilities and true targets.

        Parameters:
            y_preds (ndarray): Array of shape (sequence_length, vocab_size) containing the predicted probabilities for each time step.
            targets (ndarray): Array of shape (sequence_length, 1) containing the true targets for each time step.

        Returns:
            float: Cross-entropy loss.
        """
        loss = None

        ### TASK 1.2
        ### STUDENT CODE STARTS HERE
        # calculate cross-entropy loss
        # one_hot_targets = np.eye(y_preds.shape[1])[targets.flatten()]

        # # Vectorized calculation of cross-entropy loss
        # loss = -np.sum(np.log(np.dot(y_preds, one_hot_targets.T)))
        # loss = -np.sum(np.log(y_preds[np.arange(len(y_preds)), targets.flatten()]))
        loss = 0
        # # print(len(y_preds))
        for i in range(len(y_preds)):
            loss -= np.log(y_preds[i][targets[i]])
        loss=float(loss)
        ### STUDENT CODE ENDS HERE
        
        return loss
    
    def adamw(self, beta1=0.9, beta2=0.999, epsilon=1e-8, L2_reg=1e-4):
        """
        Updates the RNN's parameters using the AdamW optimization algorithm.
        """
        # AdamW update for Wax
        self.mWax = beta1 * self.mWax + (1 - beta1) * self.dWax
        self.vWax = beta2 * self.vWax + (1 - beta2) * np.square(self.dWax)
        m_hat = self.mWax / (1 - beta1)
        v_hat = self.vWax / (1 - beta2)
        self.Wax -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Wax)

        # AdamW update for Waa
        self.mWaa = beta1 * self.mWaa + (1 - beta1) * self.dWaa
        self.vWaa = beta2 * self.vWaa + (1 - beta2) * np.square(self.dWaa)
        m_hat = self.mWaa / (1 - beta1)
        v_hat = self.vWaa / (1 - beta2)
        self.Waa -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Waa)

        # AdamW update for Wya
        self.mWya = beta1 * self.mWya + (1 - beta1) * self.dWya
        self.vWya = beta2 * self.vWya + (1 - beta2) * np.square(self.dWya)
        m_hat = self.mWya / (1 - beta1)
        v_hat = self.vWya / (1 - beta2)
        self.Wya -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Wya)

        # AdamW update for ba
        self.mba = beta1 * self.mba + (1 - beta1) * self.dba
        self.vba = beta2 * self.vba + (1 - beta2) * np.square(self.dba)
        m_hat = self.mba / (1 - beta1)
        v_hat = self.vba / (1 - beta2)
        self.ba -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.ba)

        # AdamW update for by
        self.mby = beta1 * self.mby + (1 - beta1) * self.dby
        self.vby = beta2 * self.vby + (1 - beta2) * np.square(self.dby)
    
    def sample(self):
        """
        Sample a sequence of characters from the RNN.

        Args:
            None

        Returns:
            list: A list of integers representing the generated sequence.
        """
        # initialize input and hidden state
        x = np.zeros((self.vocab_size, 1))
        a_prev = np.zeros((self.hidden_size, 1))

        # create an empty list to store the generated character indices
        indices = []

        # idx is a flag to detect a newline character, initialize it to -1
        idx = -1

        # generate sequence of characters
        counter = 0
        max_chars = 50 # maximum number of characters to generate
        newline_character = self.data_generator.char_to_idx['\n'] # the newline character

        while (idx != newline_character and counter != max_chars):
            # compute the hidden state
            a = np.tanh(np.dot(self.Wax, x) + np.dot(self.Waa, a_prev) + self.ba)

            # compute the output probabilities
            y = self.softmax(np.dot(self.Wya, a) + self.by)

            # sample the next character from the output probabilities
            idx = np.random.choice(list(range(self.vocab_size)), p=y.ravel())

            # set the input for the next time step
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

            # store the sampled character index in the list
            indices.append(idx)

            # update the previous hidden state
            a_prev = a

            # increment the counter
            counter += 1

        # return the list of sampled character indices
        return indices

        
    def train(self, generated_names=5, threshold=5, print_flag=True):
        """
        Train the RNN on a dataset using backpropagation through time (BPTT).

        Args:
        - generated_names: an integer indicating how many example names to generate during training.

        Returns:
        - None
        """

        iter_num = 0
        threshold = threshold # stopping criterion for training
        smooth_loss = -np.log(1.0 / self.data_generator.vocab_size) * self.sequence_length  # initialize loss
        loss_vector = []

        while (smooth_loss > threshold):
            a_prev = np.zeros((self.hidden_size, 1))
            idx = iter_num % self.vocab_size
            # get a batch of inputs and targets
            inputs, targets = self.data_generator.generate_example(idx)

            # forward pass
            x, a, y_pred  = self.forward(inputs, a_prev)

            # backward pass
            self.backward(x, a, y_pred, targets)

            # calculate and update loss
            loss = self.loss(y_pred, targets)
            self.adamw()
            smooth_loss = smooth_loss * 0.999 + loss * 0.001


            # update previous hidden state for the next batch
            a_prev = a[len(self.X) - 1]
            # print progress every 500 iterations
            if iter_num % 500 == 0:
                if print_flag:
                    print("\n\niter :%d, loss:%f\n" % (iter_num, smooth_loss))
                loss_vector.append(smooth_loss)
                for i in range(generated_names):
                    sample_idx = self.sample()
                    txt = ''.join(self.data_generator.idx_to_char[idx] for idx in sample_idx)
                    txt = txt.title()  # capitalize first character 
                    # print ('%s' % (txt, ), end='')
            iter_num += 1
        return loss_vector
    
    def predict(self, start):
        """
        Generate a sequence of characters using the trained self, starting from the given start sequence.
        The generated sequence may contain a maximum of 50 characters or a newline character.

        Args:
        - start: a string containing the start sequence

        Returns:
        - txt: a string containing the generated sequence
        """

        # Initialize input vector and previous hidden state
        x = np.zeros((self.vocab_size, 1))
        a_prev = np.zeros((self.hidden_size, 1))

        # Convert start sequence to indices
        chars = [ch for ch in start]
        idxes = []
        for i in range(len(chars)):
            idx = self.data_generator.char_to_idx[chars[i]]
            x[idx] = 1
            idxes.append(idx)

        # Generate sequence
        max_chars = 50  # maximum number of characters to generate
        newline_character = self.data_generator.char_to_idx['\n']  # the newline character
        counter = 0

        # Run a while loop to generate the sequence until a newline character is generated or the maximum number of characters is reached
        while (counter != max_chars):
            # Compute next hidden state and predicted character, using the forward pass of the RNN
            a = np.tanh(np.dot(self.Wax, x) + np.dot(self.Waa, a_prev) + self.ba)
            y = self.softmax(np.dot(self.Wya, a) + self.by)

            # Sample the next character from the output probabilities
            idx = np.random.choice(list(range(self.vocab_size)), p=y.ravel())

            # Store the sampled character index in the list
            idxes.append(idx)

            # Set the input for the next time step
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

            # Update the previous hidden state
            a_prev = a

            # Increment the counter
            counter += 1

            # Break if newline character is generated
            if idx == newline_character:
                break

        # Convert indices to characters and concatenate into a string
        txt = ''.join(self.data_generator.idx_to_char[i] for i in idxes)

        # Remove newline character if it exists at the end of the generated sequence
        if txt[-1] == '\n':
            txt = txt[:-1]

        return txt

if __name__ == "__main__":
    data_generator = DataGenerator('dino.txt')
    print(data_generator.generate_example(0))
    rnn = RNN(hidden_size=200,data_generator=data_generator, sequence_length=25, learning_rate=1e-3)
    rnn.train()

    print(rnn.predict("meo"))
    print(rnn.predict("a"))
