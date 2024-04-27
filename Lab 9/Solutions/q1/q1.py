import numpy as np
import pickle as pkl

def split_data(X, Y, train_ratio=0.8):
    '''
    Split data into train and validation sets
    The first floor(train_ratio * n_sample) samples form the train set
    and the remaining the validation set

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    train_ratio - fraction of samples to be used as training data

    Returns:
    X_train, Y_train, X_val, Y_val
    '''
    num_samples = len(X)
    indices = np.arange(num_samples)
    num_train_samples = int(np.floor(num_samples * train_ratio))
    train_indices = np.random.choice(indices, num_train_samples, replace=False)
    val_indices = list(set(indices) - set(train_indices))
    X_train, Y_train, X_val, Y_val = X[train_indices], Y[train_indices], X[val_indices], Y[val_indices]
  
    return X_train, Y_train, X_val, Y_val

# Convolutional Layer
class ConvolutionLayer:
    '''
    Implements a Convolutional Layer
    '''
    def __init__(self, num_filters, filter_size):
        '''
        Args:
            num_filters : Number of filters, int
            filter_size : Size of the filter, int
        '''
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(num_filters, filter_size, filter_size)/(filter_size * filter_size) # normalize the values
        self.bias = np.zeros(num_filters)

    def forward(self, input):
        '''
        Args:
            input : 2D numpy array of shape (height, width)

        Returns:
            output : 3D numpy array of shape (num_filters, height - filter_size + 1, width - filter_size + 1)

        '''
        output = None
        self.input = input
        height, width = input.shape

        ## TODO

        output = np.zeros((self.num_filters, height - self.filter_size + 1, width - self.filter_size + 1))

        for filter_num in range(self.num_filters):
            for x in range(height - self.filter_size + 1):
                for y in range(width - self.filter_size + 1):
                    output[filter_num, x, y] = np.sum(input[x:x+self.filter_size, y:y+self.filter_size]*self.conv_filter[filter_num, :, :])
                    output[filter_num, x, y] = output[filter_num, x, y] + self.bias[filter_num]

        ## END TODO

        assert output.shape == (self.num_filters, height - self.filter_size + 1, width - self.filter_size + 1)

        return output
    
    def backward(self, output_error, learning_rate):
        '''
        Args:
            output_error : 3D numpy array of shape (num_filters, height - filter_size + 1, width - filter_size + 1)
            learning_rate : float
        
        Returns:
            input_error : 2D numpy array of shape (height, width)
        '''
        input_error = None

        ## TODO

        input_error = np.zeros(self.input.shape)

        dLdK_params = np.zeros(self.conv_filter.shape)
        for filter_num in range(self.num_filters):
            for i in range(self.filter_size):
                for j in range(self.filter_size):
                    dLdK_params[filter_num, i, j] = np.sum(output_error[filter_num, :, :]*self.input[i:i+output_error.shape[1], j:j+output_error.shape[2]])

        for filter_num in range(self.num_filters):
            self.bias[filter_num] -= learning_rate * np.sum(output_error[filter_num, :, :])

        # Saksham
        output_error = np.pad(output_error, ((0, 0), (self.filter_size-1, self.filter_size-1), (self.filter_size-1, self.filter_size-1)), 'constant')
        # Rathi

        for filter_num in range(self.num_filters):
            for i in range(self.input.shape[0]):
                for j in range(self.input.shape[1]):
                    input_error[i, j] += np.sum(output_error[filter_num, i:i+self.filter_size, j:j+self.filter_size]*self.conv_filter[filter_num, ::-1, ::-1])

        self.conv_filter -= learning_rate * dLdK_params

        ## END TODO

        assert input_error.shape == self.input.shape

        return input_error
    

class ReLULayer:
    '''
    Implements a ReLU Layer
    '''
    def __init__(self):
        pass

    def forward(self, input):
        '''
        Args:
            input : numpy array of any shape

        Returns:
            numpy array of the same shape as input
        '''
        output = None
        self.input = input

        ## TODO

        output = np.maximum(input, 0)

        ## END TODO

        assert output.shape == input.shape

        return output

    def backward(self, output_error, learning_rate):
        '''
        Args:
            output_error : numpy array of any shape
            learning_rate : float

        Returns:
            input_error : Numpy array resulting from the backward pass of the same shape as output_error
        '''
        input_error = None

        ## TODO

        input_error = output_error * (self.input > 0)

        ## END TODO

        assert input_error.shape == output_error.shape

        return input_error

# Max Pooling Layer
class MaxPoolingLayer:
    '''
    Implements a Max Pooling Layer
    '''
    def __init__(self, filter_size):
        '''
        Args:
            filter_size : int
        '''
        self.filter_size = filter_size

    def forward(self, input):
        '''
        Args:
            input : 3D numpy array of shape (num_filters, height, width)

        Returns:
            output : 3D numpy array of shape (num_filters, height // filter_size, width // filter_size)
            Assume filter_size is an integer and height and width are divisible by filter_size
        '''
        output = None
        self.input = input
        num_filters, height, width = input.shape

        ## TODO

        output = np.zeros((num_filters, height // self.filter_size, width // self.filter_size))

        for filter_num in range(num_filters):
            for r in np.arange(0, height, self.filter_size):
                for c in np.arange(0, width, self.filter_size):
                    output[filter_num, r//self.filter_size, c//self.filter_size] = np.max(input[filter_num, r:r+self.filter_size, c:c+self.filter_size])

        ## END TODO
                    
        assert output.shape == (num_filters, height // self.filter_size, width // self.filter_size)
            
        return output
    
    def backward(self, output_error, learning_rate):
        '''
        Args:
            output_error : 3D numpy array of shape (num_filters, height // filter_size, width // filter_size)
            learning_rate : float
        
        Returns:
            input_error : 3D numpy array of shape (num_filters, height, width)
        '''
        input_error = None

        ## TODO

        num_filters, height, width = self.input.shape
        input_error = np.zeros(self.input.shape)

        for filter_num in range(num_filters):
            for r in np.arange(0, height, self.filter_size):
                for c in np.arange(0, width, self.filter_size):
                    # Saksham
                    (a, b) = np.unravel_index(np.argmax(self.input[filter_num, r:r+self.filter_size, c:c+self.filter_size]), (self.filter_size, self.filter_size))
                    # Rathi
                    input_error[filter_num, r+a, c+b] = output_error[filter_num, r//self.filter_size, c//self.filter_size]

        ## END TODO
                    
        assert input_error.shape == self.input.shape
                    
        return input_error
    
    
class FlattenLayer:
    '''
    Implements a Flatten Layer
    '''
    def __init__(self):
        pass

    def forward(self, input):
        '''
        Args:
            input : numpy array of shape (num_filters, height, width)
        
        Returns:
            numpy array of shape (num_filters*height*width, 1)
        '''

        output = None
        self.input_shape = input.shape

        ## TODO

        output = input.flatten().reshape(-1, 1)

        ## END TODO

        assert output.shape == (input.shape[0]*input.shape[1]*input.shape[2], 1)

        return output

    def backward(self, output_error, learning_rate):
        '''
        Args:
            output_error : numpy array of shape (num_filters*height*width, 1)
            learning_rate : float
        
        Returns:
            input_error : numpy array of shape (num_filters, height, width)
        '''
        input_error = None

        ## TODO

        input_error = output_error.reshape(self.input_shape)

        ## END TODO

        assert input_error.shape == self.input_shape

        return input_error
    

class FCLayer:
    '''
    Implements a fully connected layer  
    '''
    def __init__(self, input_size, output_size):
        '''
        Args:
         input_size : Input shape, int
         output_size: Output shape, int 
        '''
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.random((input_size, output_size)) - 0.5
        self.bias = np.random.random((output_size, 1)) - 0.5

    def forward(self, input):
        '''
        Performs a forward pass of a fully connected network
        Args:
          input : training data, numpy array of shape (self.input_size, 1)

        Returns:
           numpy array of shape (self.output_size, 1)
        '''
        self.output = None
        self.input = input

        self.output = np.dot(self.weights.T, self.input) + self.bias

        assert self.output.shape == (self.output_size, 1)

        return self.output

    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a fully connected network and updates the weights and bias 
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          input_error : Numpy array resulting from the backward pass
        '''
        input_error = None

        input_error = np.dot(self.weights, output_error)
        weights_error = np.dot(self.input, output_error.T)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        assert input_error.shape == self.input.shape

        return input_error
    

class SoftmaxLayer:
    '''
      Implements a Softmax layer which applies softmax function on the inputs. 
    '''
    def __init__(self):
        pass
    
    def forward(self, input):
        '''
        Applies the softmax function 
        Args:
           input : numpy array of shape (n_classes, 1) on which softmax function is to be applied 

        Returns:
           output numpy array output of shape (n_classes, 1) from the softmax function
        '''
        output = None

        output = np.exp(input-np.max(input, axis=0))
        output = output/np.sum(output)

        self.output = output

        assert output.shape == input.shape

        return output
        
    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a Softmax layer
        Args:
          output_error :  numpy array of shape (n_classes, 1)
          learning_rate : float

        Returns:
          input_error : Numpy array of shape (n_classes, 1) resulting from the backward pass
        '''

        input_error = None

        input_error = self.output*output_error - self.output.T@output_error * self.output

        assert input_error.shape == output_error.shape

        return input_error
    
    
def train(image, label, layers, lr=.005):
    # Forward
    out = (image / 255) - 0.5
    for layer in layers:
        out = layer.forward(out)

    # Calculate initial difference
    output_error = np.zeros(out.shape)
    output_error[label] = -1/out[label]

    # Backprop
    for layer in reversed(layers):
        output_error = layer.backward(output_error, lr)

    # Calculate loss
    loss = -np.sum(np.log(out[label]))
    acc = 1 if np.argmax(out) == label else 0

    return loss, acc


if __name__ == "__main__":

    np.random.seed(40)

    # Load the MNIST dataset
    with open(f"./data/mnist.pkl", "rb") as file:
        dataset = pkl.load(file)

    train_images, train_labels, X_val, Y_val = split_data(dataset[0], dataset[1])

    # Initialize the layers
    conv = ConvolutionLayer(8, 3)  # 8 filters, 3x3
    relu = ReLULayer()
    pool = MaxPoolingLayer(2)  # 2x2
    flatten = FlattenLayer()
    fc = FCLayer(13 * 13 * 8, 10)  # input size, number of classes
    softmax = SoftmaxLayer() 

    layers = [conv, relu, pool, flatten, fc, softmax]

    # Train the model
    for epoch in range(3):
        print('--- Epoch %d ---' % (epoch + 1))

        # Shuffle the training data
        shuffle_data = np.random.permutation(len(train_images))
        train_images = train_images[shuffle_data]
        train_labels = train_labels[shuffle_data]

        # Train!
        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            if i % 100 == 99:
                print(
                    '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                    (i + 1, loss / 100, num_correct)
                )
                loss = 0
                num_correct = 0

            l, acc = train(im, label, layers)
            loss += l
            num_correct += acc