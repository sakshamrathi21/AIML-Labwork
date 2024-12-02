import numpy as np
import pickle as pkl
import argparse


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
    max_X = np.max(X, axis=0)
    min_X = np.min(X, axis=0)
    X_transformed = (X-min_X)/(np.maximum(max_X-min_X,1e-6))

    assert X_transformed.shape == X.shape

    num_samples = len(X)
    indices = np.arange(num_samples)
    num_train_samples = int(np.floor(num_samples * train_ratio))
    train_indices = np.random.choice(indices, num_train_samples, replace=False)
    val_indices = list(set(indices) - set(train_indices))
    X_train, Y_train, X_val, Y_val = X_transformed[train_indices], Y[train_indices], X_transformed[val_indices], Y[val_indices]
  
    return X_train, Y_train, X_val, Y_val

def cross_entropy(y_true, y_pred):
    '''
    Cross entropy loss 
    Args:
        y_true :  Ground truth labels, integer in [0, n_classes-1]
        y_pred :  Predicted labels, numpy array of shape (n_classes, 1)
    Returns:
       loss : float
    '''
    loss = None

    ## TODO
    # print(np.shape(y_pred))
    loss = -np.log(y_pred[y_true][0])
    # print(np.shape(loss))
    # loss = -np.dot(y_true, np.log(y_pred))
    ## END TODO
    
    return loss

def cross_entropy_prime(y_true, y_pred):
    '''
    Implements derivative of cross entropy function, for the backward pass
    Args:
        y_true :  Ground truth labels, integer in [0, n_classes-1]
        y_pred :  Predicted labels, numpy array of shape (n_classes, 1)
    Returns:
        Numpy array after applying derivative of cross entropy function of shape (n_classes, 1)
    '''
    output = None
    
    ## TODO
    output = np.zeros_like(y_pred)
    output[y_true][0] = -(1/y_pred[y_true][0])
    ## END TODO

    return output

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

        ## TODO
        self.output = np.dot(np.transpose(self.weights),self.input)+self.bias
        ## END TODO

        return self.output

    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a fully connected network and update self.weights and self.bias 
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          input_error : Numpy array resulting from the backward pass
        '''
        input_error = None

        ## TODO
        # print(np.shape(self.input), np.shape(output_error), np.shape(self.weights))
        input_error = np.dot(self.weights, output_error)
        self.weights -= learning_rate*(np.dot(self.input, np.transpose(output_error)))
        self.bias -= learning_rate*output_error
        
        ## END TODO

        return input_error
    

class SoftmaxLayer:
    '''
      Implements a Softmax layer which applies softmax function on the inputs. 
    '''
    def __init__(self, input_size):
        self.input_size = input_size
    
    def forward(self, input):
        '''
        Applies the softmax function 
        Args:
           input : numpy array of shape (n_classes, 1) on which softmax function is to be applied 

        Returns:
           output numpy array output of shape (n_classes, 1) from the softmax function
        '''
        output = None

        ## TODO
        output = np.exp(input)
        sum_exp = np.sum(output)
        output/=sum_exp
        #self.output, # END TODO
    
        self.output = output
        return output
        
    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a Softmax layer
        Args:
          output_error :  numpy array of shape (n_classes, 1)
          learning_rate : float
self.output, 
        Returns:
          input_error : Numpy array of shape (n_classes, 1) resulting from the backward pass
        '''

        input_error = None

        ## TODO
        input_error = np.zeros_like(output_error)
        # print("hello")
        # print(np.shape(self.output), np.shape(output_error))
        sum_used_later = np.dot(np.transpose(output_error), self.output)
        # print(sum_used_later)
        for i in range(np.shape(output_error)[0]):
            input_error[i][0] = self.output[i][0]*(output_error[i][0]-sum_used_later)
        ## END TODO

        return input_error
    


def fit(X_train, Y_train, dataset_name):

    '''
    Create and trains a feedforward network

    Args:
        X_train -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.
        Y_train -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.
        dataset_name -- name of the dataset (flowers or mnist)
    
    '''
    
    if dataset_name == 'mnist':
        X_train = X_train[::2]
        Y_train = Y_train[::2]

    print(f"train_x -- {X_train.shape}; train_y -- {Y_train.shape}")
    
    X_train, Y_train, X_val, Y_val = split_data(X_train, Y_train)

    # This network below would work for mnist and flowers dataset.

    if dataset_name == 'mnist':
        network = [
            FCLayer(28 * 28, 256),
            FCLayer(256, 10),
            SoftmaxLayer(10)
        ] # This creates the feed forward network
        epochs = 10
        learning_rate = 0.001

    elif dataset_name == 'flowers':
        network = [
            FCLayer(2048, 256),
            FCLayer(256, 5),
            SoftmaxLayer(5)
        ] # This creates the feed forward network
        epochs = 10
        learning_rate = 0.001

    final_acc_val = 0

    for epoch in range(epochs):
        error = 0
        acc_train = 0
        acc_val = 0

        for x, y_true in zip(X_train, Y_train):
            x = x.flatten().reshape(-1,1)
            y_true = int(y_true)

            # forward pass
            output = x
            for layer in network:
                output = layer.forward(output)
            
            # error (for display purposes only)
            error += cross_entropy(y_true, output)

            acc_train += np.sum(y_true == np.argmax(output))

            # backward pass
            output_error = cross_entropy_prime(y_true, output)
            for layer in reversed(network):
                output_error = layer.backward(output_error, learning_rate)

        for x, y_true in zip(X_val, Y_val):
            x = x.flatten().reshape(-1,1)
            y_true = int(y_true)
            # forward pass
            output = x
            for layer in network:
                output = layer.forward(output)

            acc_val += np.sum(y_true == np.argmax(output))

        
        error /= len(X_train)
        print(f'-------------------------------EPOCH {epoch+1}-----------------------------------')
        print('%d/%d, error=%f' % (epoch + 1, epochs, error))
        print('training accuracy', round(acc_train/len(X_train)*100, 2))
        print('validation accuracy', round(acc_val/len(X_val)*100, 2))

        final_acc_val = acc_val/len(X_val)*100

    return final_acc_val
    
    
"""
Loading data and training models
"""
if __name__ == "__main__":    
    np.random.seed(0)

    parser = argparse.ArgumentParser(description='Train and test a feed forward network')
    parser.add_argument('--dataset', type=str, required=True, help='dataset to train/test on', choices=['mnist', 'flowers'])

    args = parser.parse_args()

    dataset = args.dataset

    with open(f"./data/{dataset}_train.pkl", "rb") as file:
        train_dataset = pkl.load(file)
    
    fit(train_dataset[0],train_dataset[1],dataset)