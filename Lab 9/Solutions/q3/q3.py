
import numpy as np
from sklearn import svm

'''
Look at the sample data given for part 2 in the PDF. 
Write an appropriate kernel which returns a scalar value 
determining the relation of input in higher dimension space.
Try to find generalized solutions rather than overfitting the test cases.
Output will be graded based on accuracy of the classfier learnt with the help of the kernel.

@params
    a: numpy.ndarray shape = (n,d)
    b: numpy.ndarray shape = (n,d)
'''

def kernel_1(a,b):
    '''
    This kernel is suitable when the data is linearly separable in the input space.
    '''
    out = None
    # TODO
    out = np.dot(a, b.T) + 1
    # END TODO
    return out


def kernel_2(a,b):
    '''
    This kernel has an important hyperparameter which controls upto what dimension non linearity should be mapped
    '''
    out = None
    # TODO
    # Saksham
    gamma = 1
    norm = np.linalg.norm(a[:, np.newaxis] - b, axis=2)
    out = np.exp(-gamma * norm ** 2)
    # Rathi
    # END TODO
    return out


def kernel_3(a,b):
    '''
    For this kernel assume that the data is linearly separable in very high dimensional space
    Hint: This kernel is theoretically capable of learning infinite dimension transformations. 
    '''
    out = None
    # TODO
    constant = 1
    degree = 3
    out = (np.dot(a, b.T) + constant) ** degree
    # END TODO
    return out


def svm_predictions(kernel_func, X_train, y_train, X_test):
    '''
    Use the sklearn library to make a Support Vector Classification Model
    Train the model on X_train and y_train
    Test the model on X_test and return the predictions (should be a 1D vector or list)
    '''
    y_pred = None
    # TODO
    # Saksham
    model = svm.SVC(kernel=kernel_func, C=10, gamma=1/2, shrinking=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # END TODO
    return y_pred


# Do not change anything in this function
def test_model(kernel_func, X_train, y_train, X_test, y_test):
    y_pred = svm_predictions(kernel_func, X_train, y_train, X_test)
    acc = sum(y_pred==y_test) / len(y_pred)
    return acc


if __name__ == '__main__':

    '''
    Expected accuracies on the 3 test cases are:
    1. > 95%
    2. > 89%
    3. > 98%
    '''
    kernels = [kernel_1, kernel_2, kernel_3]
    for ker in range(1,4):
        X_train = np.loadtxt(f"train_data/kernel_{ker}_sample_x.txt")
        y_train = np.loadtxt(f"train_data/kernel_{ker}_sample_y.txt")
        X_test = np.loadtxt(f"test_data/kernel_{ker}_sample_x.txt")
        y_test = np.loadtxt(f"test_data/kernel_{ker}_sample_y.txt")
        acc = test_model(kernels[ker-1], X_train, y_train, X_test, y_test)
        print(f"Accuracy on Test Case {ker}: {acc}")
    