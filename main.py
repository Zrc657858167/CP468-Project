"""
https://github.com/Zrc657858167/CP486-Project

Jonatham Chang 190787860
    GitHub login ID: JonathamC

Ruicheng Zhao 190519490
    GitHub login ID: Zrc657858167

"""

# import
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def init_params():
    """
    Generate random values between 0 and 1. 
    """
    weights1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784)) 
    bias1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10) 
    weights2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    bias2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784)) 
    return weights1, bias1, weights2, bias2

def forward_prop(weights1, bias1, weights2, bias2, inputLayer):
    """
    Forward propagation is a way to move from the input layer to the output layer in  neural network 
    """
    unactivated_firstLayer = weights1.dot(inputLayer) + bias1
    activation1 = ReLU(unactivated_firstLayer)
    unactivated_secondLayer = weights2.dot(activation1) + bias2
    activation2 = softmax(unactivated_secondLayer)
    return unactivated_firstLayer, activation1, unactivated_secondLayer, activation2

def ReLU(n):
    """
    activation function that returns 0 if input n is negative 
    returns 1 if input n is positive and is linear
    ReLU stands for Rectified Linear Unit which is a piecewise linear function
    """
    return np.maximum(n, 0)

def softmax(n):
    """
    activation function that changes input n values into values between 0 to 1 
    it converts a vector of numbers into a vector of probabilities
    so each output nodes have a probability value
    """
    return np.exp(n) / sum(np.exp(n))

    


def derivativeReLU(Z):
    """
    """
    return Z > 0

def one_hot(Y):
    # creating correctly sized matrix 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    # for each row in Y.size at location Y, set its value to 1
    one_hot_Y[np.arange(Y.size), Y] = 1
    # transpose it so each column to be example instead of row
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(unactivated_firstLayer, activation1, unactivated_secondLayer, activation2, weights1, weights2, firstLayer, Y):
    """
    Backward propagation is a way to move backward from output to the input layer
    """
    one_hot_Y = one_hot(Y)
    dZ2 = activation2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(activation1.T)
    db2 = 1 / m * np.sum(dZ2) 

    dZ1 = weights2.T.dot(dZ2) * derivativeReLU(unactivated_firstLayer)
    dW1 = 1 / m * dZ1.dot(firstLayer.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(weights1, bias1, weights2, bias2, dW1, db1, dW2, db2, alpha):
    weights1 = weights1 - alpha * dW1
    bias1 = bias1 - alpha * db1    
    weights2 = weights2 - alpha * dW2  
    bias2 = bias2 - alpha * db2    
    return weights1, bias1, weights2, bias2

def gradient_descent(inputLayer, Y, alpha, i):
    """
    optimizing predictions by finding the local minimum/maximum
    """
    print("Training in progress!")
    weights1, bias1, weights2, bias2 = init_params()
    for i in range(i):
        unactivated_firstLayer, activation1, unactivated_secondLayer, activation2 = forward_prop(weights1, bias1, weights2, bias2, inputLayer)
        dW1, db1, dW2, db2 = backward_prop(unactivated_firstLayer, activation1, unactivated_secondLayer, activation2, weights1, weights2, inputLayer, Y)
        weights1, bias1, weights2, bias2 = update_params(weights1, bias1, weights2, bias2, dW1, db1, dW2, db2, alpha)
        
        # every 20 iterations, we check the accuracy 
        if i % 20 == 0:
            predictions = np.argmax(activation2, 0)
            print("\tPrediction Accuracy: ", np.sum(predictions == Y) / Y.size)
    return weights1, bias1, weights2, bias2, np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = np.argmax(A2, 0)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# load the data 
mnist_data = pd.read_csv('train.csv')
mnist_data = np.array(mnist_data)
# get the array dimensions of the matrix 
m, n = mnist_data.shape
#shuffle the data 
np.random.shuffle(mnist_data) 

# first 1000 data is development set
# will be used to evaluate and optimize our model 
developmentSet = mnist_data[0:1000].T
Y_dev = developmentSet[0]
X_dev = developmentSet[1:n]
X_dev = X_dev / 255.

# remaining data after first 100 will be the training data 
# will be used to train our model
training_data = mnist_data[1000:m].T
Y_train = training_data[0]
X_train = training_data[1:n]
X_train = X_train / 255.


weights1, bias1, weights2, bias2, results = gradient_descent(X_train, Y_train, 0.10, 500)

dev_predictions = make_predictions(X_dev, weights1, bias1, weights2, bias2)
correctPrediction = 0 
print("The first 10 predictions")
for i in range(10): # len(dev_predictions) for seeing result of data 
    pred = dev_predictions[i] == Y_dev[i] 
    if pred: 
        correctPrediction += 1
    print("\t[{}] = {}? -->{}".format(dev_predictions[i], Y_dev[i], dev_predictions[i] == Y_dev[i]))

print("Results: {:.2f}% efficiency".format(results * 100))


test_prediction(0, weights1, bias1, weights2, bias2)
test_prediction(1, weights1, bias1, weights2, bias2)
test_prediction(2, weights1, bias1, weights2, bias2)
test_prediction(3, weights1, bias1, weights2, bias2)