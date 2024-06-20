'''

Implementing a Vanilla Neural Network on 10 Mini-Batches of MNIST of sample size 6000

SIZE:
- 2 Layers
    - 32 neurons in the hidden layer
    - 10 neurons in the output layer
    - 25450 total parameters, including all params (weight + biases).
    - Mini-Batch Gradient Descent

Use `np.random.default_rng(seed = 1)` for reproducible results.

REUSLTS:

'''

import numpy as np
import pandas as pd
import pickle as pkl
import time

def init_params():
    '''
    Initializting Parameters, using seed = 1 for reproducibility
    '''
    rng = np.random.default_rng(seed = 1)
    w1 = rng.normal(size = (32, 784)) * np.sqrt(1/ 784) # Using Xavier Initialization
    b1 = np.zeros((32, 1))
    w2 = rng.normal(size = (10, 32)) * np.sqrt(1/ 784) # Using Xavier Initialization
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def Leaky_ReLU(z):
    '''
    ReLU activation function
    '''
    return np.where(z>0, z, 0.01 * z)

def Leaky_ReLU_deriv(z):
    '''
    Gradient of the ReLU activation function
    '''
    return np.where(z > 0, 1, .01)

def softmax(z):
    '''
    Softmax activation function
    '''
    eps = 1e-8
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims = True)

def forward(x, w1, b1, w2, b2):
    '''
    Implementing the forward pass
    '''
    z1 = np.dot(w1, x) + b1
    a1 = Leaky_ReLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return a1, z1, a2, z2

def one_hot(y):
    '''
    One-hot encoding training samples, Y
    '''
    y_onehot = np.empty(shape = (0, 10, y.shape[2]))
    for y_minibatch in y:
        y_minihot = np.zeros((np.max(y_minibatch) + 1, y_minibatch.size))
        y_minihot[y_minibatch, np.arange(y_minibatch.size)] = 1
        y_onehot = np.concatenate((y_onehot, y_minihot[np.newaxis, ...]), axis = 0)
    return y_onehot

def acc(y, a2):
    '''
    Computing the accuracy of a forward pass
    '''
    pred = np.argmax(a2, axis = 0)
    accuracy = np.sum(y == pred) / y.size * 100
    return accuracy

def CCE(y_onehot, a2):
    '''
    Computing the loss of a forward pass
    '''
    eps = 1e-8
    loss = - (1 / y_onehot.shape[1]) * np.sum(y_onehot * np.log(a2 + eps))
    return loss

def backward(x, y_onehot, w2, a2, a1, z1):
    '''
    Computing the gradients, the backward pass
    '''
    dz2 = a2 - y_onehot
    dw2 = np.dot(dz2, a1.T) / y_onehot.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims = True) / y_onehot.shape[1]
    dz1 = np.dot(w2.T, dz2) * Leaky_ReLU_deriv(z1)
    dw1 = np.dot(dz1, x.T) / y_onehot.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims=True) / y_onehot.shape[1]
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    '''
    Applying the weight updates
    '''
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    w2 = w2 - alpha * dw2
    return w1, b1, w2, b2

def gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha, file):
    '''
    Gradient Descent
    '''
    y_onehot = one_hot(y)
    for epoch in range(epochs):
        for minibatch in range(x.shape[0]):
            a1, z1, a2, z2 = forward(x[minibatch], w1, b1, w2, b2)

            accuracy = acc(y[minibatch], a2)
            loss = CCE(y_onehot[minibatch], a2)

            dw1, db1, dw2, db2 = backward(x[minibatch], y_onehot[minibatch], w2, a2, a1, z1)
            w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

            print(f"Epoch: {epoch} | Iteration: {minibatch}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {accuracy}\n")
    save_model(file, w1, b1, w2, b2)
    return w1, b1, w2, b2

def model(x, y, epochs, alpha, file):
    '''
    One function call for the model
    '''
    try:
        w1, b1, w2, b2 = load_model(file)
        print("Loaded model!")
        print("Beginning training in 3 seconds\n") # debugging
        time.sleep(3.0)
    except FileNotFoundError:
        print("Initializing new model!\n")
        w1, b1, w2, b2 = init_params()
        print("Beginning training in 3 seconds\n") #debugging
        time.sleep(3.0)
    w1, b1, w2, b2 = gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha, file)
    return w1, b1, w2, b2

def minibatches(data, num_mini_batches):
    '''
    MiniBatching data and returing each in dims: x - (batches, features, samples per mini-batch), y - (batches, 1, samples per mini-batch)
    '''
    data = data.T

    x = data[1:785, :]
    y = data[0, :].reshape(1, -1)

    x = np.array(np.split(x, indices_or_sections=num_mini_batches, axis = 1)) / 255
    y = np.array(np.split(y, indices_or_sections=num_mini_batches, axis = 1))

    print(f"X_train Dims: {x.shape}")
    print(f"Y_train Dims: {y.shape}\n")
    print(f"Succesfully Mini Batched your Data!")
    return x, y

def save_model(file, w1, b1, w2, b2):
    '''
    Saving the model
    '''
    with open(file, 'wb') as f:
        pkl.dump((w1, b1, w2, b2), f)

def load_model(file):
    '''
    Loading the pre-trained model
    Note for User: Remove `load_model` and the try and except statements from the function `model()` if you want to train your own model.
    '''
    with open(file, 'rb') as f:
        w1, b1, w2, b2 = pkl.load(f)
        return w1, b1, w2, b2

if __name__ == "__main__":
    
    epochs = 1000
    alpha = .1
    file = 'models/MiniBatchNN.pkl'
    num_mini_batches = 10
    
    data = pd.read_csv("datasets/fashion-mnist_train.csv")
    data = np.array(data) # 60000, 785

    X_train, Y_train = minibatches(data, num_mini_batches)

    w1, b1, w2, b2 = model(X_train, Y_train, epochs, alpha, file)