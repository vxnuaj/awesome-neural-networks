import numpy as np
import pandas as pd
import pickle as pkl

'''
Implementing a Neural Network with momentum based gradient descent

BETA: .9
ALPHA: .05
EPOCHS: 1000 (10000 ITERATIONS)

RESULTS:
- ACCURACY: 88.8833333333333%
- LOSS: 0.3485038665070181

NOTE: Could likely get a better results by tuning hyperparameters, BETA might be too high for the learning rate. Either decrease learning rate or decrease BETA.

'''

def save_model(file, w1, b1, w2, b2):
    with open(file, 'wb') as f:
        pkl.dump((w1, b1, w2, b2), f)

def load_model(file):
    with open(file, 'rb') as f:
        return pkl.load(f)

def minibatches(data, mini_batch_num):
    '''
    Minibatching dataset
    '''

    data = data.T
    X_train = data[1:, :] / 255
    Y_train = data[0, :].reshape(1, -1)
    
    X_train = np.split(X_train, indices_or_sections = mini_batch_num, axis = 1)
    Y_train = np.split(Y_train, indices_or_sections = mini_batch_num, axis = 1)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train

def init_params():
    '''
    Initializng Parameters
    '''
    rng = np.random.default_rng(seed = 1)
    w1 = rng.normal(size = (32, 784)) * np.sqrt(2/784) # Kaiming Init.
    b1 = np.zeros((32, 1))
    w2 = rng.normal(size = (10, 32)) * np.sqrt(2/ 32) # Kaiming Init.
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z > 0, 1, .01)

def softmax(z):
    eps = 1e-8
    return np.exp(z + eps) / np.sum(np.exp(z + eps),axis = 0, keepdims = True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def accuracy(y, a2):
    pred = np.argmax(a2, axis = 0)
    acc = np.sum(y == pred) / y.size * 100
    return acc

def one_hot(y):
    one_hot_y = np.empty((0, 10, y.shape[2]))
    for minibatch in y:
        mini_onehot = np.zeros((np.max(minibatch) + 1, minibatch.size))
        mini_onehot[minibatch, np.arange(minibatch.size)] = 1
        one_hot_y = np.concatenate((one_hot_y, mini_onehot[np.newaxis, ...]))
    return one_hot_y

def CCE(mini_onehot, a2):
    eps = 1e-8
    loss = - ( 1 / mini_onehot.shape[1] ) * np.sum(mini_onehot * np.log(a2 + eps))
    return loss

def backward(x, mini_onehot, w2, a2, a1, z1, vdw1, vdb1, vdw2, vdb2, beta):

    '''
    Defining the backward pass.
    '''
    dz2 = a2 - mini_onehot
    dw2 = np.dot(dz2, a1.T) / mini_onehot.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims = True) / mini_onehot.shape[1]
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / mini_onehot.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims = True) / mini_onehot.shape[1]

    '''
    Computing exponentially weighted averages of the gradients

    NOTE: Not using the bias correction, instead opting for warming up the averaged gradients
    '''
    vdw2 = beta * vdw2 + ( 1 - beta ) * dw2
    vdb2 = beta * vdb2 + ( 1 - beta ) * db2
    vdw1 = beta * vdw1 + ( 1 - beta ) * dw1
    vdb1 = beta * vdb1 + ( 1 - beta ) * db1
    return vdw1, vdb1, vdw2, vdb2

def update(w1, b1, w2, b2, vdw1, vdb1, vdw2, vdb2, alpha):
    w1 -= alpha * vdw1
    b1 -= alpha * vdb1
    w2 -= alpha * vdw2
    b2 -= alpha * vdb2
    return w1, b1, w2, b2

def gradient_descent(x, y, w1, b1, w2, b2, alpha, beta, epochs, file):
    one_hot_y = one_hot(y)
    vdw1, vdb1, vdw2, vdb2 = 0, 0, 0, 0 # Setting initial values of EWA gradients to 0
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            z1, a1, z2, a2 = forward(x[i], w1, b1, w2, b2)

            acc = accuracy(y[i], a2)
            loss = CCE(one_hot_y[i], a2)

            vdw1, vdb1, vdw2, vdb2 = backward(x[i], one_hot_y[i], w2, a2, a1, z1, vdw1, vdb1, vdw2,vdb2, beta)
            w1, b1, w2, b2 = update(w1, b1, w2, b2, vdw1, vdb1, vdw2, vdb2, alpha)

            print(f"Epoch: {epoch} | Iteration: {i}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {acc}")

    save_model(file, w1, b1, w2, b2)
    return w1, b1, w2, b2

def model(x, y, alpha, beta, epochs, file):
    try:
        w1, b1, w2, b2 = load_model(file)
    except FileNotFoundError:
        w1, b1, w2, b2 = init_params()

    w1, b1, w2, b2 = gradient_descent(x, y, w1, b1, w2, b2, alpha, beta, epochs, file)
    return w1, b1, w2, b2


if __name__ == "__main__":
    data = pd.read_csv('datasets/fashion-mnist_train.csv')
    data = np.array(data) # ( 60000, 785 ) 
    
    X_train, Y_train = minibatches(data, 10)

    alpha = .05
    beta = .9
    epochs = 990
    file = 'models/MomentumNN2.pkl'

    w1, b1, w2, b2 = model(X_train, Y_train, alpha, beta, epochs, file)
