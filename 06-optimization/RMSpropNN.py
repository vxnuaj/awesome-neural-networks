'''
Implementation of a Neural Network with RMSprop.

Trained on 10 minibatches of Fashion MNIST, each totaling up to 6k samples.

ALPHA: .01
BETA: .99

RESULTS:
- Accuracy: 89.75%
- Loss: .261555

'''


import numpy as np
import pandas as pd 
import pickle as pkl

def save_model(file, w1, b1, w2, b2):
    with open(file, 'wb') as f:
        pkl.dump((w1, b1, w2, b2), f)

def load_model(file):
    with open(file, 'rb') as f:
        return pkl.load(f)

def minibatches(data, minibatch_num):
    data = data.T

    X_train = data[1:, :] / 255
    Y_train = data[0, :].reshape(1, -1)

    X_train = np.split(X_train, indices_or_sections = minibatch_num, axis = 1)
    Y_train = np.split(Y_train, indices_or_sections = minibatch_num, axis = 1)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train 


def init_params():
    rng = np.random.default_rng( seed = 1 )
    w1 = rng.normal(size = (32, 784)) * np.sqrt(2 / 784) # Kaiming Initialization
    b1 = np.zeros((32, 1))
    w2 = rng.normal(size = (10, 32)) * np.sqrt( 2 / 32) # Kaiming Initialization
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 1, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z > 1, 1, .01)

def softmax(z):
    eps = 1e-8
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims = True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.empty((0, 10, y.shape[2]))
    for minibatch in y:
        mini_one_hot = np.zeros((np.max(minibatch) + 1, minibatch.size))
        mini_one_hot[minibatch, np.arange(minibatch.size)] = 1
        one_hot_y = np.concatenate((one_hot_y, mini_one_hot[np.newaxis, ...]), axis = 0)
    return one_hot_y


def accuracy(y, a2):
    pred = np.argmax(a2, axis = 0)
    acc = np.sum(y == pred) / y.size * 100
    return acc

def CCE(mini_one_hot, a2):
    eps = 1e-8
    loss = - (1 / mini_one_hot.shape[1]) * np.sum(mini_one_hot * np.log(a2 + eps))
    return loss


def backward(x, mini_one_hot, w2, a2, a1, z1, sdw2, sdb2, sdw1, sdb1, beta ):
    dz2 = a2 - mini_one_hot
    dw2 = np.dot(dz2, a1.T) / mini_one_hot.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims = True) / mini_one_hot.shape[1]
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / mini_one_hot.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims = True) / mini_one_hot.shape[1]

    ''' Computing the exponentially weighted gradients squared'''

    sdw2 = beta * sdw2 + ( 1 - beta ) * np.square(dw2)
    sdb2 = beta * sdb2 + ( 1 - beta ) * np.square(db2)
    sdw1 = beta * sdw1 + ( 1 - beta ) * np.square(dw1)
    sdb1 = beta * sdb1 + ( 1 - beta ) * np.square(db1)
    return dw1, db1, dw2, db2, sdw1, sdb1, sdw2, sdb2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, sdw1, sdb1, sdw2, sdb2, alpha):
    eps = 1e-8 

    '''
    Adaptively scaling the learning rates through RMS of exponentially weighted gradients squared
    '''

    w1 -= (alpha / np.sqrt(sdw1 + eps)) * dw1 
    b1 -= (alpha / np.sqrt(sdb1 + eps)) * db1
    w2 -= (alpha / np.sqrt(sdw2 + eps)) * dw2
    b2 -= (alpha / np.sqrt(sdb2 + eps)) * db2
    return w1, b1, w2, b2

def gradient_descent(x, y, w1, b1, w2, b2, alpha, beta, epochs, file):
    one_hot_y = one_hot(y)
    sdw1, sdb1, sdw2, sdb2 = 0, 0, 0, 0
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            z1, a1, z2, a2 = forward(x[i], w1, b1, w2, b2)

            loss = CCE(one_hot_y[i], a2)
            acc = accuracy(y[i], a2)

            dw1, db1, dw2, db2, sdw1, sdb1, sdw2, sdb2 = backward(x[i], one_hot_y[i], w2, a2, a1, z1, sdw2, sdb2, sdw1, sdb1, beta)       
            w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, sdw1, sdb1, sdw2, sdb2, alpha)

            print(f"Epoch: {epoch} | Iteration {i}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {acc}")
    print("saved model!")
    return w1, b1, w2, b2

def model(x, y, alpha, beta, epochs, file):
    try:
        w1, b1, w2, b2 = load_model(file)
        print("Succesfully loaded model!")
    except FileNotFoundError:
        w1, b1, w2, b2 = init_params()
        print("Model not found! Initializing new paramters!")

    w1, b1, w2, b2 = gradient_descent(x, y, w1, b1, w2, b2, alpha, beta, epochs, file)
    return w1, b1, w2, b2

if __name__ == "__main__":
    data = pd.read_csv('datasets/fashion-mnist_train.csv')
    data = np.array(data) # 60000, 785
   
    minibatch_num = 10

    X_train, Y_train = minibatches(data, minibatch_num)

    alpha = .01
    beta = .99
    epochs = 980
    file = 'models/RMSpropNN.pkl'

    w1, b1, w2, b2 = model(X_train, Y_train, alpha, beta, epochs, file)
