''''
Implementing a Neural Network with Adam from Scratch

Alpha: .05
Beta_1: .9
Beta_2: .9

RESULTS after 1k Epochs (10000 training steps):
- Acc: 97.05% 
- Loss: .09081896

np.random.default_rng(seed = 1) for reproducibility

''''

import numpy as np
import pandas as pd
import pickle as pkl

def save_model(file, w1, b1, w2, b2):
    with open(file, 'wb') as f:
        pkl.dump((w1, b1, w2, b2), f)

def load_model(file):
    with open(file, 'rb') as f:
        return pkl.load(f)

def minibatches(data, num_batches):

    data = data.T
    
    X_train = data[1:, :] / 255
    Y_train = data[0, :].reshape(1, -1)

    X_train = np.split(X_train, indices_or_sections = num_batches, axis = 1)
    Y_train = np.split(Y_train, indices_or_sections = num_batches, axis = 1)

    X_train = np.array(X_train) # 10, 784, 6000
    Y_train = np.array(Y_train) # 10, 1, 6000

    return X_train, Y_train

def init_params():
    rng = np.random.default_rng(seed = 1)
    w1 = rng.normal( size = (32, 784)) * np.sqrt(2/ 784) # Kaiming Init.
    b1 = np.zeros((32, 1))
    w2 = rng.normal( size = (10, 32)) * np.sqrt(2/ 32) # Kaiming Init.
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z > 0, 1, .01)

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
        mini_onehot = np.zeros((np.max(minibatch) + 1, minibatch.size))
        mini_onehot[minibatch, np.arange(minibatch.size)] = 1
        one_hot_y = np.concatenate((one_hot_y, mini_onehot[np.newaxis, ...]), axis = 0)
    return one_hot_y

def accuracy(y, a2):
    pred = np.argmax(a2, axis = 0)
    acc = np.sum( y == pred ) / y.size * 100
    return acc

def CCE(mini_onehot, a2):
    eps = 1e-8
    loss = - ( 1 / mini_onehot.shape[1] ) * np.sum(mini_onehot * np.log(a2 + eps))
    return loss

def backwards(x, mini_onehot, w2, a2, a1, z1):
    dz2 = a2 - mini_onehot
    dw2 = np.dot(dz2, a1.T) / mini_onehot.shape[1] 
    db2 = np.sum(dz2, axis = 1, keepdims = True) / mini_onehot.shape[1]
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T)/ mini_onehot.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims = True) / mini_onehot.shape[1]
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, vdw1, vdb1, vdw2, vdb2, sdw1, sdb1, sdw2, sdb2, alpha, beta_1, beta_2):
    eps = 1e-8 

    """
    Computing the First Moment
    """

    vdw1 = (beta_1 * vdw1) + ( 1 - beta_1 ) * dw1
    vdb1 = (beta_1 * vdb1) + ( 1 - beta_1 ) * db1
    vdw2 = (beta_1 * vdw2) + ( 1 - beta_1 ) * dw2
    vdb2 = (beta_1 * vdb2) + ( 1 - beta_1 ) * db2

    """
    Computing the Second Moment
    """

    sdw1 = (beta_2 * sdw1) + ( 1 - beta_2 ) * np.square(dw1)
    sdb1 = (beta_2 * sdb1) + ( 1 - beta_2 ) * np.square(db1)
    sdw2 = (beta_2 * sdw2) + ( 1 - beta_2 ) * np.square(dw2)
    sdb2 = (beta_2 * sdb2) + ( 1 - beta_2 ) * np.square(db2)

    """
    Adam's Update Rule
    """

    w1 = w1 - (alpha / np.sqrt(sdw1 + eps)) * vdw1
    b1 = b1 - (alpha / np.sqrt(sdb1 + eps)) * vdb1
    w2 = w2 - (alpha / np.sqrt(sdw2 + eps)) * vdw2
    b2 = b2 - (alpha / np.sqrt(sdb2 + eps)) * vdb2

    return w1, b1 ,w2, b2, vdw1, vdb1, vdw2, vdb2, sdw1, sdb1, sdw2, sdb2

def gradient_descent(x, y, w1, b1, w2, b2, alpha, beta_1, beta_2, epochs, file ):
    one_hot_y = one_hot(y)
    vdw1, vdb1, vdw2, vdb2 = 0, 0, 0, 0
    sdw1, sdb1, sdw2, sdb2 = 0, 0, 0, 0
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            z1, a1, z2, a2 = forward(x[i], w1, b1, w2, b2)

            acc = accuracy(y[i], a2)
            loss = CCE(one_hot_y[i], a2)
                
            dw1, db1, dw2, db2 = backwards(x[i], one_hot_y[i], w2, a2, a1, z1)
            w1, b1, w2, b2, vdw1, vdb1, vdw2, vdb2, sdw1, sdb1, sdw2, sdb2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, vdw1, vdb1, vdw2, vdb2, sdw1, sdb1, sdw2, sdb2, alpha, beta_1, beta_2)
                
            print(f"Epoch: {epoch} | Iteration: {i}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {acc}")
        

    save_model(file, w1, b1, w2, b2)
    return w1, b1, w2, b2


def model(x, y, alpha, beta_1, beta_2, epochs, file):
    try: 
        w1, b1, w2, b2 = load_model(file)
        print("Succesfully loaded model!")
    except FileNotFoundError:
        w1, b1, w2, b2 = init_params()
        print("Initializing new model!")
    w1, b1 ,w2, b2 = gradient_descent(x, y, w1, b1, w2, b2, alpha, beta_1, beta_2, epochs, file)
    return w1, b1, w2, b2


if __name__ == "__main__":

    data = pd.read_csv('datasets/fashion-mnist_train.csv')
    data = np.array(data) # 60000, 785

    num_batches = 10
    alpha = .01
    epochs = 1000
    beta_1 = .9
    beta_2 = .9
    file = 'models/AdamNN.pkl'
    
    X_train, Y_train = minibatches(data, num_batches)

    model(X_train, Y_train, alpha, beta_1, beta_2, epochs, file)
