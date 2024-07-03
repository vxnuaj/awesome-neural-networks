'''

Implementing Gradient Descent with Nesterov Momentum.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys

def load_model(file):
    with open(file) as f:
        return pickle.load(f)

def save_model(file, w1, b1, w2, b2, w3, b3):
    with open(file) as f:
        pickle.dump((w1, b1, w2, b2, w3, b3), f)

def init_params():
    rng = np.random.default_rng(seed = 1)
    w1 = rng.normal(size = (32, 784)) * np.sqrt(1/784)
    b1 = np.zeros((32, 1))
    w2 = rng.normal(size = (10, 32)) * np.sqrt(1/784)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 0, z, (.01 * z))

def leaky_relu_deriv(z):
    return np.where(z>0, 1, .1)

def softmax(z):
    eps = 1e-6
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims=True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    #print(f"Z2: {np.max(z2)}")
    try:
        a2 = softmax(z2)
    except RuntimeWarning as e:
        sys.exit(f"overflow!: {e}")
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros((np.max(y)+1, y.size)) #10, 60000
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

def cat_cross(one_hot_y, a):
    eps = 1e-10
    l = (-np.sum(one_hot_y * np.log(a + eps))) / 60000
    return l

def accuracy(y, a2):
    pred = np.argmax(a2, axis = 0)
    acc = np.sum(pred == y) / 60000 * 100
    return acc


def backward(x, one_hot_y, w2, w1, a2, a1, z2, z1):
    dz2 = a2 - one_hot_y
    dw2 = np.dot(dz2, a1.T) / 60000
    db2 = np.sum(dz2, axis = 1, keepdims=True) / 60000
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / 60000
    db1 = np.sum(dz1, axis=1, keepdims=True) / 60000
    return dw2, db2, dw1, db1

def update_momentum(x, one_hot_y, w1, b1, w2, b2, vdw1, vdb1, vdw2, vdb2, alpha, beta):

    '''
    Computing the lookahead parameters. 
    '''
    w1_lookahead = w1 - beta * vdw1 
    b1_lookahead = b1 - beta * vdb1
    w2_lookahead = w2 - beta * vdw2
    b2_lookahead = b2 - beta * vdb2

    '''
    Computing the gradients of the lookahead parameters. 
    '''
    z1_lookahead, a1_lookahead, z2_lookahead, a2_lookahead = forward(x, w1_lookahead, b1_lookahead, w2_lookahead, b2_lookahead)
    dw2, db2, dw1, db1 = backward(x, one_hot_y, w2_lookahead, w1_lookahead, a2_lookahead, a1_lookahead, z2_lookahead, z1_lookahead)    

    '''
    Accumulating the previous gradients. 
    '''
    vdw1 = beta * vdw1 + dw1 
    vdb1 = beta * vdb1 + db1
    vdw2 = beta * vdw2 + dw2
    vdb2 = beta * vdb2 + db2

    '''
    Weight Update 
    '''
    w1 = w1 - alpha * vdw1
    b1 = b1 - alpha * vdb1
    w2 = w2 - alpha * vdw2
    b2 = b2 - alpha * vdb2
    return w1, b1, w2, b2, vdw1, vdb1, vdw2, vdb2

def gradient_descent_momentum(x, y, w1, b1, w2, b2, epochs, alpha, beta, file):
    one_hot_y = one_hot(y)
    vdw1 = 0
    vdb1 = 0
    vdw2 = 0
    vdb2 = 0

    epochs_vec = []
    acc_vec = []
    loss_vec = []

    for epoch in range(epochs):
        _, _, _, a2 = forward(x, w1, b1, w2, b2)

        l = cat_cross(one_hot_y, a2)
        acc = accuracy(y, a2)

        w1, b1, w2, b2, vdw1, vdb1, vdw2, vdb2 = update_momentum(x, one_hot_y, w1, b1, w2, b2, vdw1, vdb1, vdw2, vdb2, alpha, beta)
    
        print(f"epoch: {epoch}")
        print(f"loss: {l}")
        print(f"acc: {acc}%\n")

        loss_vec.append(l)
        acc_vec.append(acc)
        epochs_vec.append(epoch)

    return w1, b1, w2, b2, loss_vec, acc_vec, epochs_vec
    
def model(x, y, epochs, alpha, beta, file):
    try:
        w1, b1, w2, b2 = load_model(file)
        print("found model! initializing model params!")

    except FileNotFoundError:
        print("model not found! initializing new params!")
        w1, b1, w2, b2= init_params()
    
    w1, b1, w2, b2, loss_vec, acc_vec, epochs_vec  = gradient_descent_momentum(x, y, w1, b1, w2, b2, epochs, alpha, beta,file)
    return w1, b1, w2, b2, loss_vec, acc_vec, epochs_vec

if __name__ == "__main__":

    data = np.array(pd.read_csv('data/fashion-mnist_train.csv'))

    X_train = data[:, 1:786].T / 255 #784, 60000
    Y_train = data[:, 0].reshape(1, -1) #1, 60000

    file = '../models/BatchNN.pkl'

    w1, b1, w2, b2, loss_vec, acc_vec, epochs_vec = model(X_train, Y_train, epochs = 250, alpha = .1, beta = .9, file = file)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(epochs_vec, acc_vec, label='Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(epochs_vec, loss_vec, label='Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.show()
