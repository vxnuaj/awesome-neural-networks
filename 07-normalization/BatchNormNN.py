'''
Implementing a Neural Network with Adam + BatchNorm from Scratch

ALPHA: .05
BETA_1: .9
BETA_2: .99
EPOCHS: 1000 ( 10000 TRAINING STEPS )

RESULTS:

- Accuracy: 100.0 ( likely 99.99999 -> ad infinitum)
- Loss: 1.7592683158737689e-06

'''

import numpy as np
import pandas as pd
import pickle as pkl

def save_model(file, w1, b1, g1, w2, b2, g2):
    with open(file, 'wb') as f:
        pkl.dump((w1, b1, g1, w2, b2, g2), f)

def load_model(file):
    with open(file, 'rb') as f:
        return pkl.load(f)

def minibatches(data, num_batches):

    data = data.T
    
    X_train = data[1:, :] / 255
    Y_train = data[0, :].reshape(1, -1)

    X_train = np.split(X_train, indices_or_sections = num_batches, axis = 1)
    Y_train = np.split(Y_train, indices_or_sections = num_batches, axis = 1)

    X_train = np.array(X_train) 
    Y_train = np.array(Y_train)

    return X_train, Y_train

def init_params():
    rng = np.random.default_rng(seed = 1)
    w1 = rng.normal( size = (32, 784)) * np.sqrt(2/ 784) # Kaiming Initialization.
    b1 = np.zeros((32, 1))
    g1 = np.ones((32, 1))
    w2 = rng.normal( size = (10, 32)) * np.sqrt(2/ 32) # Kaiming Initialization.
    b2 = np.zeros((10, 1))
    g2 = np.ones((10, 1))
    return w1, b1, g1, w2, b2, g2

def leaky_relu(z):
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z > 0, 1, .01)

def softmax(z):
    eps = 1e-8
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims = True)

def batchnorm(z):
    eps = 1e-8

    mu = np.mean(z, axis = 1, keepdims = True) 
    var = np.var(z, axis = 1, keepdims = True)
    z = (z - mu)/ (np.sqrt(var + eps))
    return z, np.sqrt(var + eps)

def forward(x, w1, b1, g1, w2, b2, g2):
    z1 = np.dot(w1, x)
    z1_l, std1 = batchnorm(z1)
    z1_norm = (z1_l * g1) + b1 
    a1 = leaky_relu(z1_norm)
    z2 = np.dot(w2, a1)
    z2_l, std2 = batchnorm(z2)
    z2_norm = (z2_l * g2) + b2
    a2 = softmax(z2_norm)
    return  z1_l, z1_norm, a1, z2_l, z2_norm, a2, std1, std2

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

def backwards(x, mini_onehot, w2, a2, a1, g2, g1, z2_l, z2_std, z1_l, z1_std, z1_norm):
    eps = 1e-8

    dz2_norm = a2 - mini_onehot
    dg2 = dz2_norm * z2_l
    db2 =  dz2_norm
    dz2 = dz2_norm * g2 * (1 / np.abs(z2_std + eps)) 
    dw2 = np.dot(dz2, a1.T) / mini_onehot.shape[1] 
     
    dz1_norm = np.dot(w2.T, dz2) * leaky_relu_deriv(z1_norm)
    dg1 = dz1_norm * z1_l
    db1 = dz1_norm 
    dz1 = dz1_norm * g1 * (1 / np.abs(z1_std+ eps))
    dw1 = np.dot(dz1, x.T) / mini_onehot.shape[1]
    return dw1, db1, dg1, dw2, db2, dg2

def update(w1, b1, g1, w2, b2, g2, dw1, db1, dg1, dw2, db2, dg2, vdw1, vdb1, vdg1, vdw2, vdb2, vdg2, sdw1, sdb1, sdg1, sdw2, sdb2, sdg2, alpha, beta_1, beta_2):
   
    '''
    Computing the first moment 
    '''
    
    vdw1 = beta_1 * vdw1 + ( 1 - beta_1) * dw1
    vdb1 = beta_1 * vdb1 + ( 1 - beta_1 ) * db1
    vdg1 = beta_1 * vdg1 + ( 1 - beta_1 ) * dg1
    
    vdw2 = beta_1 * vdw2 + ( 1 - beta_1 ) * dw2
    vdb2 = beta_1 * vdb2 + ( 1 - beta_1 ) * db2
    vdg2 = beta_1 * vdg2 + ( 1 - beta_1 ) * dg2
    
    '''
    Computing the second moment 
    ''' 
    sdw1 = beta_2 * sdw1 + ( 1 - beta_2 ) * np.square(dw1)
    sdb1 = beta_2 * sdb1 + ( 1 - beta_2 ) * np.square(db1)
    sdg1 = beta_2 * sdg1 + ( 1 - beta_2 ) * np.square(dg1)
    
    sdw2 = beta_2 * sdw2 + ( 1 - beta_2 ) * np.square(dw2)
    sdb2 = beta_2 * sdb2 + ( 1 - beta_2 ) * np.square(db2)
    sdg2 = beta_2 * sdg2 + ( 1 - beta_2 ) * np.square(dg2)
   
    '''
    Adam's Weight Update
    '''   
    eps = 1e-8 
    w1 = w1 - ( alpha / np.sqrt(sdw1 + eps)) * vdw1
    b1 = b1 - ( alpha / np.sqrt(sdb1 + eps)) * vdb1
    g1 = g1 - (alpha / np.sqrt(sdg1 + eps)) * vdg1
    
    w2 = w2 - ( alpha / np.sqrt (sdw2 + eps)) * vdw2
    b2 = b2 - (alpha / np.sqrt(sdb2 + eps)) * vdb2
    g2 = g2 - (alpha / np.sqrt (sdg2 + eps)) * vdg2 
     
    return w1, b1, g1, w2, b2, g2, vdw1, vdb1, vdg1, vdw2, vdb2, vdg2, sdw1, sdb1, sdg1, sdw2, sdb2, sdg2

def gradient_descent(x, y, w1, b1, g1, w2, b2, g2, alpha, beta_1, beta_2, epochs, file ):
    one_hot_y = one_hot(y)
    vdw1, vdb1, vdg1, vdw2, vdb2, vdg2 = 0, 0, 0, 0, 0, 0
    sdw1, sdb1, sdg1, sdw2, sdb2, sdg2 = 0, 0, 0, 0, 0, 0
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            z1_l, z1_norm, a1, z2_l, z2_norm, a2, std1, std2 = forward(x[i], w1, b1,g1, w2, b2, g2)

            acc = accuracy(y[i], a2)
            loss = CCE(one_hot_y[i], a2)
                
            dw1, db1, dg1, dw2, db2, dg2 = backwards(x[i], one_hot_y[i], w2, a2, a1, g2, g1, z2_l, std2, z1_l, std1, z1_norm)
            w1, b1, g1, w2, b2, g2, vdw1, vdb1, vdg1, vdw2, vdb2, vdg2, sdw1, sdb1, sdg1, sdw2, sdb2, sdg2 = update(w1, b1, g1, w2, b2, g2, dw1, db1, dg1, dw2, db2, dg2, vdw1, vdb1, vdg1, vdw2, vdb2, vdg2, sdw1, sdb1, sdg1, sdw2, sdb2, sdg2, alpha, beta_1, beta_2)   
               
                
            print(f"Epoch: {epoch} | Iteration: {i}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {acc}\n")

    save_model(file, w1, b1, g1, w2, b2, g2)
    return w1, b1, g1, w2, b2, g2


def model(x, y, alpha, beta_1, beta_2, epochs, file):
    try: 
        w1, b1, g1, w2, b2, g2 = load_model(file)
        print("Succesfully loaded model!")
    except FileNotFoundError:
        w1, b1, g1, w2, b2, g2 = init_params()
        print("Initializing new model!")
    w1, b1, g1 ,w2, b2, g2 = gradient_descent(x, y, w1, b1, g1, w2, b2, g2, alpha, beta_1, beta_2, epochs, file )
    return w1, b1, w2, b2


if __name__ == "__main__":

    data = pd.read_csv('datasets/fashion-mnist_train.csv')
    data = np.array(data) # 60000, 785

    num_batches = 10
    alpha = .05
    epochs = 1000
    beta_1 = .9
    beta_2 = .99
    file = 'models/BatchNorm.pkl'
    
    X_train, Y_train = minibatches(data, num_batches)

    model(X_train, Y_train, alpha, beta_1, beta_2, epochs, file)
