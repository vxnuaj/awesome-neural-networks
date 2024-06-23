'''
Implementing a Vanilla Neural Network with L2 Regularization and Mini Batch Gradient Decsent.

Mini Batches are size of 6k samples, 10 batches total of Fashion MNIST

Results:

- Epochs: 1000 (10k training steps)
- Learning Rate: .05
- Training Accuracy: 88.633333333%
- Training Loss: 0.3323216983882467

Again, use np.random.default_rng( seed = 1 ) when initializing params for reproducibility.

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

def minibatches(data, num_batches):
    '''
    Mini Batching dataset.
    '''
    data = data.T
    
    x = data[1:, :]
    y = data[0, :].reshape(1, -1)
    
    x_batched = np.split(x, indices_or_sections=num_batches, axis = 1)
    y_batched = np.split(y, indices_or_sections=num_batches, axis = 1)
    
    x_batched = np.array(x_batched) / 255
    y_batched = np.array(y_batched) 

    return x_batched, y_batched

def init_params():
    '''
    Initializing Paramters
    '''
    rng = np.random.default_rng(seed = 1)
    w1 = rng.normal( size = (32, 784)) * np.sqrt(2/ 784) # Haiming Initialization
    b1 = np.zeros((32, 1))
    w2 = rng.normal( size = (10 , 32 )) * np.sqrt( 2/ 32) # Haiming Initialization
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def leaky_relu(z):
    '''
    Defning Leaky ReLU Activation
    '''
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    '''
    Definign Leaky ReLU Derivative
    '''
    return np.where( z > 0, 1, .01)

def softmax(z):
    '''
    Defining Softmax Activation
    '''
    eps = 1e-8
    return np.exp(z + eps) / (np.sum(np.exp(z + eps), axis = 0, keepdims = True))

def forward(x, w1, b1, w2, b2):
    '''
    Defining forward pass
    '''
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    '''
    Creating 3 dimensional 1 hot y array
    '''
    y_onehot = np.empty((0, 10, y.shape[2]))
    for minibatch in y:
        one_hot_y = np.zeros((np.max(minibatch) + 1, minibatch.size))
        one_hot_y[minibatch, np.arange(minibatch.size)] = 1
        y_onehot = np.concatenate((y_onehot, one_hot_y[np.newaxis,...]), axis = 0)
    return y_onehot

def acc(y, a2):
    '''
    Computing accuracy
    '''
    pred = np.argmax(a2, axis = 0)
    accuracy = np.sum( y == pred ) / y.size * 100
    return accuracy

def CCE(one_hot_y, a2, w2, w1, lambd):
    '''
    Defining Categorical Cros Entropy Loss with Regularization Penalty
    '''
    w1_fronorm = (lambd * np.sum(np.square(w1))) / one_hot_y.shape[1] # Not using np.linalg.norm but could be replaceable with it if desired.
    w2_fronorm = (lambd * np.sum(np.square(w2))) / one_hot_y.shape[1] # Not using np.linalg.norm but could be replaceable with it if desired.
    
    eps = 1e-8
    
    loss = - np.sum(one_hot_y * np.log(a2 + eps)) / one_hot_y.shape[1]
    loss_reg = loss + w1_fronorm + w2_fronorm
    return loss_reg
    
def backward(x, one_hot_y, w2, w1, a2, a1, z1, lambd):
    '''
    Computing gradients with regularization penalty
    '''
    dz2 = a2 - one_hot_y
    dw2 = (np.dot(dz2, a1.T) + (2 * lambd * w2)) / one_hot_y.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims=True) / one_hot_y.shape[1]
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = (np.dot(dz1, x.T) + (2 * lambd * w1)) / one_hot_y.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims=True) / one_hot_y.shape[1]
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    '''
    Updating model weights
    '''
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    return w1, b1, w2, b2

def gradient_descent(x, y, w1, b1, w2, b2, alpha, epochs, lambd, file):
    '''
    Defining the loop for gradietn descent
    '''
    y_onehot = one_hot(y)
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            z1, a1, z2, a2 = forward(x[i], w1, b1, w2, b2)
            
            loss = CCE(y_onehot[i], a2, w2, w1, lambd)
            accuracy = acc(y[i], a2)
            
            dw1, db1, dw2, db2 = backward(x[i], y_onehot[i], w2, w1, a2, a1, z1, lambd)
            w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
            
            print(f"Epoch: {epoch} | Iteration: {i}")            
            print(f"Loss: {loss}")
            print(f"Accuracy: {accuracy}\n")
    save_model(file, w1, b1, w2, b2)
    return w1, b1, w2, b2

def model(x, y, alpha, epochs, lambd, file):
    '''
    Single function call for the model.
    '''
    try:
        w1, b1, w2, b2 = load_model(file)
        print("Loaded model!")
    except FileNotFoundError:
        w1, b1, w2, b2 = init_params()
        print("Initializing new model!")
    w1, b1, w2, b2 = gradient_descent(x, y, w1, b1, w2, b2, alpha, epochs, lambd, file)
    return w1, b1, w2, b2

if __name__ == "__main__":
    data = pd.read_csv('datasets/fashion-mnist_train.csv')
    data = np.array(data)
    
    num_batches = 10
    alpha = .05
    epochs = 1000
    lambd = .1
    file = 'models/L2NN.pkl'
    
    X_train, Y_train = minibatches(data, num_batches)
    
    model(X_train, Y_train, alpha, epochs, lambd, file)