'''
Implementing a Neural Network with a Cyclic Learning Rate

The learning rate was found over the lr-range-test propseod by Leslie Smith in his paper, "Cyclical Learning Rates for Training Neural Networks" ( https://arxiv.org/pdf/1506.01186 )

The minimum and maximum bounds for a learning rate were tested first over 10 eppchs, to gauge initial issues with a larger learning rate, then over 100 to gauge training issues when optimizing for a 
precise convergence on the optima of the loss.

Results:

- alpha_min: .01
- alpha_max: .3
- accuracy: 91.0833333%
- loss: 0.26272359581801596


'''


import numpy as np
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt

def load_model(file):
    with open(file, 'rb') as f:
        return pkl.load(f)
    
def save_model(file, w1, b1, w2, b2):
    with open(file, 'wb') as f:
        pkl.dump((w1, b1, w2, b2), f)

def minibatches(data, mini_batch_num):
    '''
    MiniBatching `data`, into X_train ( mini_batch_num, 784, data.shape[1] / mini_batch_num) and Y_train ( mini_batch_num, 1, 6000)
    ''' 
    data = data.T
    
    X_train = data[1:, :] / 255
    Y_train = data[0, :].reshape(1, -1)
   
    X_train = np.array(np.split(X_train, indices_or_sections = mini_batch_num, axis = 1))
    Y_train = np.array(np.split(Y_train, indices_or_sections = mini_batch_num, axis = 1))

    return X_train, Y_train # (10, 784, 6000) & (10, 1, 6000)

def init_params():
    '''
    Initializing Parameters
    '''
    rng = np.random.default_rng(seed = 1)
    w1 = rng.normal(size = (32, 784)) * np.sqrt(2 / 784) # Using Kaiming Initializaion
    b1 = np.zeros((32, 1))
    w2 = rng.normal( size = (10 ,32)) * np.sqrt(2/ 32)# Kaiming Initialization
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def leaky_relu(z):
    '''
    Leaky Relu
    '''
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    '''
    Derivative of Leaky Relu
    '''
    return np.where(z > 0, 1, .01)

def softmax(z):
    """
    Softmax acttivation
    """
    eps = 1e-10
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims = True)

def forward(x,w1, b1, w2, b2):
    '''
    Forward Pass
    '''
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2 
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    '''
    One hot Encoding Mini Batched Data
    '''
    one_hot_y = np.empty((0, 10, y.shape[2]))
    for minibatch in y:
        mini_onehot = np.zeros((np.max(minibatch) + 1, minibatch.size))
        mini_onehot[minibatch, np.arange(minibatch.size)] = 1
        one_hot_y = np.concatenate((one_hot_y, mini_onehot[np.newaxis, ...]))
    return one_hot_y

def accuracy(y, a2):
    '''
    Comuting Accuracy
    '''
    pred = np.argmax(a2, axis = 0)
    acc = np.sum(y == pred) / y.size * 100
    return acc

def CCE(mini_onehot, a2):
    '''
    Loss function for a Mini Batch
    '''
    eps = 1e-8
    loss = - (1 / mini_onehot.shape[1]) * np.sum(mini_onehot * np.log(a2 + eps))
    return loss

def backwards(x, one_hot_y, w2, a2, a1, z1):
    '''
    Backward Pass
    '''
    dz2 = a2 - one_hot_y
    dw2 = np.dot(dz2, a1.T) / one_hot_y.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims = True) / one_hot_y.shape[1]
    dz1 = np.dot(w2.T, dz2 ) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / one_hot_y.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims=True) / one_hot_y.shape[1]
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha_min, alpha_max, step_size, itr):
    
    '''
    Implementing the cyclic learning rate (cylic alpha)
    '''
    cyc = np.floor(1 + (itr / (2 * step_size)))
    x = np.abs((itr / step_size) - 2 * cyc + 1)
    alpha = alpha_min + (alpha_max - alpha_min) * np.maximum(0, 1-x)

    '''
    Update Rule
    '''
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2, alpha

def gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha_max, alpha_min, step_size, file):
    '''
    Full Gradient Descent Algorithm
    '''
    one_hot_y = one_hot(y)
    loss_vec = [] # Vector to plot the loss
    acc_vec = [] # Vector to plot the acc
    i_vec = [] # Vector to plot the iterations
    alpha_vec = [] # Vector to plot learning rate over time (iterations)
    
    itr = 0 # num to intialize iteration index
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            z1, a1, z2, a2 = forward(x[i], w1, b1, w2, b2)
            
            loss = CCE(one_hot_y[i], a2)
            acc = accuracy(y[i], a2)
            
            dw1, db1, dw2, db2 = backwards(x[i], one_hot_y[i], w2, a2, a1, z1)
            w1, b1 ,w2, b2, alpha = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha_min, alpha_max, step_size, itr)
            
            print(f"Epoch: {epoch} | Iteration: {i}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {acc}%")
            
            itr += 1
            
            alpha_vec.append(alpha)
            loss_vec.append(loss)
            acc_vec.append(acc)
            i_vec.append(itr)
            
    save_model(file, w1, b1, w2, b2)    
    return w1, b1, w2, b2, loss_vec, acc_vec, i_vec, alpha_vec
    
def model(x, y, epochs, alpha_min, alpha_max, step_size, file):
    '''
    One function model call.
    '''
    try:
        w1, b1, w2, b2 = load_model(file)
        print(f"Loading model!")
    except:
        print(f"Model not found, intializing new model!")
        w1, b1, w2, b2 = init_params()
    
    w1, b1, w2, b2, loss_vec, acc_vec, i_vec, alpha_vec = gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha_max, alpha_min, step_size, file)
    return w1, b1, w2, b2, loss_vec, acc_vec, i_vec, alpha_vec
        
        
if __name__ == "__main__":
    data = pd.read_csv('datasets/fashion-mnist_train.csv')
    data = np.array(data)

    mini_batch_num = 10
    
    X_train, Y_train = minibatches(data, mini_batch_num)
    
    epochs = 1000
    alpha_min = .01
    alpha_max = .3
    step_size = 30
    file = 'models/CyclicNN.pkl'
    
    w1, b1, w2, b2, loss_vec, acc_vec, i_vec, alpha_vec = model(X_train, Y_train, epochs, alpha_min, alpha_max, step_size, file )
    
    # If you want to visualize results, uncomment. I did so to visualize and then gauge what decent set of minimum and maximum bounds for a learning rate could be for this model, through the learning rate range test
    
    '''fig, ax = plt.subplots(3, 1)
    
    ax[0].plot(i_vec, loss_vec)
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Iteration')
    
    ax[1].plot(i_vec, acc_vec)
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Iteration')
    
    ax[2].plot(i_vec, alpha_vec)
    ax[2].set_ylabel('Alpha')
    ax[2].set_xlabel('Iteration')
    
    plt.show()'''
