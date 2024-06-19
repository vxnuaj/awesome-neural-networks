'''

Implementation of Logistic Regression on dataset: "datasets/logistic_regression_dataset.csv"

'''


import numpy as np
import pandas as pd

def init_params():
    '''
    Initializing model parameters, w and b.

    Using np.random.default_rng(seed = 1) for reproducibility if need be (https://numpy.org/doc/stable/reference/random/generator.html)
    '''
    rng = np.random.default_rng(seed = 1)    
    w = rng.normal( size = (1,2))
    b = np.zeros((1, 1))
    return w, b

def sigmoid(z):
    '''
    Defining the sigmoid activation function.
    Adding `eps` to avoid instability.
    '''
    eps = 1e-10
    a = 1 / (1 + np.exp(-z + eps))
    return a

def forward(x, w, b):
    '''
    Defining the forward pass of the model.
    Takes in X (input features), W (initialized weights), and B (initialized bias).
    '''
    z = np.dot(w, x) + b
    a = sigmoid(z)
    return z, a

def acc(y, a):
    '''
    Computing the average accuracy of each forward pass over the total number of samples
    '''
    pred = np.around(a).astype(int)
    accuracy = np.mean(y == pred) * 100
    return accuracy

def BCE(y, a):
    '''
    Takes in Y (the true labels) and A (aka `Y_hat`, the final predictions).
    Outputs the averaged loss over all samples for a given forward pass
    Using eps to avoid instablity
    '''
    eps = 1e-8
    loss = - np.mean( y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps) )
    return loss

def backward(x, y, a):
    '''
    Comuting gradients `dw` and `db` for the weight update, averaged over all samples
    '''
    dw = np.dot((a - y), x.T) / y.size 
    db = np.sum(a - y) / y.size
    return dw, db

def update(w, b, dw, db, alpha):
    '''
    Performing the weight update, rate of learning dependent on alpha
    '''
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def gradient_descent(x, y, w, b, alpha, epochs):
    '''
    Performing the gradient deescent, for number of epochs specified
    '''
    for epoch in range(epochs):
        z, a = forward(x, w, b)
        
        loss = BCE(y, a) 
        accuracy = acc(y, a)

        dw, db = backward(x, y, a)
        w, b = update(w, b, dw, db, alpha)


        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {accuracy}%\n")
    
    return w, b

def model(x, y, alpha, epochs):
    '''
    Initializing parameters, running gradient descent
    '''
    w, b = init_params()
    w, b = gradient_descent(x, y, w, b, alpha, epochs)
    return w, b

if __name__ == "__main__":
    data = pd.read_csv('datasets/logistic_regression_dataset.csv')
    data = np.array(data) # 1000, 3
    
    X_train = data[:, 0:-1].T # Slicing and Transposing to dimensions -> (features, samples) | (2, 1000) | FEATURES
    Y_train = data[:, -1].reshape(1, -1) # Slicing and Reshaping 1D array (303, ) -> 2D array of dimensions (1,, samples) | (1, 1000) | LABELS
    
    w, b = model(X_train, Y_train, alpha = .1, epochs = 10000)
