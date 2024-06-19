import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def init_params():
    '''
    Initializing model parameters
    '''
    rng = np.random.default_rng(seed = 1)
    w = rng.normal(size = (3, 4)) # (3, 3)
    b = np.zeros((3, 1)) # (3, 1)
    return w, b

def softmax(z):
    '''
    Computing the softmax
    '''
    z = z.astype(float)
    eps = 1e-8
    a = np.exp(z + eps) / (np.sum(np.exp(z + eps), axis = 0, keepdims = True))
    return a

def forward(x, w, b):
    '''
    Computing the forward pass
    '''
    z = np.dot(w, x) + b # (3, 3) • (3, 1000) -> (3, 1000)
    a = softmax(z) # (3, 1000)
    return z, a

def one_hot(y):
    '''
    Converting labels into one_hot_encodings
    '''
    y_onehot = np.zeros((int(np.max(y) + 1), y.size))
    y_onehot[y.astype(int), np.arange(y.size)] = 1
    return y_onehot

def accuracy(y, a):
    '''
    Computing accuracy
    '''
    pred = np.argmax(a, axis = 0, keepdims=True)
    acc = np.sum(y == pred) / y.size * 100
    return acc

def CCE(y_onehot, a):
    '''
    Computing the categorical cross entropy loss
    '''
    eps = 1e-8
    loss = - np.sum(y_onehot * np.log(a + eps)) / y_onehot.shape[1]
    return loss

def backward(x, y_onehot, a):
    '''
    Computing backward pass
    '''
    dz = a - y_onehot # (3, 1000)
    dw = np.dot(dz, x.T) / y_onehot.shape[1]
    db = np.sum(dz, axis = 1, keepdims=True) / y_onehot.shape[1]
    return dw, db

def update(w, b, dw, db, alpha):
    '''
    Computing update step
    '''
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def gradient_descent(x, y, w, b, epochs, alpha):
    '''
    Computing weight update
    '''
    y_onehot = one_hot(y)
    for epoch in range(epochs):
        z, a = forward(x, w, b)
        
        loss = CCE(y_onehot, a)
        acc = accuracy(y, a)

        dw, db = backward(x, y_onehot, a)
        w, b = update(w, b, dw, db, alpha)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {acc}%\n")
    return w, b

def model(x, y, alpha, epochs):
    w, b = init_params()
    w, b = gradient_descent(x, y, w, b, epochs, alpha)
    return w, b



if __name__ == "__main__":
    data = pd.read_csv('datasets/iris.csv')
    data = np.array(data) # 1000, 4

    X_train = data[:, 0:4].T # 3, 1000
    Y_train = data[:, 4] # 1, 1000

    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train).reshape(1, -1)
    print(Y_train.shape)

    w, b = model(X_train, Y_train, alpha = .1, epochs = 1000)