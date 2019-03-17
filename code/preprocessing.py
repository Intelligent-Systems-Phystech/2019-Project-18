import os
import numpy as np
from scipy.io import loadmat

def flatten(data):
    data = np.array(data)
    N = data.shape[0]
    M = data.size//N
    return data.reshape(N,M)

def normalize_data(X_train, X_test, Y_train, Y_test):
    mean_x_train = np.mean(X_train, axis=0)
    std_x_train = np.std(X_train, axis=0)
    mean_y_train = np.mean(Y_train, axis=0)
    std_y_train = np.std(Y_train, axis=0)

    X_train = (X_train-mean_x_train)/(std_x_train+0.001)
    Y_train = (Y_train-mean_y_train)/(std_y_train+0.001)
    X_test = (X_test-mean_x_train)/(std_x_train+0.001)
    Y_test = (Y_test-mean_y_train)/(std_y_train+0.001)

    return X_train, X_test, Y_train, Y_test

def load_data():
    assert os.path.exists('../data/ECoG_X_test.mat'), 'Data directory should contain train and test mats'

    X_train = loadmat('../data/ECoG_X_train.mat')
    X_test = loadmat('../data/ECoG_X_test.mat')
    Y_train = loadmat('../data/ECoG_Y_train.mat')
    Y_test = loadmat('../data/ECoG_Y_test.mat')

    X_train, X_test, Y_train, Y_test = X_train['X_train'], X_test['X_hold_out'],\
                                         Y_train['Y_train'], Y_test['Y_hold_out']
    return normalize_data(flatten(X_train), flatten(X_test), Y_train, Y_test)

