import numpy as np

import os

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def pls_regression(n_comps, X_train, X_test, Y_train, Y_test):
    
    pls = PLSRegression(n_components=n_comps)
    pls.fit(X_train, Y_train)
    Y_pred = pls.predict(X_test)
    print('n_comps = {}; mae = {}, mse = {}, r2 = {}'.format(
            n_comps, 
            mean_absolute_error(Y_test, Y_pred),
            mean_squared_error(Y_test, Y_pred),
            r2_score(Y_test, Y_pred)))

    return Y_pred