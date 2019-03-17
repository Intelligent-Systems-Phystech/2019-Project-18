import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def plot_res(y_test, y_pred, outfile='../report/pls_results.pdf'):
    plt.rc('font', size=30)
    axes = ['$x$', '$y$', '$z$']
    plt.figure(figsize=(30, 30))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.ylabel(axes[i], rotation=0, labelpad=20)
        plt.xlabel('$t$', labelpad=20)
        plt.plot(y_test[0:1000,i], label='наблюдаемое движение', linewidth=3.0)
        plt.plot(y_pred[0:1000,i], label='предсказание', linewidth=3.0)
        plt.grid(True)
        plt.legend(loc='upper right')
    if outfile == None:
        plt.show()
    else:
        plt.savefig(outfile)

def pls_regression(n_comps, X_train, X_test, Y_train, Y_test, plot=None):
    
    pls = PLSRegression(n_components=n_comps)
    pls.fit(X_train, Y_train)
    Y_pred = pls.predict(X_test)
    print('n_comps = {}; mae = {}, mse = {}, r2 = {}'.format(
            n_comps, 
            mean_absolute_error(Y_test, Y_pred),
            mean_squared_error(Y_test, Y_pred),
            r2_score(Y_test, Y_pred)))
    
    if plot != None:
        plot_res(Y_test, Y_pred)

    return Y_pred