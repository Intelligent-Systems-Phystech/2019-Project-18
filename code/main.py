from pls import plot_res, pls_regression
import numpy as np

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = load_mats()
    X_train = norm_dist_local_model(X_train)
    X_test = norm_dist_local_model(X_test)

    y_pred = test_pls(2, X_train, y_train, X_test, y_test)
    plot_res(y_test, y_pred, '../report/main_algo.pdf')

    for n_comps in range(3, 101):
        test_pls(n_comps, X_train, y_train, X_test, y_test)
