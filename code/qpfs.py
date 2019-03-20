import numpy as np
import sklearn.feature_selection as sklfs
import scipy as sc
import cvxpy as cvx


def get_corr_matrix(X, Y=None, fill=0):
    if Y is None:
        Y = X
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    
    X_ = (X - X.mean(axis=0))
    Y_ = (Y - Y.mean(axis=0))
    
    idxs_nz_x = np.where(np.sum(X_ ** 2, axis = 0) != 0)[0]
    idxs_nz_y = np.where(np.sum(Y_ ** 2, axis = 0) != 0)[0]
    
    X_ = X_[:, idxs_nz_x]
    Y_ = Y_[:, idxs_nz_y]
    
    corr = np.ones((X.shape[1], Y.shape[1])) * fill
    
    for i, x in enumerate(X_.T):
        corr[idxs_nz_x[i], idxs_nz_y] = Y_.T.dot(x) / np.sqrt(np.sum(x ** 2) * np.sum(Y_ ** 2, axis=0, keepdims=True))
    return corr


def shift_spectrum(Q, eps=0.):
    lamb_min = sc.linalg.eigh(Q)[0][0]
    if lamb_min < 0:
        Q = Q - (lamb_min - eps) * np.eye(*Q.shape)
    return Q, lamb_min


class QPFS:
    def __init__(self, sim='corr'):
        if sim not in ['corr', 'info']:
            raise ValueError('Similarity measure should be "corr" or "info"')
        self.sim = sim
    
    def get_params(self, X, y):
        self.m, self.n = X.shape
        self.r = y.shape[1] if len(y.shape) > 1 else 1
        if self.sim == 'corr':
            self.Q = np.abs(get_corr_matrix(X, fill=1))
            self.b = np.sum(np.abs(get_corr_matrix(X, y)), axis=1)[:, np.newaxis]
        elif self.sim == 'info':
            self.Q = np.ones([X.shape[1], X.shape[1]])
            self.b = np.zeros((X.shape[1], 1))
            for j in range(n_features):
                self.Q[:, j] = sklfs.mutual_info_regression(X, X[:, j])
            if len(y.shape) == 1:
                self.b = sklfs.mutual_info_regression(X, y)[:, np.newaxis]
            else:
                for y_ in y:
                    self.b += sklfs.mutual_info_regression(X, y_)
        self.Q, self.lamb_min = shift_spectrum(self.Q)
    
    def get_alpha(self):
        return self.Q.mean() / (self.Q.mean() + self.b.mean())

    def fit(self, X, y):
        self.get_params(X, y)
        alpha = self.get_alpha()
        self.solve_problem(alpha)
    
    def solve_problem(self, alpha):
        z = cvx.Variable(self.n)
        c = np.ones((self.n, 1))
        objective = cvx.Minimize((1 - alpha) * cvx.quad_form(z, self.Q) - 
                                 alpha * self.b.T * z)
        constraints = [z >= 0, c.T * z == 1]
        prob = cvx.Problem(objective, constraints)

        prob.solve()

        self.status = prob.status
        self.zx = np.array(z.value).flatten()
        self.zy = 1. / self.r * np.ones(self.r)
        
    def get_topk_indices(self, k=10):
        return self.score.argsort()[::-1][:k]


class MultivariateQPFS():
    def __init__(self):
        pass
    
    def get_params(self, X, Y):
        self.m, self.n = X.shape
        self.r = Y.shape[1] if len(Y.shape) > 1 else 1
        
        self.Qx = np.abs(get_corr_matrix(X, fill=1))
        self.Qy = np.abs(get_corr_matrix(Y, fill=1))
        self.B = np.abs(get_corr_matrix(X, Y))
        self.b = self.B.max(axis=0, keepdims=True)

    def get_alpha(self, mode, alpha3=None):
        if mode in ['SymImp', 'MaxMin', 'MinMax', 'MaxRel', 'MinMax2']:
            den = (np.mean(self.Qx) * np.mean(self.B)
                   + np.mean(self.Qx) * np.mean(self.Qy)
                   + np.mean(self.Qy) * np.mean(self.B))
            if alpha3 is None:
                alpha3 = np.mean(self.Qx) * np.mean(self.B) / den

            alpha1 = (1 - alpha3) * np.mean(self.B) / (np.mean(self.Qx) + np.mean(self.B))
            alpha2 = (1 - alpha3) * np.mean(self.Qx) / (np.mean(self.Qx) + np.mean(self.B))
        elif mode == 'AsymImp':
            den = (np.mean(self.Qx) * (np.mean(self.b) - np.mean(self.B))
                   + np.mean(self.Qx) * np.mean(self.Qy)
                   + np.mean(self.Qy) * np.mean(self.B))
            if alpha3 is None:
                alpha3 = np.mean(self.Qx) * (np.mean(self.b) - np.mean(self.B)) / den

            alpha1 = (1 - alpha3) * np.mean(self.B) / (np.mean(self.Qx) + np.mean(self.B))
            alpha2 = (1 - alpha3) * np.mean(self.Qx) / (np.mean(self.Qx) + np.mean(self.B))
        else:
            raise ValueError('Unknown mode: '+ mode)
        return np.array([alpha1, alpha2, alpha3])


    def fit(self, X, Y, mode='SymImp'):
        self.get_params(X, Y)
        alphas = self.get_alpha(mode)
        self.solve_problem(alphas, mode)
    
    def solve_problem(self, alphas, mode='SymImp'):
        if mode == 'SymImp':
            self._symimp(alphas)
        elif mode == 'MaxMin':
            self._maxmin(alphas)
        elif mode == 'MinMax':
            self._minmax(alphas)
        elif mode == 'MaxRel':
            self._maxrel(alphas)
        elif mode == 'AsymImp':
            self._asymimp(alphas)
        else:
            raise ValueError('Unknown mode: '+ mode)
    
    def _symimp(self, alphas):
        # Parameters
        Q = np.vstack((np.hstack((alphas[0] * self.Qx, -alphas[1] / 2 * self.B)),
                       np.hstack(( -alphas[1] / 2 * self.B.T, alphas[2] * self.Qy))))
        Q, lamb_min = shift_spectrum(Q)
        
        c = np.zeros((2, self.n + self.r))
        c[0, :self.n] = 1
        c[1, self.n:] = 1
        
        # Problem
        z = cvx.Variable(self.n + self.r)
        obj = cvx.Minimize(cvx.quad_form(z, Q))
        constr = [z >= 0, c * z == 1]
        prob = cvx.Problem(obj, constr)
        prob.solve()

        # Results
        score = np.array(z.value).flatten()
        self.status = prob.status
        self.zx = score[:self.n]
        self.zy = score[-self.r:]
    
    def _minmax(self, alphas):
        # Parameters
        Qyinv = np.linalg.pinv(self.Qy)
        Q1 = alphas[1] ** 2 * self.B.dot(Qyinv).dot(self.B.T) + 4 * alphas[0] * alphas[2] * self.Qx
        Q2 = Qyinv
        Q3 = Qyinv.sum(keepdims=True)
        Q12 = alphas[1] * self.B.dot(Qyinv)
        Q13 = alphas[1] * Q12.sum(axis=1, keepdims=True)
        Q23 = Qyinv.sum(axis=1, keepdims=True)
        
        Q = np.vstack([
            np.hstack([Q1, -Q12, -Q13]),
            np.hstack([-Q12.T, Q2, Q23]),
            np.hstack([-Q13.T, Q23.T, Q3])
        ])
        
        Q, lamb_min = shift_spectrum(Q)
        
        c = np.zeros(self.n + self.r + 1)
        c[:self.n] = 1
        
        # Problem
        zx = cvx.Variable(self.n + self.r + 1)
        obj = cvx.Minimize(cvx.quad_form(zx, Q) - 4 * alphas[2] * zx[-1])
        constr = [zx[:-1] >= 0, c * zx == 1]
        prob = cvx.Problem(obj, constr)
        prob.solve(solver=cvx.CVXOPT)
        
        # Results
        score = np.array(zx.value).flatten()
        self.status = prob.status
        self.zx = score[:self.n]
        self.zy = 1. / (2 * alphas[2]) * Qyinv.dot(-alphas[1] * self.B.T.dot(self.zx) + 
                                                   score[-1] + score[self.n:-1])
    
    def _maxmin(self, alphas):
        # Parameters
        Qxinv = np.linalg.pinv(self.Qx)
        Q1 = alphas[1] ** 2 * self.B.T.dot(Qxinv).dot(self.B) + 4 * alphas[0] * alphas[2] * self.Qy
        Q2 = Qxinv
        Q3 = Qxinv.sum(keepdims=True)
        Q12 = alphas[1] * self.B.T.dot(Qxinv)
        Q13 = alphas[1] * Q12.sum(axis=1, keepdims=True)
        Q23 = Qxinv.sum(axis=1, keepdims=True)
        
        Q = np.vstack([
            np.hstack([Q1, Q12, Q13]),
            np.hstack([Q12.T, Q2, Q23]),
            np.hstack([Q13.T, Q23.T, Q3])
        ])
        
        Q, lamb_min = shift_spectrum(Q)
        
        c = np.zeros(self.n + self.r + 1)
        c[:self.r] = 1
        
        # Problem
        zy = cvx.Variable(self.n + self.r + 1)
        obj = cvx.Minimize(cvx.quad_form(zy, Q) - 4 * alphas[0] * zy[-1])
        constr = [zy[:-1] >= 0, c * zy == 1]
        prob = cvx.Problem(obj, constr)

        prob.solve(solver=cvx.CVXOPT)
        
        # Results
        self.status = prob.status
        self.zy = np.array(zy.value).flatten()[:self.r]
        #----------------------
        # Parameters
        Q, lamb_min = shift_spectrum(self.Qx)
        
        c = np.ones(self.n)
        
        # Problem
        zx = cvx.Variable(self.n)
        objective = cvx.Minimize(cvx.quad_form(zx, Q) - self.B.dot(self.zy).flatten() * zx)
        constraints = [zx >= 0, c * zx == 1]
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.CVXOPT)
        
        # Results
        self.zx = np.array(zx.value).flatten()
    
    def _maxrel(self, alphas):
        # Parameters
        C = np.zeros((self.r + 1, self.n + self.r + 1))
        C[0, :self.n] = 1
        C[1:, :self.n] = alphas[1] * self.B.T
        C[1:, self.n:self.n + self.r] = -np.eye(self.r)
        C[1:, self.n + self.r] = -1
        d = np.zeros(self.r + 1)
        d[0] = 1
        
        Q, lamb_min = shift_spectrum(self.Qx, eps=1e-9)
        
        # Problem
        z = cvx.Variable(self.n + self.r + 1)
        obj = cvx.Minimize(alphas[0] * cvx.quad_form(z[:self.n], Q) - z[-1])
        constr = [z[:-1] >= 0, C * z == d]
        prob = cvx.Problem(obj, constr)
        prob.solve(solver=cvx.CVXOPT)
        
        # Results
        self.status = prob.status
        self.zx = np.array(z.value).flatten()[:self.n]
        self.zy = None
    
    def _minmax2(self, alphas):
        # Parameters
        c = np.ones(self.r)
        
        Q, lamb_min = shift_spectrum(self.Qy, eps=1e-9)
        
        # Problem
        zy = cvx.Variable(self.r)
        obj = cvx.Minimize(alphas[2] * cvx.quad_form(zy, Q) + 
                           alphas[1] * self.B.mean(axis=0) * zy)
        constr = [zy >= 0, c * zy == 1]
        prob = cvx.Problem(obj, constr)
        prob.solve(solver=cvx.CVXOPT)
        
        # Results
        self.zy = np.array(zy.value).flatten()
        #-------------------------------------
        # Parameters
        c = np.ones(self.n)
        
        Q, lamb_min = shift_spectrum(self.Qx, eps=1e-9)
        
        # Problem
        zx = cvx.Variable(self.n)
        obj = cvx.Minimize(alphas[0] * cvx.quad_form(zx, Q) - 
                           alphas[1] * self.B.dot(self.zy).flatten() * zx)
        constr = [zx >= 0, c * zx == 1]
        prob = cvx.Problem(obj, constr)
        prob.solve(solver=cvx.CVXOPT)
        
        # Results
        self.status = prob.status
        self.zx = np.array(zx.value).flatten()
        self.zy = np.array(zy.value).flatten()
    
    def _asymimp(self, alphas):
        # Parameters
        Q = np.vstack((np.hstack((alphas[0] * self.Qx, -alphas[1] / 2 * self.B)),
                       np.hstack((-alphas[1] / 2 * self.B.T, alphas[2] * self.Qy))))
        Q, lamb_min = shift_spectrum(Q)
        
        c = np.zeros((2, self.n + self.r))
        c[0, :self.n] = 1
        c[1, self.n:] = 1
        
        # Problem
        z = cvx.Variable(self.n + self.r)
        obj = cvx.Minimize(cvx.quad_form(z, Q) + alphas[1] * self.b * z[self.n:])
        constr = [z >= 0, c * z == 1]
        prob = cvx.Problem(obj, constr)
        prob.solve()

        # Results
        score = np.array(z.value).flatten()
        self.status = prob.status
        self.zx = score[:self.n]
        self.zy = score[-self.r:]
    
    def get_topk_indices(self, k=10):
        return self.zx.argsort()[::-1][:k]