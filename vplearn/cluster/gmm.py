from vplearn._model import ModelBaseModel
from vplearn.cluster import Kmeans

from typing import Union

import pandas as pd
import numpy as np

from scipy.stats import multivariate_normal

class GMM(ModelBaseModel):
    def __init__(self):
        super().__init__()
        
    def fit(
        self, 
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        k: int,
        init_mu: list = None,
        init_sigma: np.ndarray = None,
        init_pi: np.ndarray = None,
        num_iters: int = 20,
        reg_covar: float = 1e-6
        ) -> np.ndarray:
        """Define a model with known number of clusters and dimensions.

        Args:
            X (Union[pd.DataFrame, pd.Series, np.ndarray]): data (batch_size, dim)
            k (int): Number of Gaussian clusters
            init_mu (list, optional): initial value of mean of clusters (k, dim). Defaults to None.
            init_sigma (np.ndarray, optional): initial value of covariance matrix of clusters (k, dim, dim). Defaults to None.
            init_pi (np.ndarray, optional): initial value of cluster weights (k,). Defaults to None.
            reg_covar: Regularization added to the diagonal of covariance matrices to prevent singular matrices (default: 1e-6)

        Raises:
            TypeError: k must be an integer.
        """
        self._check_fit_input_format(X)
        
        if not isinstance(k, int):
            raise TypeError('k must be an integer.')
        
        X = self._convert_to_numpy(X)
        
        self.k = k
        self.dim = X.shape[1]
        if init_mu is None:
            kmeans = Kmeans()
            kmeans.fit(X, k = self.k)
            init_mu = kmeans.centers
        self.mu = init_mu
        if init_sigma is None:
            init_sigma = np.zeros((k, self.dim, self.dim))
            for i in range(k):
                init_sigma[i] = np.eye(self.dim)
        self.sigma = init_sigma
        if init_pi is None:
            init_pi = np.ones(self.k) / self.k
        self.pi = init_pi
        self.num_iters = num_iters
        self.reg_covar = reg_covar
        
        self._fit_gaussian_cluster(X)
        
        return self._assign_cluster_for_points()

    def _init_em(self, X):
        '''
        Initialization for EM algorithm.
        input:
            - X: data (batch_size, dim)
        '''
        self.data = X
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))
    
    def _e_step(self):
        '''
        E-step of EM algorithm.
        '''
        for i in range(self.k):
            self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
        self.z /= self.z.sum(axis=1, keepdims=True)
    
    def _m_step(self):
        '''
        M-step of EM algorithm.
        '''
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.num_points
        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None]
        for i in range(self.k):
            j = np.expand_dims(self.data, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i] )
            self.sigma[i] /= sum_z[i]
            # Add regularization to prevent singular covariance matrices
            self.sigma[i] += self.reg_covar * np.eye(self.dim)
            
    def _log_likelihood(self, X):
        '''
        Compute the log-likelihood of X under current parameters
        input:
            - X: Data (batch_size, dim)
        output:
            - log-likelihood of X: Sum_n Sum_k log(pi_k * N( X_n | mu_k, sigma_k ))
        '''
        ll = []
        for d in X:
            tot = 0
            for i in range(self.k):
                tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
            ll.append(np.log(tot))
        return np.sum(ll)

    def _fit_gaussian_cluster(self, X: np.ndarray):
        self._init_em(X)
        log_likelihood = [self._log_likelihood(X)]
        
        for _ in range(self.num_iters):
            self._e_step()
            self._m_step()
            
            log_likelihood.append(self._log_likelihood(X))
        
    def _assign_cluster_for_points(self) -> np.ndarray:
        return np.argmax(self.z, axis = 1)