import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def rbfKernel(X, Y, gamma=15):
    pairwiseDists = pairwise_distances(X, Y, metric='sqeuclidean')
    K = np.exp(-gamma * pairwiseDists)
    return K

def polyKernel(X, Y, deg=2):
    K = (1 + X @ Y.T)**deg
    return K

def noKernel(X, Y, **kwargs):
    K = X @ Y.T
    return K

class KernelPCA(object):
    def __init__(self, dims, kernel, offset=0, **kernelKwargs):
        self.dims = dims
        self.dim_offset = offset
        if type(kernel) == str:
            if kernel == 'rbf':
                self.kernel = rbfKernel
            elif kernel == 'poly':
                self.kernel = polyKernel
            elif kernel == None or kernel == 'None':
                self.kernel = noKernel
            else:
                raise ValueError("Error: no valid kernel mentioned")
        else:
            self.kernel = kernel

        self.kernelKwargs = kernelKwargs
    
    def fit(self, X, y=None, **fit_params):
        self.X = X
        n = len(X)

        # calculate the kernel matrix
        K = self.kernel(X, X, **self.kernelKwargs)
            
        # center the kernel matrix
        In = np.ones_like(K) / n # I / n
        K = K - In @ K - K @ In + In @ K @ In

        # Kernel EVD 
        e, U = eigh(K) # eigen values are in ascending order, each column is an eigen vector

        self.U = U[:, ::-1]#[:, :self.dims] # last d eigen vector columns
        e = np.maximum(e, 0) ** .5
        e = e[::-1]#[:self.dims] # last d eigen values
        self.D = np.diag(e)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=None, **fit_params)
        return self.transform(X)

    def transform(self, X):
        K = self.kernel(self.X, X, **self.kernelKwargs)
        D_inv = self.D[self.dim_offset: self.dim_offset + self.dims, self.dim_offset: self.dim_offset + self.dims]
        D_inv = np.linalg.inv(D_inv)
        U = self.U[:, self.dim_offset: self.dim_offset + self.dims]
        return (D_inv @ U.T @ K).T

    def plot_components(self, kernel_name, X_train, X_test, y_train, y_test, fit=True):
        fig, ax = plt.subplots(1, 3, figsize=(24, 8))

        sns.scatterplot(x=X_train[:,0], y=X_train[:,1], hue=y_train, ax=ax[0])
        ax[0].set_xlabel('x1')
        ax[0].set_ylabel('x2')
        ax[0].set_title('Data', fontsize=17)

        if fit:
            P_train = self.fit_transform(X_train)
        else:
            P_train = self.transform(X_train)
        P_test  = self.transform(X_test)

        sns.scatterplot(x=P_train[:,0], y=P_train[:,1], hue=y_train, ax=ax[1])
        ax[1].set_xlabel('x1')
        ax[1].set_ylabel('x2')
        ax[1].set_title(f'Projected training data for {kernel_name}', fontsize=17)


        sns.scatterplot(x=P_test[:,0], y=P_test[:,1], hue=y_test, ax=ax[2])
        ax[2].set_xlabel('x1')
        ax[2].set_ylabel('x2')
        ax[2].set_title(f'Projected test data for {kernel_name}', fontsize=17)
        return P_train, P_test