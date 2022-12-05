# define the linear system
import torch
import time

import numpy as np
import numpy.random as nrd
import numpy.linalg as nla
import matplotlib.pyplot as plt
from scipy.stats import wishart
from copy import deepcopy

class Dynamic_System:
    def __init__(self, dim_x, dim_u, dim_y):
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_y = dim_y

    def init_coef(self):
        pass

    def step(self, x, u):
        pass
    
    def gen(self, x0, u, length):
        pass

    def get_y(self, x):
        pass
    
    
class Linear_Dynamic_System(Dynamic_System):
    '''
    dim_x: int, dimension of inate state of the system
    dim_u: int, dimension of input
    dim_y: int, dimension of observation (output)
    sys_param: dict, parameters of the dynamic matrices and noises' variance
      x_t = Ax_{t-1} + Bu_t + epsilon
      y_t = Cx_t + eta
      sys_param['A'] - matrix on x_{t-1}
      sys_param['B'] - matrix on u_t
      sys_param['C'] - matrix on x_t
      sys_param['Q'] - variance of noise epsilon (Gaussian white noise)
      sys_param['R'] - variance of noise eta (Gaussian white noise)
    '''
    def __init__(self, dim_x, dim_u, dim_y, sys_param={}):
        super().__init__(dim_x, dim_u, dim_y)
        # self.dim_x = dim_x
        # self.dim_u = dim_u
        # self.dim_y = dim_y
        self.sys = self.init_coef(sys_param)
    
    def init_coef(self, sys):
        """
        if not ('R' in sys):
            sys['R'] = wishart.rvs(20, np.eye(self.dim_y, dtype='float32')) / 10
            sys['R'] = np.zeros_like(sys['R'])
        if not ('Q' in sys):
            sys['Q'] = wishart.rvs(200, 0.5 * np.eye(self.dim_x, dtype='float32'))
            sys['Q'] = np.zeros_like(sys['Q'])
        """
        if not ('A' in sys):
            A0 = wishart.rvs(200, 2 * np.eye(self.dim_x, dtype='float32'))
            [_, eig_vec] = nla.eig(A0)
            eig_val = (np.diag(nrd.random(self.dim_x) - 0.5) ) * 2
            # set abs eigenvalues no larger than 1 so that it would not blow up
            eig_val = eig_val.astype(np.float32)
            sys['A'] = eig_vec @ eig_val @ np.transpose(eig_vec)
        if not ('B' in sys): 
            sys['B'] = nrd.randn(self.dim_x, self.dim_u) * 0.1
        if not ('C' in sys):
            sys['C'] = nrd.randn(self.dim_y, self.dim_x) * 0.1
            sys['C'] = sys['C'].astype(np.float32)

        for k, v in sys.items():
            sys[k] = np.array(v, dtype='float32')
        return sys
    
    def step(self, x, u):
        # print()
        x = self.sys['A'] @ x + self.sys['B'] @ u #  + epsilon
        y = self.sys['C'] @ x
        return x, y
    
    def gen(self, x0, u, length):
        '''
        Input: 
          x0 - initial state of the system, shape (dim_x, 1)
          u - stimulus of the system, shape (dim_u, length)
          length - time steps to simulate
        Output:
          x - inate state, shape (dim_x, length)
          y - observation of the system, shape (dim_y, length) 
        '''
        xt = x0
        for i in range(length):
            # epsilon = nrd.multivariate_normal(np.zeros(self.dim_x), self.sys['Q'])
            # epsilon = epsilon.reshape(self.dim_x, -1)
            # if self.dim_y == 1:
            #     eta = nrd.randn(1) * self.sys['R']
            # else:    
            #     eta = nrd.multivariate_normal(np.zeros(self.dim_y), self.sys['R'])
            # eta = eta.reshape(self.dim_y, -1)
            ut = u[:, i].reshape(self.dim_u, -1)
            xt, yt = self.step(xt, ut)
            if i == 0:
                x = xt
                y = yt
            else: 
                x = np.concatenate([x, xt], axis=1)
                y = np.concatenate([y, yt], axis=1)
        return x, y

    def get_y(self, x):
        # print(self.sys['C'].shape, x.shape)
        return self.sys['C'] @ x 


