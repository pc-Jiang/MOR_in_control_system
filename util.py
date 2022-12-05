import numpy as np
def mse(x, y):
    # print('in mse', x.shape, y.shape)
    assert x.shape == y.shape
    return np.sum((x - y)**2)/x.size