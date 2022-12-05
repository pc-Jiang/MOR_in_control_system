from DynamicSystem import Linear_Dynamic_System
import os
import numpy.random as nrd
from models import AE_Reduced_System, balanced_truncated
import numpy as np
from util import mse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# define the dimension of the system
dim_x = 50 # dimension of inate state
dim_y = 1 # dimension of output
dim_u = 5 # dimention of input
time_steps = 200 # time steps to be generated
dim_x_reduct = 5 # dimension after MOR


original_sys = Linear_Dynamic_System(dim_x, dim_u, dim_y) # define a system


# randomly generate the input u and x0
u = nrd.rand(dim_u, time_steps)
x0 = nrd.rand(dim_x, 1)

# generate data
x, y = original_sys.gen(x0, u, time_steps)
print("the shape of x: ", x.shape)
print("the shape of y: ", y.shape)

# x_, y_ = original_sys.gen(x0, u, time_steps)
# print(mse(x, x_))

# reconst_x_BT, reconst_y_BT = balanced_truncated(original_sys, x0, u, time_steps, dim_x_reduct)
# print(mse(x, reconst_x_BT))
# print(mse(y, reconst_y_BT))

ae_sys = AE_Reduced_System(original_sys, dim_x_reduct)
reconst_x_AE, reconst_y_AE = ae_sys.fit(x)
print(mse(x, reconst_x_AE))
print(mse(y, reconst_y_AE))

# print(mse(x, np.zeros_like(x)))