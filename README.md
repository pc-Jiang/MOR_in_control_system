# Project: Model Order Reduction (MOR) in Control System
### Course: DSC 210: Linear Algebra/Data Science, FA22@UCSD
### Instructor: Dr. Tsui-wei Weng

#### Instructions: 
- Ensure that the following libraries are installed in python 3 environment:
    - numpy
    - torch
    - matplotlib
    - scipy
- Open MOR_in_control_system.ipynb and run all the cells of the notebook.

#### Results: 
We have a linear time-invariant system: 
$$
\mathbf{x}_{t} = \mathbf{A} \mathbf{x}_{t-1} + \mathbf{B} \mathbf{u}_t
\mathbf{y}_t = \mathbf{C} \mathbf{x}_t
$$


- Predictions results of Proper orthognal decomposition (POD) and Autoencoder (AE) with nonlinearity

Predictions on training data, y:


Restored x of training data: 


Predictions on test data: 


Restored x of test data: 

- Predictions results of Proper orthognal decomposition (POD) and Autoencoder (AE) without nonlinearity

Predictions on training data, y:


Restored x of training data: 


Predictions on test data: 


Restored x of test data: 