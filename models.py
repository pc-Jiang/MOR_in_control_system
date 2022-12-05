import torch
import numpy as np
from DynamicSystem import Linear_Dynamic_System, Dynamic_System
from copy import deepcopy

class AE_linear(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
         
        self.encoder = torch.nn.Linear(input_dim, output_dim)
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Linear(output_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AE_nonlinear(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
            torch.nn.ReLU()
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(output_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded