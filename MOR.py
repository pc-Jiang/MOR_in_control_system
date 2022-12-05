from models import AE_linear, AE_nonlinear
from DynamicSystem import Linear_Dynamic_System, Dynamic_System
import torch
import numpy as np

class AE_Reduced_System(Dynamic_System):
    
    def __init__(self, original_sys, dim_x_reduct, nonlinear=True):
        super().__init__(original_sys.dim_x, original_sys.dim_u, original_sys.dim_y)
        self.original_sys = original_sys
        self.dim_x_reduct = dim_x_reduct
        self.nonlinear = nonlinear

    def fit(self, x):
        tensor_x = torch.from_numpy(x.T).float()

        if self.nonlinear:
            model = AE_nonlinear(tensor_x.shape[1], self.dim_x_reduct)
        else:
            model = AE_linear(tensor_x.shape[1], self.dim_x_reduct)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        
        for epoch in range(2000):
            optimizer.zero_grad()
            outputs = model(tensor_x)
            loss = criterion(outputs, tensor_x)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if epoch % 100 == 0:
                pass
                # print('epoch [{}/{}], loss:{:.4f}'.format(epoch, 1000, loss.item()))

        self.model = model
        reconst_x = self.model(tensor_x).detach().numpy().T  # dim_x * n
        reconst_y = self.original_sys.get_y(reconst_x) # dim_y * n
        return reconst_x, reconst_y 

    
    def step(self, x, u):
        tensor_x = torch.from_numpy(x).reshape(1, -1)  # dim: 1,5
        # compressed_x = self.model.encoder(tensor_x).detach().numpy().T
        reconst_x = self.model.decoder(tensor_x)  # dim: 1, 50
        encoder_jacob = torch.autograd.functional.jacobian(self.model.encoder, reconst_x)[0, :, 0, :]
        # 5, 50
        new_x, new_y = self.original_sys.step(reconst_x.reshape(-1).detach().numpy(), u)
        dx_dt = torch.from_numpy(new_x) - reconst_x.reshape(-1) # dim: 50
        # print(dx_dt.shape, encoder_jacob.shape)
        dx_compress = encoder_jacob @ dx_dt.reshape(-1, 1).float()
        # print(dx_compress.shape)
        new_compress_x = tensor_x + dx_compress.T
        # pred_y = self.original_sys.get_y(self.model.decoder(new_compress_x).reshape(-1))
        return new_compress_x.reshape(-1).detach().numpy(), new_y
        
    def compress(self, x):
        return self.model.encoder(torch.from_numpy(x).float().reshape(1, -1)).reshape(-1).detach().numpy()

    def decompress(self, x):
        return self.model.decoder(torch.from_numpy(x).float().reshape(1, -1)).reshape(-1).detach().numpy()

class POD_Reduced_System(Linear_Dynamic_System):
    def __init__(self, original_sys, dim_x_reduct):
        super().__init__(dim_x_reduct, original_sys.dim_u, original_sys.dim_y)
        self.original = original_sys
    def fit(self, x):
        U, _, _ = np.linalg.svd(x)
        self.Ur = Ur = U[..., :self.dim_x]
        self.sys = self.init_coef({'A': Ur.T @ self.original.sys['A'] @ Ur,
                                   'B': Ur.T @ self.original.sys['B'],
                                   'C': self.original.sys['C'] @ Ur})
        rec_x = Ur @ Ur.T @ x
        rec_y = self.original.sys['C'] @ rec_x
        return rec_x, rec_y
    def step(self, x, u):
        return super().step(x, u)
    def compress(self, x):
        return self.Ur.T @ x
    def decompress(self, x):
        return self.Ur @ x
