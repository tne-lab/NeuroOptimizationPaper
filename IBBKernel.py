import math

import gpytorch
import torch


class IBBKernel(gpytorch.kernels.Kernel):
    def __init__(self, ep, beta, **kwargs):
        super().__init__(**kwargs)
        self.ep = ep
        self.beta = beta
        self.eps = 1e-5

    def forward(self, x1, x2, diag=False, **params):
        M = math.ceil(1 / math.pi * math.sqrt(
            self.eps ** (-1 / self.beta) * (x1.shape[0] ** 2 * math.pi ** 2 + self.ep ** 2) - self.ep ** 2))
        n = torch.arange(1, M + 1, device=x1.device, dtype=x1.dtype)
        # Lambda = torch.sparse.spdiags((((n.cpu() * torch.pi) ** 2 + self.ep ** 2) / (torch.pi ** 2 + self.ep ** 2)) ** (-self.beta), torch.tensor([0], device='cpu'), (M, M))
        # Lambda = Lambda.to(x1.device)
        Lambda = torch.diag((((n * torch.pi) ** 2 + self.ep ** 2) / (torch.pi ** 2 + self.ep ** 2)) ** (-self.beta))
        K = None
        for i in range(x1.shape[-1]):
            x1_interp = math.sqrt(2) * torch.sin(torch.pi * x1[..., i:i + 1] * n)
            x2_interp = math.sqrt(2) * torch.sin(torch.pi * x2[..., i:i + 1] * n)
            if K is None:
                K = x1_interp @ Lambda @ torch.transpose(x2_interp, -1, -2)
            else:
                K *= x1_interp @ Lambda @ torch.transpose(x2_interp, -1, -2)
        if diag:
            return torch.diagonal(K)
        K[K < 1e-5] = 1e-5
        return K

