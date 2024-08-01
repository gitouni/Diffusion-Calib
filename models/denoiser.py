from .tools.core import FusionNet
import torch.nn as nn
import torch
from typing import Dict, Sequence

class Surrogate(nn.Module):
    def __init__(self, hidden_dims:Sequence[int], x_dim:int, encoder_argv:Dict) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.encoder = FusionNet(**encoder_argv)
        mlps = [nn.Linear(self.encoder.out_dim, hidden_dims[0]),
                nn.LeakyReLU()]
        for i in range(len(hidden_dims) -1):
            mlps.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            mlps.append(nn.LeakyReLU())
        mlps.append(nn.Linear(hidden_dims[-1], x_dim))
        self.mlps = nn.Sequential(*mlps)

    def forward(self, img:torch.Tensor, pcd:torch.Tensor, camera_info:Dict):
        feat = self.encoder(img, pcd, camera_info)  # (B, D)
        x0 = self.mlps(feat)
        return x0  # (B, x_dim)
    
