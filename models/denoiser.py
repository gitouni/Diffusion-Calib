from .tools.core import FusionNet
from .util import se3
from .util.seq_utils import transformer_encoder_wrapper
import torch.nn as nn
import torch
from typing import Dict, Sequence, Tuple



class Surrogate(nn.Module):
    def __init__(self, hidden_dims:Sequence[int], x_dim:int, encoder_argv:Dict) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.encoder = FusionNet(**encoder_argv)
        mlps = [nn.Linear(self.encoder.out_dim, hidden_dims[0]),
                nn.ReLU(inplace=True)]
        for i in range(len(hidden_dims) -1):
            mlps.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            mlps.append(nn.ReLU(inplace=True))
        mlps.append(nn.Linear(hidden_dims[-1], x_dim))
        self.mlps = nn.Sequential(*mlps)

    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict):
        feat = self.encoder(img, pcd, Tcl, camera_info)  # (B, D)
        x0 = self.mlps(feat)
        return x0  # (B, x_dim)
    
    def restore_buffer(self, x_cond:Tuple[torch.Tensor, torch.Tensor]):
        img, pcd = x_cond
        self.encoder.restore_buffer(img, pcd)

    def clear_buffer(self):
        self.encoder.clear_buffer()
    
class SeqSurrogate(nn.Module):
    def __init__(self, hidden_dims:Sequence[int], x_dim:int, encoder_argv:Dict, atten_argv:Dict) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.encoder = FusionNet(**encoder_argv)
        d_model = self.encoder.out_dim
        self.atten = transformer_encoder_wrapper(d_model=d_model, **atten_argv)
        self.reg_token = nn.Parameter(torch.rand(1, 1, d_model))
        mlps = [nn.Linear(d_model, hidden_dims[0]),
                nn.ReLU(inplace=True)]
        for i in range(len(hidden_dims) -1):
            mlps.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            mlps.append(nn.ReLU(inplace=True))
        mlps.append(nn.Linear(hidden_dims[-1], x_dim))
        self.mlps = nn.Sequential(*mlps)

    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict):
        B, K, C, H, W = img.shape
        B, K, _, N = pcd.shape
        Tcl_expand = Tcl.unsqueeze(1).expand(B, K, 4, 4)
        feat = self.encoder(img.view(B*K, C, H, W), pcd.view(B*K,-1,N), Tcl_expand.view(B*K,4,4), camera_info)  # (B*K, D)
        feat = self.atten(torch.cat([feat.view(B, K, -1), self.reg_token.expand(B, 1, -1)], dim=1))  # (B, K+1, D)
        x0 = self.mlps(feat[:,-1,:])  # the final token has cross attention with other tokens
        return x0  # (B, x_dim)
    
class Denoiser(nn.Module):
    def __init__(self, model:Surrogate):
        super().__init__()
        self.model = model

    def restore_buffer(self, x_cond:Tuple[torch.Tensor, torch.Tensor]):
        img, pcd = x_cond
        self.model.restore_buffer(img, pcd)

    def clear_buffer(self):
        self.model.clear_buffer()

    def forward(self, x_t:torch.Tensor, x_cond:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        img, pcd, Tcl, camera_info = x_cond
        se3_x_t = se3.exp(x_t)
        Tcl = se3_x_t @ Tcl
        delta_x0 = self.model(img, pcd, Tcl, camera_info)
        x0 = se3.exp(delta_x0) @ se3_x_t  # sequentially transformed by se3_x_t and x0
        return se3.log(x0)
    
class SeqDenoiser(nn.Module):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x_t:torch.Tensor, x_cond:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        img, pcd, Tcl, camera_info = x_cond  # (B, K, C, H, W),  (B, K, 3, N), Dict
        K = pcd.shape[1]
        se3_x_t = se3.exp(x_t) # (B, 4, 4)
        Tcl = se3_x_t @ Tcl
        delta_x0 = self.model(img, pcd, Tcl, camera_info)
        x0 = se3.exp(delta_x0) @ se3_x_t  # (B, 4, 4)
        return se3.log(x0)  # (B, 6)
    
# class GuidanceSampler:
#     def __init__(self, corr_data):
