import torch.nn as nn
import torch
from typing import Literal, Dict, List

from ..Modules import resnet18 as custom_resnet18
from ..tools.core import MLPNet, get_activation_func
from .vae import VanillaVAE as VAE
from ..tools.core import ResnetEncoder

class RGGNet(nn.Module):
    def __init__(self,
            vae_path:str,
            vae_argv:Dict,
            resnet_argv:Dict,
            mlp_head_dims:List[int]=[512],
            mlp_sub_dims:List[int]=[512],
            activation:Literal['leakyrelu','relu','elu','gelu']='leakyrelu',
            inplace:bool=True) -> None:
        super().__init__()
        self.vae = VAE(**vae_argv)
        self.vae_img_size = vae_argv.get('img_size')
        self.vae.load_state_dict(torch.load(vae_path, map_location='cpu')['model'])
        self.vae.requires_grad_(False)
        activation_fn = get_activation_func(activation, inplace)
        self.img_encoder = ResnetEncoder(**resnet_argv)
        self.depth_encoder = custom_resnet18(inplanes=1, planes=64)
        inplanes = self.img_encoder.num_ch_enc[-1] + self.depth_encoder.out_chans[-1]
        self.pooling = nn.AdaptiveMaxPool2d((1,1))
        self.mlp = MLPNet([inplanes] + mlp_head_dims, mlp_sub_dims + [3], activation_fn)
        self.buffer = dict()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, 0.1)

    def forward(self, rgb:torch.Tensor, depth:torch.Tensor):
        if len(self.buffer.keys()) == 0:
            x1 = self.img_encoding(rgb)
        else:
            x1 = self.get_buffer()
        x2 = self.depth_encoder(depth)[-1]
        x = torch.cat([x1, x2], dim=1)
        x = torch.flatten(self.pooling(x), start_dim=1)  # (B, D)
        x = self.mlp(x)
        return x
    
    def compute_ELBO(self, rgb:torch.Tensor, depth:torch.Tensor, kld_weight:float):
        x_input = torch.cat([rgb, depth], dim=1)
        x_est, mu, log_var = self.vae(x_input)
        ELBO = self.vae.loss_function(depth, x_est, mu, log_var, kld_weight)
        return ELBO

    def img_encoding(self, rgb:torch.Tensor):
        x1 = self.img_encoder(rgb)[-1]
        return x1

    def restore_buffer(self, img:torch.Tensor):
        x1 = self.img_encoding(img)
        self.buffer['x1'] = x1

    def clear_buffer(self):
        self.buffer.clear()

    def get_buffer(self):
        return self.buffer['x1']

