import torch.nn as nn
import torch
from torchvision.models import (ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,
                ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights)
from typing import Literal, Dict, List, Tuple
import numpy as np
from ..Modules import resnet18 as custom_resnet18
from ..tools.core import MLPNet, get_activation_func
from .vae import VanillaVAE as VAE

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers:Literal[18, 34, 50, 101, 152], pretrained:bool):
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: (resnet18, ResNet18_Weights.DEFAULT),
                   34: (resnet34, ResNet34_Weights.DEFAULT),
                   50: (resnet50, ResNet50_Weights.DEFAULT),
                   101: (resnet101, ResNet101_Weights.DEFAULT),
                   152: (resnet152, ResNet152_Weights.DEFAULT)}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        func, weights = resnets[num_layers]
        if not pretrained:
            weights = None
        self.encoder:ResNet = func(weights=weights)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, x:torch.Tensor):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        # self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        features.append(self.encoder.maxpool(features[-1]))
        features.append(self.encoder.layer1(features[-1]))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))
        return features
    
class RGGNet(nn.Module):
    def __init__(self,
            vae_path:str,
            vae_argv:Dict,
            resnet_depth:Literal[18, 34, 50, 101, 152]=18,
            resnet_pretrained:bool=True,
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
        self.img_encoder = ResnetEncoder(resnet_depth, resnet_pretrained)
        self.depth_encoder = custom_resnet18(inplanes=1, planes=64)
        inplanes = self.img_encoder.num_ch_enc[-1] + self.depth_encoder.out_chans[-1]
        self.pooling = nn.AdaptiveMaxPool2d((1,1))
        self.mlp = MLPNet([inplanes] + mlp_head_dims, mlp_sub_dims + [3], activation_fn)
        self.buffer = dict()

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

