import torch
import torch.nn as nn
import numpy as np
from torchvision.models import (ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,
                ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights)
from torch.nn import functional as F
# import logging
from .point_conv import PointConv
from .mlp import Conv2dNormRelu, MLP1d
from .utils import project_pc2image, build_pc_pyramid_single, se3_transform, knn_interpolation
from .csrc import correlation2d
from .clfm import CLFM
# from .embedding import PoseEmbedding
# from mmdet.models.backbones import ResNet
from typing import Literal, List, Dict, Tuple
from functools import partial

def get_activation_func(activation:Literal['leakyrelu','relu','elu','gelu'], inplace:bool) -> nn.Module:
    if activation == 'leakyrelu':
        activation_func = nn.LeakyReLU(0.1, inplace=inplace)
    elif activation == 'relu':
        activation_func = nn.ReLU(inplace=inplace)
    elif activation == 'elu':
        activation_func = nn.ELU(inplace=inplace)
    elif activation == 'gelu':
        activation_func = nn.GELU()
    return activation_func

class DepthImgGenerator:
    def __init__(self, pooling_size=1, max_depth=50.0):
        assert (pooling_size-1) % 2 == 0, 'pooling size must be odd to keep image size constant'
        if pooling_size == 1:
            self.pooling = lambda x:x
        else:
            self.pooling = torch.nn.MaxPool2d(kernel_size=pooling_size,stride=1,padding=(pooling_size-1)//2)
        self.max_depth = max_depth
        # InTran (3,4) or (4,4)

    @torch.no_grad()
    def project(self, pcd:torch.Tensor, pcd_norm:torch.Tensor, camera_info:Dict)->torch.Tensor:
        """transform point cloud to image

        Args:
            pcd (torch.Tensor): (B, 3, N)
            pcd_norm (torch.Tensor): (B, N) distance of each point from the original point
            camera_info (Dict): project information

        Returns:
            torch.Tensor: depth image (B, 1, H, W)
        """
        B = pcd.shape[0]
        uv = project_pc2image(pcd, camera_info)
        proj_x = uv[:,0,:].type(torch.long)
        proj_y = uv[:,1,:].type(torch.long)
        H, W = camera_info['sensor_h'], camera_info['sensor_w']
        rev = ((proj_x>=0)*(proj_x<W)*(proj_y>=0)*(proj_y<H)*(pcd[:,2,:]>0)).type(torch.bool)  # [B,N]
        batch_depth_img = torch.zeros(B,H,W,dtype=torch.float32).to(pcd.device)  # [B,H,W]
        # size of rev_i is not constant so that a batch-formed operdation cannot be applied
        for bi in range(B):
            rev_i = rev[bi,:]  # (N,)
            proj_xrev = proj_x[bi,rev_i]
            proj_yrev = proj_y[bi,rev_i]
            batch_depth_img[bi*torch.ones_like(proj_xrev),proj_yrev,proj_xrev] = pcd_norm[bi, rev_i] / self.max_depth # z
        return batch_depth_img.unsqueeze(1)   # (B,1,H,W)
    
    def __call__(self,ExTran:torch.Tensor,pcd:torch.Tensor):
        """transform pcd and project it to img

        Args:
            ExTran (torch.Tensor): B,4,4
            pcd (torch.Tensor): B,3,N

        Returns:
            tuple: depth_img (B,H,W), transformed_pcd (B,3,N)
        """
        assert len(ExTran.size()) == 3, 'ExTran size must be (B,4,4)'
        assert len(pcd.size()) == 3, 'pcd size must be (B,3,N)'
        return self.transform(ExTran,pcd)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, padding=1,
                 dilation=1, activation_fnc=nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride,
                             padding=padding, dilation=dilation,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activate_func = activation_fnc
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate_func(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activate_func(out)

        return out

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers:Literal[18, 34, 50, 101, 152], pretrained:bool):
        super(ResnetEncoder, self).__init__()

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

    def forward(self, x):
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
    
class Encoder2D(nn.Module):
    def __init__(self, depth:Literal[18, 34, 50, 101, 152]=18, out_chan_list:List[int]=[64, 96, 128, 192], pretrained:bool=True):
        super().__init__()
        self.resnet = ResnetEncoder(depth, pretrained)
        assert len(out_chan_list) <= 5, 'too many channels, should no more than 5'
        in_chan_list = [64] + [self.resnet.encoder.base_width * 2 ** i for i in range(4)]
        self.out_chan = out_chan_list
        in_chan_list = in_chan_list[-len(out_chan_list):]
        self.align_list = nn.ModuleList([Conv2dNormRelu(in_chan_list[i], out_chan) for i, out_chan in enumerate(out_chan_list)])
    
    def forward(self, x) -> List[torch.Tensor]:
        xs:List = self.resnet(x)  # List of features
        xs.pop(1)  # xs[1] obtained from xs[0] by max pooling
        feats = []
        num_layers = len(self.align_list)
        assert num_layers <= len(xs), 'too many align layers, should no more than {}'.format(len(xs))
        for xi, align in zip(xs[-num_layers:], self.align_list):
            feat = align(xi)
            feats.append(feat)
        return feats  # List of [B, C, H, W]

class Encoder3D(nn.Module):
    def __init__(self, n_channels:List[int], pcd_pyramid:List[int], embed_norm:bool=False, norm=None, k=16):
        super().__init__()
        assert len(pcd_pyramid)  == len(n_channels), "length of n_channels ({}) != length of pcd_pyramid ({})".format(len(n_channels), len(pcd_pyramid))
        self.pyramid_func = partial(build_pc_pyramid_single, n_samples_list=pcd_pyramid)
        in_chan = 4 if embed_norm else 3
        self.embed_norm = embed_norm
        self.level0_mlp = MLP1d(in_chan, [n_channels[0], n_channels[0]])
        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.n_chans = n_channels
        for i in range(len(n_channels) - 1):
            self.mlps.append(MLP1d(n_channels[i], [n_channels[i], n_channels[i + 1]]))
            self.convs.append(PointConv(n_channels[i + 1], n_channels[i + 1], norm=norm, k=k))

    def forward(self, pcd:torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """pcd hierchical encoding

        Args:
            pcd (torch.Tensor): B, 3, N

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: feats, xyzs
        """
        xyzs, _ = self.pyramid_func(pcd)
        inputs = xyzs[1]  # [bs, 3, n_points]
        if self.embed_norm:
            norm = torch.linalg.norm(inputs, dim=1, keepdim=True)
            inputs = torch.cat([inputs, norm], dim=1)
        feats = [self.level0_mlp(inputs)]

        for i in range(1,len(xyzs) - 1):
            feat = self.mlps[i-1](feats[-1])
            feat = self.convs[i-1](xyzs[i], feat, xyzs[i + 1])
            feats.append(feat)
        return feats, xyzs  
    
class CorrelationNet(nn.Module):
    def __init__(self, corr_dist:int, planes:int, activation:str, inplace:bool):
        super().__init__()
        activation_func = get_activation_func(activation, inplace)
        corr_dim = (2 * corr_dist + 1) ** 2
        self.corr_block = partial(correlation2d, max_displacement=corr_dist, cpp_impl=True)
        self.corr_conv = BasicBlock(corr_dim, planes, activation_fnc=activation_func)
    def forward(self, img1:torch.Tensor, img2:torch.Tensor):
        corr = self.corr_block(img1, img2)  # (B, D, H, W)
        corr = self.corr_conv(corr)  # (B, C, H, W)
        return corr


    
    
class FusionNet(nn.Module):
    def __init__(self, resnet_depth:Literal[18, 34, 50, 101, 152],
                resnet_pretrained:bool,
                encoder_2d_chans:List[int]=[16, 32, 64, 128],
                encoder_3d_chans:List[int]=[16, 32, 64, 128],
                poolings:List[Tuple[int,int]] = [[4,8],[4,8],[2,4],[2,4]],
                fuse_chans:List[int] = [1024, 1024, 1024, 1024],
                pcd_pyramid:List[int]=[4096, 2048, 1024, 512],
                fusion_fn:Literal['add','concat','gated','sk']='concat',
                fusion_knn:int=3,
                ):
        assert len(encoder_2d_chans) == len(fuse_chans) == len(poolings)
        assert len(encoder_3d_chans) == len(pcd_pyramid)
        assert len(encoder_2d_chans) <= len(encoder_3d_chans)
        super().__init__()
        self.fnet_2d = Encoder2D(resnet_depth, encoder_2d_chans, pretrained=resnet_pretrained)
        self.fnet_3d = Encoder3D(encoder_3d_chans, pcd_pyramid, norm='batch_norm', k=16)
        num_levels = len(encoder_2d_chans)
        clfm_list = []
        mlp_net_list = []
        pool2d_list = []
        start_dim_3d = len(encoder_3d_chans) - num_levels
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        for i in range(num_levels):
            clfm_list.append(CLFM(self.fnet_2d.out_chan[i], self.fnet_3d.n_chans[i+start_dim_3d], fusion_fn=fusion_fn, norm='batch_norm', fusion_knn=fusion_knn))
            pool2d_list.append(nn.AdaptiveAvgPool2d(poolings[i]))
            inplanes = self.fnet_2d.out_chan[i] * poolings[i][0] * poolings[i][1] + self.fnet_3d.n_chans[i+start_dim_3d]
            mlp_net_list.append(nn.Linear(inplanes, fuse_chans[i]))
        self.clfm_list = nn.ModuleList(clfm_list)
        self.mlp_net_list = nn.ModuleList(mlp_net_list)
        self.pool2d_list = nn.ModuleList(pool2d_list)
        self.out_dim = sum(fuse_chans)
        self.feat_buffer = dict()
    
    def clear_buffer(self):
        self.feat_buffer.clear()

    def restore_buffer(self, image:torch.Tensor, pcd:torch.Tensor):
        assert self.training == False, 'Cannot set buffer during training'
        feat_2ds = self.fnet_2d(image)
        feat_3ds, xyzs = self.fnet_3d(pcd)
        self.feat_buffer['feat_2ds'] = feat_2ds
        self.feat_buffer['feat_3ds'] = feat_3ds
        self.feat_buffer['xyzs'] = xyzs

    def get_buffer(self):
        return self.feat_buffer['feat_2ds'], self.feat_buffer['feat_3ds'], self.feat_buffer['xyzs']
    
    def forward(self, image:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict):
        """fusion net embedding

        Args:
            image (torch.Tensor): B, C, H, W
            pcd (torch.Tensor): B, D, N
            Tcl (torch.Tensor): B, 4, 4
            camera_info (Dict): intran information for projection

        Returns:
            torch.Tensor: unique features for each query (Tcl)
        """
        if len(self.feat_buffer.keys()) == 0:
            feat_2ds = self.fnet_2d(image)
            feat_3ds, xyzs = self.fnet_3d(pcd)
        else:
            feat_2ds, feat_3ds, xyzs = self.get_buffer()
        # B = image.shape[0]
        aggregated_feats = []
        num_levels = len(feat_2ds)
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        for i, (feat2d, feat3d, xyz) in enumerate(zip(feat_2ds, feat_3ds[-num_levels:], xyzs[-num_levels:])):
            xyz_tf = se3_transform(Tcl, xyz)  # (B, 3, N) -> (B, 3, N)
            feat_camera_info = camera_info.copy()
            feat_w, feat_h = feat2d.shape[-1], feat2d.shape[-2]
            kx = feat_w / sensor_w
            ky = feat_h / sensor_h
            feat_camera_info.update({
                'sensor_w': feat_w,
                'sensor_h': feat_h,
                'fx': kx * camera_info['fx'],
                'fy': ky * camera_info['fy'],
                'cx': kx * camera_info['cx'],
                'cy': kx * camera_info['cy']
            })
            uv = project_pc2image(xyz_tf, feat_camera_info)  # (B, 2, N)
            fused_2d, fused_3d = self.clfm_list[i](uv, feat2d, feat3d)  # (B, C, H, W)
            feat_2d_vec = self.pool2d_list[i](fused_2d)  # (B, C, H, W)
            feat_2d_vec = torch.flatten(feat_2d_vec, start_dim=1) # (B, C, H, W) -> (B,C*H*W)
            feat_3d_vec = self.pool1d(fused_3d).squeeze(-1)
            final_feat = self.mlp_net_list[i](torch.cat([feat_2d_vec, feat_3d_vec], dim=1)) # (B,D)
            aggregated_feats.append(final_feat)
        return torch.cat(aggregated_feats, dim=-1)  # (B, C1 + C2 + C3 + C4)
    