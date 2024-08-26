import torch
import torch.nn as nn
import numpy as np
from torchvision.models import (ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,
                ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights)
from torch.nn import functional as F
# import logging
from .point_conv import PointConv
from .mlp import Conv2dNormRelu, MLP1d
from .utils import project_pc2image, build_pc_pyramid_single, se3_transform
from .csrc import correlation2d
from .clfm import CLFM_2D
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
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        features = [x]
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features
    
class Encoder2D(nn.Module):
    def __init__(self, depth:Literal[18, 34, 50, 101, 152]=18, out_chan_list:List[int]=[64, 96, 128, 192], pretrained:bool=True):
        super().__init__()
        self.resnet = ResnetEncoder(depth, pretrained)
        in_chan_list = [self.resnet.encoder.base_width * 2 ** i for i in range(4)]
        self.out_chan = out_chan_list
        self.align_list = nn.ModuleList([Conv2dNormRelu(in_chan_list[i], out_chan) for i, out_chan in enumerate(out_chan_list)])
    
    def forward(self, x) -> List[torch.Tensor]:
        xs = self.resnet(x)  # List of features
        feats = []
        for xi, align in zip(xs, self.align_list):
            feat = align(xi)
            feats.append(feat)
        return feats  # List of [B, C, H, W]

class Encoder3D(nn.Module):
    def __init__(self, n_channels:List[int], pcd_pyramid:List[int], norm=None, k=16):
        super().__init__()
        assert len(pcd_pyramid) + 1 == len(n_channels), "length of n_channels ({}) != length of pcd_pyramid ({})".format(len(n_channels), len(pcd_pyramid))
        self.pyramid_func = partial(build_pc_pyramid_single, n_samples_list=pcd_pyramid)
        self.level0_mlp = MLP1d(3, [n_channels[0], n_channels[0]])
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
        inputs = xyzs[0]  # [bs, 3, n_points]
        feats = [self.level0_mlp(inputs)]

        for i in range(len(xyzs) - 1):
            feat = self.mlps[i](feats[-1])
            feat = self.convs[i](xyzs[i], feat, xyzs[i + 1])
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
                fusion_knn:int=3,
                ):
        assert len(encoder_2d_chans) == len(encoder_3d_chans) == len(pcd_pyramid) + 1
        super().__init__()
        self.fnet_2d = Encoder2D(resnet_depth, encoder_2d_chans, pretrained=resnet_pretrained)
        self.fnet_3d = Encoder3D(encoder_3d_chans, pcd_pyramid, norm='batch_norm', k=16)
        num_levels = len(encoder_3d_chans)
        clfm_list = []
        mlp_net_list = []
        pooling_list = []
        for i in range(num_levels):
            clfm_list.append(CLFM_2D(self.fnet_2d.out_chan[i], self.fnet_3d.n_chans[i], norm='batch_norm', fusion_knn=fusion_knn))
            pooling_list.append(nn.AdaptiveAvgPool2d(poolings[i]))
            inplanes = self.fnet_2d.out_chan[i] * poolings[i][0] * poolings[i][1]
            mlp_net_list.append(nn.Linear(inplanes, fuse_chans[i]))
        self.clfm_list = nn.ModuleList(clfm_list)
        self.mlp_net_list = nn.ModuleList(mlp_net_list)
        self.pooling_list = nn.ModuleList(pooling_list)
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
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        for i, (feat2d, feat3d, xyz) in enumerate(zip(feat_2ds, feat_3ds, xyzs)):
            xyz_tf = se3_transform(Tcl, xyz)  # (B, 3, N) -> (B, 3, N)
            uv = project_pc2image(xyz_tf, camera_info)  # (B, 2, N)
            uv[..., 0] *= (feat2d.shape[-1] - 1) / (sensor_w - 1)  # normalize x
            uv[..., 1] *= (feat2d.shape[-2] - 1) / (sensor_h - 1)  # normalize y
            fused_2d = self.clfm_list[i](uv, feat2d, feat3d)  # (B, C, H, W)
            final_feat = self.pooling_list[i](fused_2d)  # (B, C, H, W)
            final_feat = torch.flatten(final_feat, start_dim=1) # (B, C, H, W) -> (B,C*H*W)
            final_feat = self.mlp_net_list[i](final_feat) # (B,D)
            aggregated_feats.append(final_feat)
        return torch.cat(aggregated_feats, dim=-1)  # (B, C1 + C2 + C3 + C4)
    
    def logit_forward(self, image:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict) -> torch.Tensor:
        """fusion net probability embedding

        Args:
            image (torch.Tensor): B, C, H, W
            pcd (torch.Tensor): B, D, N
            Tcl (torch.Tensor): B, K, 4, 4
            camera_info (Dict): intran information for projection

        Returns:
            torch.Tensor: unique features for each query (Tcl)
        """
        def expand_dim(x:torch.Tensor, K:int):
            feat_shape = x.shape[1:]
            B = x.shape[0]
            return x.unsqueeze(1).expand(-1,K,*feat_shape).reshape(B*K, *feat_shape)
        B,K = Tcl.shape[:2]
        feat_2ds = self.fnet_2d(image)  # List of [B, D, H, W]
        feat_3ds, xyzs = self.fnet_3d(pcd)  # List of [B, D, N], [B, 3, N]
        aggregated_feats = []
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        Tcl_merged = Tcl.view(B*K, 4, 4)
        for i, (feat2d, feat3d, xyz) in enumerate(zip(feat_2ds, feat_3ds, xyzs)):
            transformed_xyz = se3_transform(Tcl_merged, expand_dim(xyz, K))  # (B*K, 3, N)
            uv = project_pc2image(transformed_xyz, camera_info)  # (B*K, 2, N)
            uv[..., 0] *= (feat2d.shape[-1] - 1) / (sensor_w - 1)  # normalize coordinates
            uv[..., 1] *= (feat2d.shape[-2] - 1) / (sensor_h - 1)
            fused_2d = self.clfm_list[i](uv, expand_dim(feat2d, K), expand_dim(feat3d,K))  # (B*K, C, H, W)
            fused_feat = torch.flatten(F.adaptive_avg_pool2d(fused_2d, (1,1)), start_dim=1).reshape(B, K, -1)  # # (B*K, C, 1, 1) -> (B*K, C) -> (B, K, C)
            aggregated_feats.append(fused_feat)
        return torch.cat(aggregated_feats, dim=-1)  # (B, K, C1 + C2 + C3 + C4)