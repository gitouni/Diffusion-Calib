import torch
import torch.nn as nn
import numpy as np
from torchvision.models import (ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,
                ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights)
from torch.nn import functional as F
# import logging
from .point_conv import PointConv
from .mlp import MLP1d
from .utils import project_pc2image, build_pc_pyramid_single, se3_transform
from .csrc import correlation2d
from ..pointgpt import load_pointgpt
from .clfm import FusionAwareInterp
from ..Modules import resnet18 as custom_resnet
from ..Modules import BottleneckBlock, ResidualBlock, FeatureEncoder
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
    def project(self, pcd:torch.Tensor, camera_info:Dict)->torch.Tensor:
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
            batch_depth_img[bi*torch.ones_like(proj_xrev),proj_yrev,proj_xrev] = pcd[bi, 2, rev_i] / self.max_depth # z
        return batch_depth_img.unsqueeze(1)   # (B,1,H,W)
    
    @staticmethod
    @torch.no_grad()
    def binary_project(pcd:torch.Tensor, camera_info:Dict)->torch.Tensor:
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
        batch_mask_img = torch.zeros(B,H,W,dtype=torch.float32).to(pcd.device)  # [B,H,W]
        # size of rev_i is not constant so that a batch-formed operdation cannot be applied
        for bi in range(B):
            rev_i = rev[bi,:]  # (N,)
            proj_xrev = proj_x[bi,rev_i]
            proj_yrev = proj_y[bi,rev_i]
            batch_mask_img[bi*torch.ones_like(proj_xrev),proj_yrev,proj_xrev] = 1 # z
        return batch_mask_img.unsqueeze(1)   # (B,1,H,W)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, padding=1,
                 dilation=1, activation_fnc=nn.ReLU(), has_downsample:bool=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride,
                             padding=padding, dilation=dilation,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activate_func = activation_fnc
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if has_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))
        else:
            self.downsample = nn.Identity()
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

class SimpleEncoder(nn.Module):
    def __init__(self, block_type:Literal['residualblock','bottleneck'], 
                 norm_type:Literal['instancenorm','batchnorm'],
            in_chan:int, layers:List[int]=[96,96,128,192,256], strides:List[int]=[2,1,2,2]):
        """Simple Encoder for 8x Downsampling

        Args:
            block_type (Literal[&#39;residualblock&#39;,&#39;bottleneck&#39;]): _description_
            norm_type (Literal[&#39;instancenorm&#39;,&#39;batchnorm&#39;]): _description_
            in_chan (int): input channel
            layers (List[int], optional): 5 elements. Defaults to [96,96,128,192,256].
            strides (List[int], optional): 4. Defaults to [2,1,2,2].
        """
        if block_type == 'bottleneck':
            block = BottleneckBlock
        else:
            block = ResidualBlock
        if norm_type == 'instancenorm':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.encoder = FeatureEncoder(block=block, in_chan=in_chan, layers=layers, strides=strides, norm_layer=norm_layer)
        self.downsample_factor = self.encoder.downsample_factor
    def forward(self, x):
        return self.encoder(x)
    
class Encoder2D(nn.Module):
    def __init__(self, depth:Literal[18, 34, 50, 101, 152]=18, pretrained:bool=True, resnet_frozen:bool=False):
        super().__init__()
        self.resnet = ResnetEncoder(depth, pretrained)
        self.resnet.requires_grad_(not resnet_frozen)
        self.out_chans = [64] + [self.resnet.encoder.base_width * 2 ** i for i in range(4)]
    
    def forward(self, x) -> List[torch.Tensor]:
        xs:List = self.resnet(x)  # List of features
        xs.pop(1)  # xs[1] obtained from xs[0] by max pooling
        return xs  # List of [B, C, H, W]

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

class MLPNet(nn.Module):
    def __init__(self, head_dims:List[int], sub_dims:List[int], activation_func:nn.Module):
        super().__init__()
        assert head_dims[-1] == sub_dims[0]
        head_mlps = []
        rot_mlps = []
        tsl_mlps = []
        for i in range(len(head_dims) -1):
            head_mlps.append(nn.Linear(head_dims[i], head_dims[i+1]))
            head_mlps.append(activation_func)
        for i in range(len(sub_dims) - 2):
            rot_mlps.append(nn.Linear(sub_dims[i], sub_dims[i+1]))
            rot_mlps.append(activation_func)
            tsl_mlps.append(nn.Linear(sub_dims[i], sub_dims[i+1]))
            tsl_mlps.append(activation_func)
        rot_mlps.append(nn.Linear(sub_dims[-2], sub_dims[-1]))
        tsl_mlps.append(nn.Linear(sub_dims[-2], sub_dims[-1]))
        self.head_mlps = nn.Sequential(*head_mlps)
        self.rot_mlps = nn.Sequential(*rot_mlps)
        self.tsl_mlps = nn.Sequential(*tsl_mlps)

    def forward(self, x:torch.Tensor):
        x = self.head_mlps(x)
        rot_x = self.rot_mlps(x)
        tsl_x = self.tsl_mlps(x)
        return torch.cat([rot_x, tsl_x],dim=-1)


class FusionNet(nn.Module):
    def __init__(self, resnet_depth:Literal[18, 34, 50, 101, 152],
                resnet_pretrained:bool,
                encoder_3d_chans:List[int]=[64, 128, 256, 512],
                pcd_pyramid:List[int]=[4096, 2048, 1024, 512],
                proj_planes:int = 32,
                max_depth:float = 50.0,
                encoder_3d_knn:int = 16,
                fusion_knn:int=4
                ):
        assert len(encoder_3d_chans) == len(pcd_pyramid) 
        super().__init__()
        self.fnet_2d = Encoder2D(resnet_depth, pretrained=resnet_pretrained)
        self.fnet_3d = Encoder3D(encoder_3d_chans, pcd_pyramid, norm='batch_norm', k=encoder_3d_knn)
        self.depth_gen = DepthImgGenerator(pooling_size=1, max_depth=max_depth)
        self.fnet_proj = custom_resnet(inplanes=1, planes=proj_planes)
        self.fusion = FusionAwareInterp(self.fnet_3d.n_chans[-1], k = fusion_knn, norm='batch_norm')
        planes = self.fnet_2d.out_chans[-1] + encoder_3d_chans[-1] + self.fnet_proj.out_chans[-1]
        self.out_dim = planes
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
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        xyz_tf = se3_transform(Tcl, xyzs[0])  # (B, 3, N) -> (B, 3, N)
        depth_img = self.depth_gen.project(xyz_tf, camera_info)
        feat_projs = self.fnet_proj(depth_img)
        feat_2d, feat_3d, feat_proj, xyz = feat_2ds[-1], feat_3ds[-1], feat_projs[-1], xyzs[-1]
        xyz_tf = se3_transform(Tcl, xyz)  # (B, 3, N) -> (B, 3, N)
        feat_camera_info = camera_info.copy()
        feat_w, feat_h = feat_2d.shape[-1], feat_2d.shape[-2]
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
        interp_2d = self.fusion(uv, feat_2d.shape, feat_3d.detach())  # (B, C, H, W)
        fused_2d = torch.cat([feat_2d, interp_2d, feat_proj], dim=1)  # 512 + 256 + 256
        return fused_2d  # (B, C, h, w)
    
    # def logit_forward(self, image:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict) -> torch.Tensor:
    #     """fusion net probability embedding

    #     Args:
    #         image (torch.Tensor): B, C, H, W
    #         pcd (torch.Tensor): B, D, N
    #         Tcl (torch.Tensor): B, K, 4, 4
    #         camera_info (Dict): intran information for projection

    #     Returns:
    #         torch.Tensor: unique features for each query (Tcl)
    #     """
    #     def expand_dim(x:torch.Tensor, K:int):
    #         feat_shape = x.shape[1:]
    #         B = x.shape[0]
    #         return x.unsqueeze(1).expand(-1,K,*feat_shape).reshape(B*K, *feat_shape)
    #     B,K = Tcl.shape[:2]
    #     feat_2ds = self.fnet_2d(image)  # List of [B, D, H, W]
    #     feat_3ds, xyzs = self.fnet_3d(pcd)  # List of [B, D, N], [B, 3, N]
    #     aggregated_feats = []
    #     sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
    #     Tcl_merged = Tcl.view(B*K, 4, 4)
    #     for i, (feat2d, feat3d, xyz) in enumerate(zip(feat_2ds, feat_3ds, xyzs)):
    #         transformed_xyz = se3_transform(Tcl_merged, expand_dim(xyz, K))  # (B*K, 3, N)
    #         uv = project_pc2image(transformed_xyz, camera_info)  # (B*K, 2, N)
    #         uv[..., 0] *= (feat2d.shape[-1] - 1) / (sensor_w - 1)  # normalize coordinates
    #         uv[..., 1] *= (feat2d.shape[-2] - 1) / (sensor_h - 1)
    #         fused_2d = self.clfm_list[i](uv, expand_dim(feat2d, K), expand_dim(feat3d,K))  # (B*K, C, H, W)
    #         fused_feat = torch.flatten(F.adaptive_avg_pool2d(fused_2d, (1,1)), start_dim=1).reshape(B, K, -1)  # # (B*K, C, 1, 1) -> (B*K, C) -> (B, K, C)
    #         aggregated_feats.append(fused_feat)
    #     return torch.cat(aggregated_feats, dim=-1)  # (B, K, C1 + C2 + C3 + C4)


class FusionNetV2(nn.Module):
    def __init__(self,
                # encoder 2d
                resnet_depth:Literal[18, 34, 50, 101, 152],
                resnet_pretrained:bool,
                resnet_frozen:bool,
                # projection_net
                max_depth:float,
                proj_planes:int,
                # encoder 3d
                pointgpt_config:str,
                pointgpt_checkpoint:str,
                pointgpt_frozen:bool,
                # Interploation Fusion net
                fusion_knn:int=4
                ):
        super().__init__()
        self.fnet_2d = Encoder2D(resnet_depth, pretrained=resnet_pretrained, resnet_frozen=resnet_frozen)
        self.fnet_3d, self.fnet_3d_max_depth = load_pointgpt(pointgpt_config, pointgpt_checkpoint)
        if pointgpt_frozen:
            self.fnet_3d.requires_grad_(False)
        self.depth_gen = DepthImgGenerator(pooling_size=1, max_depth=max_depth)
        self.fnet_proj = custom_resnet(inplanes=1, planes=proj_planes)
        self.fusion = FusionAwareInterp(self.fnet_3d.encoder_dims, k = fusion_knn, norm='batch_norm')
        planes = self.fnet_2d.out_chans[-1] + self.fnet_3d.encoder_dims + self.fnet_proj.out_chans[-1]
        self.out_dim = planes
        self.feat_buffer = dict()
    
    def clear_buffer(self):
        self.feat_buffer.clear()

    @torch.inference_mode()
    def cache_features(self, image:torch.Tensor, pcd:torch.Tensor):
        feat_2d = self.fnet_2d(image)[-1]
        pcd_T = pcd.transpose(-1,-2).contiguous()  # B, N, 3
        xyz, feat_3d = self.fnet_3d.extraction(pcd_T / self.fnet_3d_max_depth)  # feat_3d: B, K, D
        feat_3d = feat_3d.transpose(-1,-2).contiguous()  # B, K, D -> B, D, K
        xyz = xyz.transpose(-1,-2).contiguous() # xyz: B, 3, K
        return dict(feat_2d=feat_2d,
                    feat_3d=feat_3d,
                    xyz=xyz)  # without gradient
    
    def restore_buffer(self, image:torch.Tensor, pcd:torch.Tensor):
        data:Dict[str, torch.Tensor] = self.cache_features(image, pcd)
        self.feat_buffer.update(data)
    
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
            cache = self.cache_features(image, pcd)
            feat_2d, feat_3d, xyz = cache['feat_2d'], cache['feat_3d'], cache['xyz']
        else:
            feat_2d, feat_3d, xyz = self.feat_buffer['feat_2d'], self.feat_buffer['feat_3d'], self.feat_buffer['xyz']
        # B = image.shape[0]
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        xyz_tf = se3_transform(Tcl, xyz)  # (B, 3, N) -> (B, 3, N)
        depth_img = self.depth_gen.project(xyz_tf, camera_info)
        feat_proj = self.fnet_proj(depth_img)[-1]
        feat_camera_info = camera_info.copy()
        feat_w, feat_h = feat_2d.shape[-1], feat_2d.shape[-2]
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
        interp_2d = self.fusion(uv, feat_2d.shape, feat_3d)  # (B, C, H, W)
        fused_2d = torch.cat([feat_2d, interp_2d, feat_proj], dim=1)  # 512 + 256 + 256
        return fused_2d  # (B, C, h, w)

class FusionNetDepthOnly(nn.Module):
    def __init__(self, resnet_depth:Literal[18, 34, 50, 101, 152],
                resnet_pretrained:bool,
                resnet_frozen:bool,
                proj_planes:int = 32,
                max_depth:float = 50.,
                ):
        super().__init__()
        self.fnet_2d = Encoder2D(resnet_depth, pretrained=resnet_pretrained, resnet_frozen=resnet_frozen)
        self.depth_gen = DepthImgGenerator(pooling_size=1, max_depth=max_depth)
        self.fnet_proj = custom_resnet(inplanes=1, planes=proj_planes)
        planes = self.fnet_2d.out_chans[-1] + self.fnet_proj.out_chans[-1]
        self.out_dim = planes
        self.feat_buffer = dict()
    
    def clear_buffer(self):
        self.feat_buffer.clear()

    def restore_buffer(self, image:torch.Tensor, pcd:torch.Tensor):
        assert self.training == False, 'Cannot set buffer during training'
        feat_2ds = self.fnet_2d(image)
        self.feat_buffer['feat_2ds'] = feat_2ds

    def get_buffer(self):
        return self.feat_buffer['feat_2ds']
    
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
        else:
            feat_2ds = self.get_buffer()
        # B = image.shape[0]
        xyz_tf = se3_transform(Tcl, pcd)  # (B, 3, N) -> (B, 3, N)
        depth_img = self.depth_gen.project(xyz_tf, camera_info)
        feat_projs = self.fnet_proj(depth_img)
        feat_2d, feat_proj = feat_2ds[-1], feat_projs[-1]
        fused_2d = torch.cat([feat_2d, feat_proj], dim=1)
        return fused_2d  # (B, C*h*w)


class FusionNetProjectOnly(nn.Module):
    def __init__(self, resnet_depth:Literal[18, 34, 50, 101, 152],
                resnet_pretrained:bool,
                resnet_frozen:bool,
                encoder_3d_chans:List[int]=[64, 128, 256, 512],
                pcd_pyramid:List[int]=[4096, 2048, 1024, 512],
                encoder_3d_knn:int = 16,
                fusion_knn:int=4
                ):
        assert len(encoder_3d_chans) == len(pcd_pyramid) 
        super().__init__()
        self.fnet_2d = Encoder2D(resnet_depth, pretrained=resnet_pretrained, resnet_frozen=resnet_frozen)
        self.fnet_3d = Encoder3D(encoder_3d_chans, pcd_pyramid, norm='batch_norm', k=encoder_3d_knn)
        self.fusion = FusionAwareInterp(self.fnet_3d.n_chans[-1], k = fusion_knn, norm='batch_norm')
        planes = self.fnet_2d.out_chans[-1] + encoder_3d_chans[-1]
        self.out_dim = planes
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
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        xyz_tf = se3_transform(Tcl, xyzs[0])  # (B, 3, N) -> (B, 3, N)
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        xyz_tf = se3_transform(Tcl, xyzs[0])  # (B, 3, N) -> (B, 3, N)
        feat_2d, feat_3d, xyz = feat_2ds[-1], feat_3ds[-1], xyzs[-1]
        xyz_tf = se3_transform(Tcl, xyz)  # (B, 3, N) -> (B, 3, N)
        feat_camera_info = camera_info.copy()
        feat_w, feat_h = feat_2d.shape[-1], feat_2d.shape[-2]
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
        interp_2d = self.fusion(uv, feat_2d.shape, feat_3d.detach())  # (B, C, H, W)
        fused_2d = torch.cat([feat_2d, interp_2d], dim=1)
        return fused_2d  # (B, C, h, w)