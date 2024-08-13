import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from .point_conv import PointConv
from .mlp import Conv2dNormRelu, MLP1d
from .utils import project_pc2image, build_pc_pyramid_single
from .clfm import CLFM_2D
from .embedding import PoseEmbedding
from mmdet.models.backbones import ResNet
from typing import Literal, List, Dict, Tuple
from functools import partial

def se3_transform(g: torch.Tensor, pcd: torch.Tensor):
    # g : SE(3),  * x 4 x 4
    # a : R^3,    * x 3[x N]
    return g[...,:3,:3] @ pcd + g[...,:3,[3]]

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
    
class Encoder2D(ResNet):
    def __init__(self, depth:Literal[18, 34, 50, 101, 152]=18, out_chan_list:List[int]=[64, 96, 128, 192], pretrained=None):
        # for debugging purposes, please comment the following lines.
        from mmcv.utils.logging import get_logger
        get_logger('root').setLevel(logging.ERROR)
        get_logger('mmcv').setLevel(logging.ERROR)
        get_logger('mmengine').setLevel(logging.ERROR)
        super().__init__(
            depth=depth,
            init_cfg=dict(
                type='Pretrained',
                checkpoint=pretrained
            )
        )
        planes = [self.base_channels * 2 ** i for i in range(self.num_stages)]
        self.out_chan = out_chan_list
        self.align_list = nn.ModuleList([Conv2dNormRelu(planes[i], out_chan) for i, out_chan in enumerate(out_chan_list)])
        self.init_weights()
    
    def forward(self, x) -> List[torch.Tensor]:
        xs = super().forward(x)
        feats = []
        for xi, align in zip(xs, self.align_list):
            feat = align(xi)
            feats.append(feat)
        return feats  # List of [B, C, H, W]
    
class FusionNet(nn.Module):
    def __init__(self, resnet_depth:Literal[18, 34, 50, 101, 152]=18,
                resnet_pretrained:str="pretrained/resnet18-5c106cde.pth",
                encoder_2d_chans:List[int]=[64, 96, 128, 192],
                encoder_3d_chans:List[int]=[64, 96, 128, 192],
                pcd_pyramid:List[int]=[4096, 2048, 1024],
                ):
        assert len(encoder_2d_chans) == len(encoder_3d_chans) == len(pcd_pyramid) + 1
        super().__init__()
        self.fnet_2d = Encoder2D(resnet_depth, encoder_2d_chans, pretrained=resnet_pretrained)
        self.fnet_3d = Encoder3D(encoder_3d_chans, pcd_pyramid, norm='batch_norm', k=16)
        num_levels = len(encoder_3d_chans)
        clfm_list = []
        for i in range(num_levels):
            clfm_list.append(CLFM_2D(self.fnet_2d.out_chan[i], self.fnet_3d.n_chans[i], norm='batch_norm'))
        self.clfm_list = nn.ModuleList(clfm_list)
        self.out_dim = sum(encoder_2d_chans)
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
        if len(self.feat_buffer.keys()) == 0:
            feat_2ds = self.fnet_2d(image)
            feat_3ds, xyzs = self.fnet_3d(pcd)
        else:
            feat_2ds, feat_3ds, xyzs = self.get_buffer()
        aggregated_feats = []
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        for i, (feat2d, feat3d, xyz) in enumerate(zip(feat_2ds, feat_3ds, xyzs)):
            uv = project_pc2image(se3_transform(Tcl, xyz), camera_info)
            uv[..., 0] *= (feat2d.shape[-1] - 1) / (sensor_w - 1)
            uv[..., 1] *= (feat2d.shape[-2] - 1) / (sensor_h - 1)
            fused_2d = self.clfm_list[i](uv, feat2d, feat3d)  # (B, C, H, W)  - > (B, C, 1, 1)
            fused_feat = torch.flatten(F.adaptive_avg_pool2d(fused_2d, (1,1)), start_dim=1)  # # (B, C, 1, 1) -> (B, C)
            aggregated_feats.append(fused_feat)
        return torch.cat(aggregated_feats, dim=-1)  # (B, C1 + C2 + C3 + C4)