from .tools.core import FusionNet, FusionNetDepthOnly, FusionNetProjectOnly, get_activation_func
from .util import se3
# from .util.seq_utils import transformer_encoder_wrapper
import torch.nn as nn
import torch
from typing import Dict, Callable, List, Tuple, Literal, Optional, Sequence
from abc import abstractmethod
from .calibnet.CalibNet import CalibNet as VanillaCalibNet
from .rggnet.rggnet import RGGNet as VanillaRGGNet
from .lccnet.LCCNet import LCCNet as VanillaLCCNet
from .lccraft.convgru import LCCRAFT as VanillaLCCRAFT
from .tools.core import DepthImgGenerator, BasicBlock, MLPNet
from .tools.utils import timer
from functools import partial

class Surrogate(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @timer.timer_func
    @abstractmethod
    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict, *args):
        pass

    @abstractmethod
    def restore_buffer(self, img:torch.Tensor, pcd:torch.Tensor):
        pass

    @abstractmethod
    def clear_buffer(self):
        pass

class CalibNet(Surrogate):
    def __init__(self, calibnet_argv:Dict, pcd2depth_argv:Dict):
        super().__init__()
        self.encoder = VanillaCalibNet(**calibnet_argv)
        self.pcd2depth = DepthImgGenerator(**pcd2depth_argv)

    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict):
        # pcd_norm = torch.linalg.norm(pcd, dim=1)  # (B, N)
        pcd_tf = se3.transform(Tcl, pcd)
        depth_img = self.pcd2depth.project(pcd_tf, camera_info)
        x0 = self.encoder(img, depth_img)  # (B, D)
        return x0  # (B, x_dim)
    
    def restore_buffer(self, img:torch.Tensor, pcd:torch.Tensor):
        self.encoder.restore_buffer(img)

    def clear_buffer(self):
        self.encoder.clear_buffer()

class RGGNet(Surrogate):
    def __init__(self, rggnet_argv:Dict, pcd2depth_argv:Dict, kld_weight:float, ELBO_weight:float):
        super().__init__()
        self.encoder = VanillaRGGNet(**rggnet_argv)
        self.pcd2depth = DepthImgGenerator(**pcd2depth_argv)
        self.kld_weight = kld_weight
        self.elbo_weight = ELBO_weight
        self.depth_img = None

    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict, store_depth:bool=True):
        # pcd_norm = torch.linalg.norm(pcd, dim=1)  # (B, N)
        pcd_tf = se3.transform(Tcl, pcd)
        depth_img = self.pcd2depth.project(pcd_tf, camera_info)
        x0 = self.encoder(img, depth_img)  # (B, D)
        if store_depth:
            self.depth_img = depth_img
        else:
            self.depth_img = None
        return x0  # (B, x_dim)
    
    def restore_buffer(self, img:torch.Tensor, pcd:torch.Tensor):
        self.encoder.restore_buffer(img)

    def clear_buffer(self):
        self.encoder.clear_buffer()
    
    def loss(self, rgb:torch.Tensor, depth:torch.Tensor):
        ELBO = self.encoder.compute_ELBO(rgb, depth, kld_weight=self.kld_weight)
        return ELBO * self.elbo_weight

class LCCNet(Surrogate):
    def __init__(self, lccnet_argv:Dict, pcd2depth_argv:Dict):
        super().__init__()
        self.encoder = VanillaLCCNet(**lccnet_argv)
        self.pcd2depth = DepthImgGenerator(**pcd2depth_argv)
    
    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict):
        # pcd_norm = torch.linalg.norm(pcd, dim=1)  # (B, N)
        pcd_tf = se3.transform(Tcl, pcd)
        depth_img = self.pcd2depth.project(pcd_tf, camera_info)
        x0 = self.encoder(img, depth_img)  # (B, D)
        return x0  # (B, x_dim)
    
    def restore_buffer(self, img:torch.Tensor, pcd:torch.Tensor):
        self.encoder.restore_buffer(img)

    def clear_buffer(self):
        self.encoder.clear_buffer()

class LCCRAFT(Surrogate):
    def __init__(self, lccraft_argv:Dict, num_iters:int) -> None:
        super().__init__()
        self.encoder = VanillaLCCRAFT(**lccraft_argv)
        self.num_iters = num_iters

    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict):
        # pcd_norm = torch.linalg.norm(pcd, dim=1)  # (B, N)
        pcd_tf = se3.transform(Tcl, pcd)
        x0 = self.encoder(img, pcd_tf, camera_info, self.num_iters) # (B, D)
        return x0  # List of (B, x_dim)
    
    def restore_buffer(self, img:torch.Tensor, pcd:torch.Tensor):
        self.encoder.restore_buffer(img)

    def clear_buffer(self):
        self.encoder.clear_buffer()

    def sequence_loss(self, x_pred_list:List[torch.Tensor], x_gt:torch.Tensor, loss_fn:Callable):
        loss = 0
        gamma_prod = 1.0
        for x_pred in x_pred_list:
            loss += loss_fn(x_pred, x_gt) * gamma_prod
            gamma_prod *= self.encoder.loss_gamma
        return loss


class Aggregation(nn.Module):
    def __init__(self, inplanes=768, planes=96, final_feat=(2,4)):
        super(Aggregation,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes,out_channels=planes*4,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(planes*4)
        self.conv2 = nn.Conv2d(in_channels=planes*4,out_channels=planes*4,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(planes*4)
        self.conv3 = nn.Conv2d(in_channels=planes*4,out_channels=planes*2,kernel_size=1,stride=1)
        self.bn3 = nn.BatchNorm2d(planes*2)
        self.tr_conv = nn.Conv2d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.tr_bn = nn.BatchNorm2d(planes)
        self.rot_conv = nn.Conv2d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.rot_bn = nn.BatchNorm2d(planes)
        self.tr_pool = nn.AdaptiveAvgPool2d(output_size=final_feat)
        self.rot_pool = nn.AdaptiveAvgPool2d(output_size=final_feat)
        self.fc1 = nn.Linear(planes*final_feat[0]*final_feat[1],3) 
        self.fc2 = nn.Linear(planes*final_feat[0]*final_feat[1],3) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        nn.init.xavier_normal_(self.fc1.weight,0.1)
        nn.init.xavier_normal_(self.fc2.weight,0.05)

    def forward(self,x:torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x_tr = self.tr_conv(x)
        x_tr = self.tr_bn(x_tr)
        x_tr = self.tr_pool(x_tr)  # (19,6)
        x_tr = self.fc1(x_tr.view(x_tr.shape[0],-1))
        x_rot = self.rot_conv(x)
        x_rot = self.rot_bn(x_rot)
        x_rot = self.rot_pool(x_rot)  # (19.6)
        x_rot = self.fc2(x_rot.view(x_rot.shape[0],-1))
        return torch.cat([x_rot, x_tr],dim=1)

class ResAggregation(nn.Module):
    def __init__(self,inplanes:int, planes=96, rot_mlp_dims:Optional[List[int]]=None, tsl_mlp_dims:Optional[List[int]]=None, final_feat=(2,4), activation_fn:nn.Module=nn.ReLU(inplace=True)):
        super(ResAggregation,self).__init__()
        self.head_conv = nn.Sequential(
            BasicBlock(inplanes, planes*8, activation_fnc=activation_fn),
            BasicBlock(planes*8, planes*4, activation_fnc=activation_fn)   
        )
        self.rot_conv = nn.Sequential(
            BasicBlock(planes*4, planes*2, activation_fnc=activation_fn),
            BasicBlock(planes*2, planes, activation_fnc=activation_fn),
            nn.AdaptiveAvgPool2d(output_size=final_feat)
        )
        self.tsl_conv = nn.Sequential(
            BasicBlock(planes*4, planes*2, activation_fnc=activation_fn),
            BasicBlock(planes*2, planes, activation_fnc=activation_fn),
            nn.AdaptiveAvgPool2d(output_size=final_feat)
        )
        if rot_mlp_dims is None:
            rot_mlp_dims = []
        if tsl_mlp_dims is None:
            tsl_mlp_dims = []
        rot_mlp_dims = [planes*final_feat[0]*final_feat[1]] + rot_mlp_dims + [3]
        tsl_mlp_dims = [planes*final_feat[0]*final_feat[1]] + tsl_mlp_dims + [3]
        rot_fc = []
        tsl_fc = []
        for i in range(len(rot_mlp_dims) - 2):
            rot_fc.append(nn.Linear(rot_mlp_dims[i], rot_mlp_dims[i+1]))
            rot_fc.append(activation_fn)
        for i in range(len(tsl_mlp_dims) - 2):
            tsl_fc.append(nn.Linear(tsl_mlp_dims[i], tsl_mlp_dims[i+1]))
            tsl_fc.append(activation_fn)
        rot_fc.append(nn.Linear(rot_mlp_dims[-2], rot_mlp_dims[-1]))
        tsl_fc.append(nn.Linear(tsl_mlp_dims[-2], tsl_mlp_dims[-1]))
        self.rot_fc = nn.Sequential(*rot_fc)
        self.tsl_fc = nn.Sequential(*tsl_fc)
        for m in self.head_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.rot_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.tsl_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.rot_fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight,0.1)
        for m in self.tsl_fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight,0.05)

    def forward(self,x:torch.Tensor):
        x = self.head_conv(x)
        x_rot = self.rot_conv(x)
        x_tsl = self.tsl_conv(x)
        x_rot = self.rot_fc(torch.flatten(x_rot, start_dim=1))
        x_tsl = self.tsl_fc(torch.flatten(x_tsl, start_dim=1))
        return torch.cat([x_rot, x_tsl], dim=1)

class ProjFusionNet(Surrogate):
    def __init__(self, activation:Literal['leakyrelu','relu','elu','gelu'], inplace:bool, encoder_argv:Dict, aggregation_argv:Dict,
            proj_features:bool=True, depth_features:bool=True) -> None:
        super().__init__()
        assert proj_features or depth_features, 'at least either should be true.'
        activation_func = get_activation_func(activation, inplace)
        if proj_features and depth_features:
            fusion_class = FusionNet
        elif not proj_features:
            fusion_class = FusionNetDepthOnly
        else:
            fusion_class = FusionNetProjectOnly
        self.encoder = fusion_class(**encoder_argv)
        self.mlp = ResAggregation(inplanes=self.encoder.out_dim, activation_fn=activation_func, **aggregation_argv)

    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict):
        feat = self.encoder(img, pcd, Tcl, camera_info)  # (B, D)
        x0 = self.mlp(feat)
        return x0  # (B, x_dim)
    
    def restore_buffer(self, img:torch.Tensor, pcd:torch.Tensor):
        self.encoder.restore_buffer(img, pcd)

    def clear_buffer(self):
        self.encoder.clear_buffer()

    
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

class RGGDenoiser(nn.Module):
    def __init__(self, model:RGGNet):
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
    
    def loss(self, x0_hat:torch.Tensor, x_cond:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        img, pcd, Tcl, camera_info = x_cond
        Tcl = se3.exp(x0_hat) @ Tcl
        depth = self.model.pcd2depth.project(pcd, camera_info)
        return self.model.loss(img, depth)

class RAFTDenoiser(nn.Module):
    def __init__(self, model:LCCRAFT):
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
        pred_se3_list = self.model(img, pcd, Tcl, camera_info)
        x0_list = [se3.log(se3_x @ se3_x_t) for se3_x in pred_se3_list]  # sequentially transformed by se3_x_t and x0
        return x0_list
    
    def loss(self, loss_fn:Callable, gamma:float):
        return partial(self.model.encoder.sequence_loss, loss_fn=loss_fn, gamma=gamma)

# class ProbNet(nn.Module):
#     def __init__(self, hidden_dims:Sequence[int], encoder_argv:Dict, pretrained_encoder:Optional[OrderedDict]=None, freeze_encoder:bool=False):
#         super().__init__()
#         def load_part_state_dict(submodule:nn.Module, state_dict:OrderedDict):
#             own_state = submodule.state_dict()
#             for name, param in state_dict.items():
#                 if name not in own_state:
#                     continue
#                 if isinstance(param, nn.Parameter):
#                     # backwards compatibility for serialized parameters
#                     param = param.data
#                 own_state[name].copy_(param)
#                 print('\033[33;1mparam {} loaded\033[0m'.format(name))
#         self.encoder = FusionNet(**encoder_argv)
#         if pretrained_encoder is not None:
#             load_part_state_dict(self, torch.load(pretrained_encoder))
#         if freeze_encoder:
#             self.encoder.requires_grad_(False)
#         mlps = [nn.Linear(self.encoder.out_dim, hidden_dims[0]),
#                 nn.ReLU(inplace=True)]
#         for i in range(len(hidden_dims) -1):
#             mlps.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
#             mlps.append(nn.ReLU(inplace=True))
#         mlps.append(nn.Linear(hidden_dims[-1], 1))
#         self.mlp = nn.Sequential(*mlps)
        
    
#     def forward(self, image:torch.Tensor, pcd:torch.Tensor, queries_se3:torch.Tensor, camera_info:Dict, softmax=True):
#         feats = self.encoder.logit_forward(image, pcd, queries_se3, camera_info)  # (B, K, F)
#         logits = self.mlp(feats).squeeze(-1)  # (B, K, 1) -> (B, L)
#         if softmax:
#             logits = torch.softmax(logits, 1)  # (B, K)
#             V = torch.pi**2 / queries_se3.shape[1]
#             return 1 / V * logits
#         else:
#             return logits
        
#     def single_foward(self, image:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict):
#         feats = self.encoder.forward(image, pcd, Tcl, camera_info)  # (B, F)
#         logits = self.mlp(feats).squeeze(-1)  # (B, 1) -> (B,)
#         return logits  # log likelihood

# class SeqDenoiser(nn.Module):
#     def __init__(self, model:nn.Module):
#         super().__init__()
#         self.model = model

#     def forward(self, x_t:torch.Tensor, x_cond:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
#         img, pcd, Tcl, camera_info = x_cond  # (B, K, C, H, W),  (B, K, 3, N), Dict
#         K = pcd.shape[1]
#         se3_x_t = se3.exp(x_t) # (B, 4, 4)
#         Tcl = se3_x_t @ Tcl
#         delta_x0 = self.model(img, pcd, Tcl, camera_info)
#         x0 = se3.exp(delta_x0) @ se3_x_t  # (B, 4, 4)
#         return se3.log(x0)  # (B, 6)
    
    
# class SeqSurrogate(nn.Module):
#     def __init__(self, hidden_dims:Sequence[int], x_dim:int, encoder_argv:Dict, atten_argv:Dict) -> None:
#         super().__init__()
#         self.x_dim = x_dim
#         self.encoder = FusionNet(**encoder_argv)
#         d_model = self.encoder.out_dim
#         self.atten = transformer_encoder_wrapper(d_model=d_model, **atten_argv)
#         self.reg_token = nn.Parameter(torch.rand(1, 1, d_model))
#         mlps = [nn.Linear(d_model, hidden_dims[0]),
#                 nn.ReLU(inplace=True)]
#         for i in range(len(hidden_dims) -1):
#             mlps.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
#             mlps.append(nn.ReLU(inplace=True))
#         mlps.append(nn.Linear(hidden_dims[-1], x_dim))
#         self.mlps = nn.Sequential(*mlps)

#     def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict):
#         B, K, C, H, W = img.shape
#         B, K, _, N = pcd.shape
#         Tcl_expand = Tcl.unsqueeze(1).expand(B, K, 4, 4)
#         feat = self.encoder(img.view(B*K, C, H, W), pcd.view(B*K,-1,N), Tcl_expand.view(B*K,4,4), camera_info)  # (B*K, D)
#         feat = self.atten(torch.cat([feat.view(B, K, -1), self.reg_token.expand(B, 1, -1)], dim=1))  # (B, K+1, D)
#         x0 = self.mlps(feat[:,-1,:])  # the final token has cross attention with other tokens
#         return x0  # (B, x_dim)
        
__classdict__ = {'ProjFusionNet':ProjFusionNet, 'LCCNet':LCCNet, 'CalibNet':CalibNet, 'LCCRAFT':LCCRAFT, 'RGGNet':RGGNet}