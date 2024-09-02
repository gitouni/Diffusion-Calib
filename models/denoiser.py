from .tools.core import FusionNet, get_activation_func
from .util import se3
# from .util.seq_utils import transformer_encoder_wrapper
import torch.nn as nn
import torch
from typing import Dict, Iterable, List, Tuple, Literal, Optional, Sequence
from abc import abstractmethod, ABC
from .calibnet.CalibNet import CalibNet as VanillaCalibNet
from .lccnet.LCCNet import LCCNet as VanillaLCCNet
from .tools.core import DepthImgGenerator, BasicBlock


class Surrogate(ABC):
    @abstractmethod
    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict):
        pass

    @abstractmethod
    def restore_buffer(self, img:torch.Tensor, pcd:torch.Tensor):
        pass

    @abstractmethod
    def clear_buffer(self):
        pass

class CalibNet(Surrogate, nn.Module):
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

class LCCNet(Surrogate, nn.Module):
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

class MLPNet(nn.Module):
    def __init__(self, head_dims:Iterable[int], sub_dims:Iterable[int], activation_func:nn.Module):
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


class Aggregation(nn.Module):
    def __init__(self,inplanes:int, planes=96, mlp_dims:Optional[List[int]]=None, final_feat=(2,4), activation_fn:nn.Module=nn.ReLU(inplace=True)):
        super(Aggregation,self).__init__()
        self.head_conv = nn.Sequential(
            BasicBlock(inplanes, planes*4, activation_fnc=activation_fn),
            BasicBlock(planes*4, planes*2, activation_fnc=activation_fn)
        )
        self.rot_conv = nn.Sequential(
            BasicBlock(planes*2, planes, activation_fnc=activation_fn),
            nn.AdaptiveAvgPool2d(output_size=final_feat)
        )
        self.tsl_conv = nn.Sequential(
            BasicBlock(planes*2, planes, activation_fnc=activation_fn),
            nn.AdaptiveAvgPool2d(output_size=final_feat)
        )
        if mlp_dims is None:
            mlp_dims = []
        mlp_dims = [planes*final_feat[0]*final_feat[1]] + mlp_dims + [3]
        rot_fc = []
        tsl_fc = []
        for i in range(len(mlp_dims) - 2):
            rot_fc.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
            rot_fc.append(activation_fn)
            tsl_fc.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
            tsl_fc.append(activation_fn)
        rot_fc.append(nn.Linear(mlp_dims[-2], mlp_dims[-1]))
        tsl_fc.append(nn.Linear(mlp_dims[-2], mlp_dims[-1]))
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
                nn.init.xavier_normal_(m.weight,0.1)

    def forward(self,x:torch.Tensor):
        x = self.head_conv(x)
        x_rot = self.rot_conv(x)
        x_tsl = self.tsl_conv(x)
        x_rot = self.rot_fc(torch.flatten(x_rot, start_dim=1))
        x_tsl = self.tsl_fc(torch.flatten(x_tsl, start_dim=1))
        return torch.cat([x_rot, x_tsl], dim=1)

class ProjFusionNet(Surrogate, nn.Module):
    def __init__(self, activation:Literal['leakyrelu','relu','elu','gelu'], inplace:bool, encoder_argv:Dict, aggregation_argv:Dict) -> None:
        super().__init__()
        activation_func = get_activation_func(activation, inplace)
        self.encoder = FusionNet(**encoder_argv)
        self.mlp = Aggregation(inplanes=self.encoder.out_dim, activation_fn=activation_func, **aggregation_argv)

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
        
__classdict__ = {'ProjFusionNet':ProjFusionNet, 'LCCNet':LCCNet, 'CalibNet':CalibNet}