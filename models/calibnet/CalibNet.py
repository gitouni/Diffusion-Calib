import torch.nn as nn
import torch
from typing import Literal, Dict
import numpy as np
from ..Modules import resnet18 as custom_resnet18
from ..tools.core import ResnetEncoder

class Aggregation(nn.Module):
    def __init__(self,inplanes=768,planes=96,final_feat=(2,4),dropout=0.0):
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
        self.tr_drop = nn.Dropout2d(p=dropout)
        self.rot_drop = nn.Dropout2d(p=dropout)
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
        nn.init.xavier_normal_(self.fc2.weight,0.1)

    def forward(self,x:torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x_tr = self.tr_conv(x)
        x_tr = self.tr_bn(x_tr)
        x_tr = self.tr_drop(x_tr)
        x_tr = self.tr_pool(x_tr)  # (19,6)
        x_tr = self.fc1(x_tr.view(x_tr.shape[0],-1))
        x_rot = self.rot_conv(x)
        x_rot = self.rot_bn(x_rot)
        x_rot = self.rot_drop(x_rot)  
        x_rot = self.rot_pool(x_rot)  # (19.6)
        x_rot = self.fc2(x_rot.view(x_rot.shape[0],-1))
        return x_rot, x_tr

class CalibNet(nn.Module):
    def __init__(self, resnet_argv:Dict, depth_resnet_argv:Dict, aggregation_argv:Dict):
        super(CalibNet,self).__init__()
        self.rgb_resnet = ResnetEncoder(**resnet_argv)  # outplanes = 512
        self.depth_resnet = custom_resnet18(**depth_resnet_argv)
        self.aggregation = Aggregation(inplanes=512+256, **aggregation_argv)
        self.buffer = dict()
        
    def forward(self,rgb:torch.Tensor,depth:torch.Tensor):
        # rgb: [B,3,H,W]
        # depth: [B,1,H,W]
        if len(self.buffer.keys()) == 0:
            x1 = self.img_encoding(rgb)
        else:
            x1 = self.get_buffer()
        x2 = self.depth_resnet(depth)[-1]
        feat = torch.cat((x1, x2),dim=1)  # [B,C1+C2,H,W]
        x_rot, x_tr = self.aggregation(feat)
        return torch.cat([x_rot, x_tr], dim=1)
    
    def img_encoding(self, rgb:torch.Tensor):
        x1 = self.rgb_resnet(rgb)[-1]
        return x1

    def restore_buffer(self, img:torch.Tensor):
        x1 = self.img_encoding(img)
        self.buffer['x1'] = x1

    def clear_buffer(self):
        self.buffer.clear()

    def get_buffer(self):
        return self.buffer['x1']