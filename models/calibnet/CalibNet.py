import torch.nn as nn
import torch
from torchvision.models import (ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,
                ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights)
from typing import Literal, Dict
import numpy as np
from .Modules import resnet18 as custom_resnet18

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


class Aggregation(nn.Module):
    def __init__(self,inplanes=768,planes=96,final_feat=(2,4)):
        super(Aggregation,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes,out_channels=planes*4,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(planes*4)
        self.conv2 = nn.Conv2d(in_channels=planes*4,out_channels=planes*4,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(planes*4)
        self.conv3 = nn.Conv2d(in_channels=planes*4,out_channels=planes*2,kernel_size=(2,1),stride=2)
        self.bn3 = nn.BatchNorm2d(planes*2)
        self.tr_conv = nn.Conv2d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.tr_bn = nn.BatchNorm2d(planes)
        self.rot_conv = nn.Conv2d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.rot_bn = nn.BatchNorm2d(planes)
        self.tr_drop = nn.Dropout2d(p=0.2)
        self.rot_drop = nn.Dropout2d(p=0.2)
        self.tr_pool = nn.AdaptiveAvgPool2d(output_size=final_feat)
        self.rot_pool = nn.AdaptiveAvgPool2d(output_size=final_feat)
        self.fc1 = nn.Linear(planes*final_feat[0]*final_feat[1],3)  # 96*10
        self.fc2 = nn.Linear(planes*final_feat[0]*final_feat[1],3)  # 96*10
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
    def __init__(self, res_num:Literal[18, 34, 50, 101, 152], depth_resnet_argv:Dict, aggregation_argv:Dict):
        super(CalibNet,self).__init__()
        self.rgb_resnet = ResnetEncoder(num_layers=res_num, pretrained=True)  # outplanes = 512
        self.depth_resnet = custom_resnet18(**depth_resnet_argv)
        self.aggregation = Aggregation(inplanes=512+256, **aggregation_argv)
        
    def forward(self,rgb:torch.Tensor,depth:torch.Tensor):
        # rgb: [B,3,H,W]
        # depth: [B,1,H,W]
        x1 = self.rgb_resnet(rgb)[-1]
        x2 = self.depth_resnet(depth)[-1]
        feat = torch.cat((x1, x2),dim=1)  # [B,C1+C2,H,W]
        x_rot, x_tr = self.aggregation(feat)
        return torch.cat([x_rot, x_tr], dim=1)