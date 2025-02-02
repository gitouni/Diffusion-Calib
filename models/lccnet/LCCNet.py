"""
Original implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018
Jinwei Gu and Zhile Ren
Modified version (CMRNet) by Daniele Cattaneo
Modified version (LCCNet) by Xudong Lv
"""

import torch
import numpy as np
from torch.autograd import Variable
# import torch.utils.model_zoo as model_zoo
#from models.CMRNet.modules.attention import *
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import math
# import argparse
import os
import os.path
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from PIL import Image
from typing import Literal, Tuple, Dict
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# from .networks.submodules import *
# from .networks.correlation_package.correlation import Correlation
from .correlation_package.correlation import Correlation
from ..tools.core import ResnetEncoder

# __all__ = [
#     'calib_net'
# ]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.elu = nn.ELU()
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.leakyRELU = nn.LeakyReLU(0.1)
        # self.attention = SCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = SElayer(planes * self.expansion, ratio=reduction)
        # self.attention = ECAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = SElayer_conv(planes * self.expansion, ratio=reduction)
        # self.attention = SCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = ModifiedSCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = DPCSAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = PAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = CAlayer(planes * self.expansion, ratio=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=1, bias=True),
        nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class LCCNet(nn.Module):
    """
    Based on the PWC-DC net. add resnet encoder, dilation convolution and densenet connections
    """

    def __init__(self, resnet_argv:Dict, image_size:Tuple[int,int], use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func:Literal['leakyrelu','relu','elu']='leakyrelu', attention=False):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(LCCNet, self).__init__()
        input_lidar = 1
        self.res_num = resnet_argv['num_layers']
        self.use_feat_from = use_feat_from
        if use_reflectance:
            input_lidar = 2

        self.net_encoder = ResnetEncoder(**resnet_argv)

        # resnet with leakyRELU
        if Action_Func == 'leakyrelu':
            self.activate_func = nn.LeakyReLU(0.1)
        elif Action_Func == 'elu':
            self.activate_func = nn.ELU()
        elif Action_Func == 'relu':
            self.activate_func = nn.ReLU()
        else:
            raise NotImplementedError()
        self.attention = attention
        self.inplanes = 64
        if self.res_num == 50:
            layers = [3, 4, 6, 3]
            add_list = [1024, 512, 256, 64]
        elif self.res_num == 18:
            layers = [2, 2, 2, 2]
            add_list = [256, 128, 64, 64]

        if self.attention:
            block = SEBottleneck
        else:
            if self.res_num == 50:
                block = Bottleneck
            elif self.res_num == 18:
                block = BasicBlock

        # lidar_image
        self.inplanes = 64
        self.conv1_lidar = nn.Conv2d(input_lidar, 64, kernel_size=7, stride=2, padding=3)
        self.elu_lidar = nn.ELU()
        self.leakyRELU_lidar = nn.LeakyReLU(0.1)
        self.relu_lidar = nn.ReLU()
        self.maxpool_lidar = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar = self._make_layer(block, 64, layers[0])
        self.layer2_lidar = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_lidar = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_lidar = self._make_layer(block, 512, layers[3], stride=2)

        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        if use_feat_from > 1:
            self.predict_flow6 = predict_flow(od + dd[4])
            self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

            od = nd + add_list[0] + 4
            self.conv5_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv5_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv5_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv5_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv5_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        if use_feat_from > 2:
            self.predict_flow5 = predict_flow(od + dd[4])
            self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

            od = nd + add_list[1] + 4
            self.conv4_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv4_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv4_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv4_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv4_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        if use_feat_from > 3:
            self.predict_flow4 = predict_flow(od + dd[4])
            self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

            od = nd + add_list[2] + 4
            self.conv3_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv3_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv3_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv3_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv3_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        if use_feat_from > 4:
            self.predict_flow3 = predict_flow(od + dd[4])
            self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

            od = nd + add_list[3] + 4
            self.conv2_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv2_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv2_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv2_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv2_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        if use_feat_from > 5:
            self.predict_flow2 = predict_flow(od + dd[4])
            self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

            self.dc_conv1 = myconv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
            self.dc_conv2 = myconv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
            self.dc_conv3 = myconv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
            self.dc_conv4 = myconv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
            self.dc_conv5 = myconv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
            self.dc_conv6 = myconv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
            self.dc_conv7 = predict_flow(32)

        fc_size = od + dd[4]
        downsample = 128 // (2**use_feat_from)
        if image_size[0] % downsample == 0:
            fc_size *= image_size[0] // downsample
        else:
            fc_size *= (image_size[0] // downsample)+1
        if image_size[1] % downsample == 0:
            fc_size *= image_size[1] // downsample
        else:
            fc_size *= (image_size[1] // downsample)+1
        self.fc1 = nn.Linear(fc_size * 4, 512)

        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256,3)

        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, 0.1)
        self.buffer = dict()

    # def _make_layer(self, block, planes, blocks, stride=1):
    #     layers = []
    #     layers.append(block(self.inplanes, planes, 1))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks - 1):
    #         layers.append(block(self.inplanes, planes, 1))
    #     layers.append(block(self.inplanes, planes, 2))
    #
    #     return nn.Sequential(*layers)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=False)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=False)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        # mask[mask<0.9999] = 0.0
        # mask[mask>0] = 1.0
        mask = torch.floor(torch.clamp(mask, 0, 1))

        return output * mask

    def img_encoding(self, rgb:torch.Tensor):
        features1 = self.net_encoder(rgb)
        c12 = features1[0]  # 2
        c13 = features1[2]  # 4
        c14 = features1[3]  # 8
        c15 = features1[4]  # 16
        c16 = features1[5]  # 32
        return c12, c13, c14, c15, c16
    
    def restore_buffer(self, rgb):
        c12, c13, c14, c15, c16 = self.img_encoding(rgb)
        self.buffer['c12'] = c12
        self.buffer['c13'] = c13
        self.buffer['c14'] = c14
        self.buffer['c15'] = c15
        self.buffer['c16'] = c16

    def get_buffer(self):
        return self.buffer['c12'],self.buffer['c13'],self.buffer['c14'],self.buffer['c15'],self.buffer['c16']
    
    def clear_buffer(self):
        self.buffer.clear()

    def forward(self, rgb, lidar):
        # H, W = rgb.shape[2:4]
        #encoder
        if len(self.buffer.keys()) == 0:
            c12, c13, c14, c15, c16 = self.img_encoding(rgb)
        else:
            c12, c13, c14, c15, c16 = self.get_buffer()

        # lidar_image
        x2 = self.conv1_lidar(lidar)
        c22 = self.activate_func(x2)
        c23 = self.layer1_lidar(self.maxpool_lidar(c22))  # 4
        c24 = self.layer2_lidar(c23)  # 8
        c25 = self.layer3_lidar(c24)  # 16
        c26 = self.layer4_lidar(c25)  # 32

        corr6 = self.corr(c16, c26) #[1, 81, 8, 16]
        corr6 = self.leakyRELU(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)

        if self.use_feat_from > 1:
            flow6 = self.predict_flow6(x)
            up_flow6 = self.deconv6(flow6)
            up_feat6 = self.upfeat6(x)

            warp5 = self.warp(c25, up_flow6*0.625)
            corr5 = self.corr(c15, warp5)
            corr5 = self.leakyRELU(corr5)
            x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
            x = torch.cat((self.conv5_0(x), x), 1)
            x = torch.cat((self.conv5_1(x), x), 1)
            x = torch.cat((self.conv5_2(x), x), 1)
            x = torch.cat((self.conv5_3(x), x), 1)
            x = torch.cat((self.conv5_4(x), x), 1)

        if self.use_feat_from > 2:
            flow5 = self.predict_flow5(x)
            up_flow5 = self.deconv5(flow5)
            up_feat5 = self.upfeat5(x)

            warp4 = self.warp(c24, up_flow5*1.25)
            corr4 = self.corr(c14, warp4)
            corr4 = self.leakyRELU(corr4)
            x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
            x = torch.cat((self.conv4_0(x), x), 1)
            x = torch.cat((self.conv4_1(x), x), 1)
            x = torch.cat((self.conv4_2(x), x), 1)
            x = torch.cat((self.conv4_3(x), x), 1)
            x = torch.cat((self.conv4_4(x), x), 1)

        if self.use_feat_from > 3:
            flow4 = self.predict_flow4(x)
            up_flow4 = self.deconv4(flow4)
            up_feat4 = self.upfeat4(x)

            warp3 = self.warp(c23, up_flow4*2.5)
            corr3 = self.corr(c13, warp3)
            corr3 = self.leakyRELU(corr3)
            x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
            x = torch.cat((self.conv3_0(x), x),1)
            x = torch.cat((self.conv3_1(x), x),1)
            x = torch.cat((self.conv3_2(x), x),1)
            x = torch.cat((self.conv3_3(x), x),1)
            x = torch.cat((self.conv3_4(x), x),1)

        if self.use_feat_from > 4:
            flow3 = self.predict_flow3(x)
            up_flow3 = self.deconv3(flow3)
            up_feat3 = self.upfeat3(x)


            warp2 = self.warp(c22, up_flow3*5.0)
            corr2 = self.corr(c12, warp2)
            corr2 = self.leakyRELU(corr2)
            x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
            x = torch.cat((self.conv2_0(x), x), 1)
            x = torch.cat((self.conv2_1(x), x), 1)
            x = torch.cat((self.conv2_2(x), x), 1)
            x = torch.cat((self.conv2_3(x), x), 1)
            x = torch.cat((self.conv2_4(x), x), 1)

        if self.use_feat_from > 5:
            # flow2 = self.predict_flow2(x)
            x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
            #flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.leakyRELU(self.fc1(x))

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)

        return torch.cat([rot, transl], dim=1)  # (B, 3), (B, 3) -> (B, 6)


