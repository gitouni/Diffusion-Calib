# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:39:37 2021

@author: 17478
"""
import torch.nn as nn
from torch.nn import functional as F
import torch
from torchvision.ops import Conv2dNormActivation
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    """
        3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)


class BottleneckBlock(nn.Module):
    """Slightly modified BottleNeck block (extra relu and biases)"""

    def __init__(self, in_channels, out_channels, *, norm_layer, stride=1):
        super().__init__()

        # See note in ResidualBlock for the reason behind bias=True
        self.convnormrelu1 = Conv2dNormActivation(
            in_channels, out_channels // 4, norm_layer=norm_layer, kernel_size=1, bias=True
        )
        self.convnormrelu2 = Conv2dNormActivation(
            out_channels // 4, out_channels // 4, norm_layer=norm_layer, kernel_size=3, stride=stride, bias=True
        )
        self.convnormrelu3 = Conv2dNormActivation(
            out_channels // 4, out_channels, norm_layer=norm_layer, kernel_size=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = Conv2dNormActivation(
                in_channels,
                out_channels,
                norm_layer=norm_layer,
                kernel_size=1,
                stride=stride,
                bias=True,
                activation_layer=None,
            )

    def forward(self, x):
        y = x
        y = self.convnormrelu1(y)
        y = self.convnormrelu2(y)
        y = self.convnormrelu3(y)

        x = self.downsample(x)

        return self.relu(x + y)


class ResidualBlock(nn.Module):
    """Slightly modified Residual block with extra relu and biases."""

    def __init__(self, in_channels, out_channels, *, norm_layer, stride=1, always_project: bool = False):
        super().__init__()

        # Note regarding bias=True:
        # Usually we can pass bias=False in conv layers followed by a norm layer.
        # But in the RAFT training reference, the BatchNorm2d layers are only activated for the first dataset,
        # and frozen for the rest of the training process (i.e. set as eval()). The bias term is thus still useful
        # for the rest of the datasets. Technically, we could remove the bias for other norm layers like Instance norm
        # because these aren't frozen, but we don't bother (also, we woudn't be able to load the original weights).
        self.convnormrelu1 = Conv2dNormActivation(
            in_channels, out_channels, norm_layer=norm_layer, kernel_size=3, stride=stride, bias=True
        )
        self.convnormrelu2 = Conv2dNormActivation(
            out_channels, out_channels, norm_layer=norm_layer, kernel_size=3, bias=True
        )

        # make mypy happy
        self.downsample: nn.Module

        if stride == 1 and not always_project:
            self.downsample = nn.Identity()
        else:
            self.downsample = Conv2dNormActivation(
                in_channels,
                out_channels,
                norm_layer=norm_layer,
                kernel_size=1,
                stride=stride,
                bias=True,
                activation_layer=None,
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = x
        y = self.convnormrelu1(y)
        y = self.convnormrelu2(y)

        x = self.downsample(x)

        return self.relu(x + y)
    
class FeatureEncoder(nn.Module):
    """The feature encoder, used both as the actual feature encoder, and as the context encoder.

    It must downsample its input by 8.
    """

    def __init__(
        self, *, block=ResidualBlock, in_chan=3, layers=(64, 64, 96, 128, 256), strides=(2, 1, 2, 2), norm_layer=nn.BatchNorm2d
    ):
        super().__init__()

        if len(layers) < 3:
            raise ValueError(f"The expected number of layers should over 3, instead got {len(layers)}")

        # See note in ResidualBlock for the reason behind bias=True
        self.convnormrelu = Conv2dNormActivation(
            in_chan, layers[0], norm_layer=norm_layer, kernel_size=7, stride=strides[0], bias=True
        )
        nn_layers = []
        for i in range(len(layers)-2):
            nn_layers.append(self._make_2_blocks(block, layers[i], layers[i+1], norm_layer=norm_layer, first_stride=strides[i+1]))
        self.layers = nn.Sequential(*nn_layers)
        self.conv = nn.Conv2d(layers[-2], layers[-1], kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        num_downsamples = len(list(filter(lambda s: s == 2, strides)))
        self.output_dim = layers[-1]
        self.downsample_factor = 2**num_downsamples

    def _make_2_blocks(self, block, in_channels, out_channels, norm_layer, first_stride):
        block1 = block(in_channels, out_channels, norm_layer=norm_layer, stride=first_stride)
        block2 = block(out_channels, out_channels, norm_layer=norm_layer, stride=1)
        return nn.Sequential(block1, block2)

    def forward(self, x):
        x = self.convnormrelu(x)
        x = self.layers(x)
        x = self.conv(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1,
                 dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride,
                             padding=padding, dilation=dilation,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ConvModule(nn.Module):
    def __init__(self,inplanes, planes, **kwargs):
        super(ConvModule,self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, **kwargs)
        self.bn = nn.BatchNorm2d(planes)
        self.activate = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.activate(x)
        return out
       
class resnet18(nn.Module):
    def __init__(self, inplanes=1, planes=32):
        super(resnet18,self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(inplanes, planes, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(planes, planes, stride=1, padding=1),
            BasicBlock(planes, planes, stride=1, padding=1),
            )
        self.layer2 = nn.Sequential(
            BasicBlock(planes, planes*2, stride=2, padding=1, downsample=nn.Sequential(
                nn.Conv2d(planes, planes*2, 1, stride=2, bias=False),
                nn.BatchNorm2d(planes*2),
                )),
            BasicBlock(planes*2, planes*2, stride=1, padding=1),
            )
        self.layer3 = nn.Sequential(
            BasicBlock(planes*2, planes*4, stride=2, padding=1, downsample=nn.Sequential(
                nn.Conv2d(planes*2, planes*4, 1, stride=2, bias=False),
                nn.BatchNorm2d(planes*4),
                )),
            BasicBlock(planes*4, planes*4, stride=1, padding=1),
            )
        self.layer4 = nn.Sequential(
            BasicBlock(planes*4, planes*8, stride=2, padding=1, downsample=nn.Sequential(
                nn.Conv2d(planes*4, planes*8, 1, stride=2, bias=False),
                nn.BatchNorm2d(planes*8),
                )),
            BasicBlock(planes*8, planes*8, stride=1, padding=1),
            )
        self.out_chans = [planes, planes*2, planes*4, planes*8]
    def forward(self,x):
        out = self.stem(x)
        out = self.maxpool(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1, out2, out3, out4
    
if __name__ == "__main__":
    model = resnet18()
    x = torch.rand(1,3,32,32)
    outs = model(x)
    print(outs[0].size(),outs[1].size(),outs[2].size(),outs[3].size())
        