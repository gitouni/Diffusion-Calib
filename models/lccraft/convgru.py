from typing import List, Optional, Tuple, Dict, Callable, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from ..tools.core import MLPNet, DepthImgGenerator
from ..tools.utils import project_pc2image, furthest_point_sampling, batch_indexing, knn_interpolation
from ..util import se3

def grid_sample(img: torch.Tensor, absolute_grid: torch.Tensor, mode: str = "bilinear", align_corners: Optional[bool] = None):
    """Same as torch's grid_sample, with absolute pixel coordinates instead of normalized coordinates."""
    h, w = img.shape[-2:]

    xgrid, ygrid = absolute_grid.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (w - 1) - 1
    # Adding condition if h > 1 to enable this function be reused in raft-stereo
    if h > 1:
        ygrid = 2 * ygrid / (h - 1) - 1
    normalized_grid = torch.cat([xgrid, ygrid], dim=-1)

    return F.grid_sample(img, normalized_grid, mode=mode, align_corners=align_corners)


def make_coords_grid(batch_size: int, h: int, w: int, device: str = "cpu"):
    device = torch.device(device)
    coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij") # [(H,W), (H,W)]
    coords = torch.stack(coords[::-1], dim=0).float()  # (2, H, W)
    return coords[None].repeat(batch_size, 1, 1, 1)  # (B, 2, H, W)

def upsample_flow(flow, up_mask: Optional[torch.Tensor] = None, factor: int = 8):
    """Upsample flow by the input factor (default 8).

    If up_mask is None we just interpolate.
    If up_mask is specified, we upsample using a convex combination of its weights. See paper page 8 and appendix B.
    Note that in appendix B the picture assumes a downsample factor of 4 instead of 8.
    """
    batch_size, num_channels, h, w = flow.shape
    new_h, new_w = h * factor, w * factor

    if up_mask is None:
        return factor * F.interpolate(flow, size=(new_h, new_w), mode="bilinear", align_corners=True)

    up_mask = up_mask.view(batch_size, 1, 9, factor, factor, h, w)
    up_mask = torch.softmax(up_mask, dim=2)  # "convex" == weights sum to 1

    upsampled_flow = F.unfold(factor * flow, kernel_size=3, padding=1).view(batch_size, num_channels, 9, 1, 1, h, w)
    upsampled_flow = torch.sum(up_mask * upsampled_flow, dim=2)

    return upsampled_flow.permute(0, 1, 4, 2, 5, 3).reshape(batch_size, num_channels, new_h, new_w)

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

        if len(layers) != 5:
            raise ValueError(f"The expected number of layers is 5, instead got {len(layers)}")

        # See note in ResidualBlock for the reason behind bias=True
        self.convnormrelu = Conv2dNormActivation(
            in_chan, layers[0], norm_layer=norm_layer, kernel_size=7, stride=strides[0], bias=True
        )

        self.layer1 = self._make_2_blocks(block, layers[0], layers[1], norm_layer=norm_layer, first_stride=strides[1])
        self.layer2 = self._make_2_blocks(block, layers[1], layers[2], norm_layer=norm_layer, first_stride=strides[2])
        self.layer3 = self._make_2_blocks(block, layers[2], layers[3], norm_layer=norm_layer, first_stride=strides[3])

        self.conv = nn.Conv2d(layers[3], layers[4], kernel_size=1)

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv(x)

        return x


    
class MotionEncoder(nn.Module):
    """The motion encoder, part of the update block.

    Takes the current predicted flow and the correlation features as input and returns an encoded version of these.
    """

    def __init__(self, *, in_channels_corr:int, corr_layers=(96,), flow_layers=(64, 32), out_channels=80):
        super().__init__()

        if len(flow_layers) != 2:
            raise ValueError(f"The expected number of flow_layers is 2, instead got {len(flow_layers)}")
        if len(corr_layers) not in (1, 2):
            raise ValueError(f"The number of corr_layers should be 1 or 2, instead got {len(corr_layers)}")

        self.convcorr1 = Conv2dNormActivation(in_channels_corr+1, corr_layers[0], norm_layer=None, kernel_size=1)
        if len(corr_layers) == 2:
            self.convcorr2 = Conv2dNormActivation(corr_layers[0], corr_layers[1], norm_layer=None, kernel_size=3)
        else:
            self.convcorr2 = nn.Identity()

        self.convflow1 = Conv2dNormActivation(2, flow_layers[0], norm_layer=None, kernel_size=7)
        self.convflow2 = Conv2dNormActivation(flow_layers[0], flow_layers[1], norm_layer=None, kernel_size=3)

        # out_channels - 2 because we cat the flow (2 channels) at the end
        self.conv = Conv2dNormActivation(
            corr_layers[-1] + flow_layers[-1], out_channels - 2, norm_layer=None, kernel_size=3
        )

        self.out_channels = out_channels

    def forward(self, flow:torch.Tensor, corr_features:torch.Tensor, confidence_map:torch.Tensor):
        corr = self.convcorr1(torch.cat([corr_features, confidence_map], dim=1))
        corr = self.convcorr2(corr)

        flow_orig = flow
        flow = self.convflow1(flow)
        flow = self.convflow2(flow)

        corr_flow = torch.cat([corr, flow], dim=1)
        corr_flow = self.conv(corr_flow)
        return torch.cat([corr_flow, flow_orig], dim=1)


class ConvGRU(nn.Module):
    """Convolutional Gru unit."""

    def __init__(self, *, input_size, hidden_size, kernel_size, padding):
        super().__init__()
        self.convz = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class RecurrentBlock(nn.Module):
    """Recurrent block, part of the update block.

    Takes the current hidden state and the concatenation of (motion encoder output, context) as input.
    Returns an updated hidden state.
    """

    def __init__(self, *, input_size:Tuple[int,int], hidden_size:int, kernel_size=((1, 5), (5, 1)), padding=((0, 2), (2, 0))):
        super().__init__()

        if len(kernel_size) != len(padding):
            raise ValueError(
                f"kernel_size should have the same length as padding, instead got len(kernel_size) = {len(kernel_size)} and len(padding) = {len(padding)}"
            )
        if len(kernel_size) not in (1, 2):
            raise ValueError(f"kernel_size should either 1 or 2, instead got {len(kernel_size)}")

        self.convgru1 = ConvGRU(
            input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size[0], padding=padding[0]
        )
        if len(kernel_size) == 2:
            self.convgru2 = ConvGRU(
                input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size[1], padding=padding[1]
            )
        else:
            self.convgru2 = _pass_through_h

        self.hidden_size = hidden_size

    def forward(self, h, x):
        h = self.convgru1(h, x)
        h = self.convgru2(h, x)
        return h

class CorrBlock(nn.Module):
    """The correlation block.

    Creates a correlation pyramid with ``num_levels`` levels from the outputs of the feature encoder,
    and then indexes from this pyramid to create correlation features.
    The "indexing" of a given centroid pixel x' is done by concatenating its surrounding neighbors that
    are within a ``radius``, according to the infinity norm (see paper section 3.2).
    Note: typo in the paper, it should be infinity norm, not 1-norm.
    """

    def __init__(self, *, num_levels: int = 4, radius: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius

        self.corr_pyramid: List[torch.Tensor] = [torch.tensor(0)]  # useless, but torchscript is otherwise confused :')

        # The neighborhood of a centroid pixel x' is {x' + delta, ||delta||_inf <= radius}
        # so it's a square surrounding x', and its sides have a length of 2 * radius + 1
        # The paper claims that it's ||.||_1 instead of ||.||_inf but it's a typo:
        # https://github.com/princeton-vl/RAFT/issues/122
        self.out_channels = num_levels * (2 * radius + 1) ** 2

    def build_pyramid(self, fmap1:torch.Tensor, fmap2:torch.Tensor):
        """Build the correlation pyramid from two feature maps.

        The correlation volume is first computed as the dot product of each pair (pixel_in_fmap1, pixel_in_fmap2)
        The last 2 dimensions of the correlation volume are then pooled num_levels times at different resolutions
        to build the correlation pyramid.
        """

        if fmap1.shape != fmap2.shape:
            raise ValueError(
                f"Input feature maps should have the same shape, instead got {fmap1.shape} (fmap1.shape) != {fmap2.shape} (fmap2.shape)"
            )
        corr_volume = self._compute_corr_volume(fmap1, fmap2)

        batch_size, h, w, num_channels, _, _ = corr_volume.shape  # _, _ = h, w
        corr_volume = corr_volume.reshape(batch_size * h * w, num_channels, h, w)
        self.corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr_volume)

    def index_pyramid(self, centroids_coords:torch.Tensor):
        """Return correlation features by indexing from the pyramid."""
        neighborhood_side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
        di = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
        dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
        delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), dim=-1).to(centroids_coords.device)
        delta = delta.view(1, neighborhood_side_len, neighborhood_side_len, 2)

        batch_size, _, h, w = centroids_coords.shape  # _ = 2
        centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 2)

        indexed_pyramid = []
        for corr_volume in self.corr_pyramid:
            sampling_coords = centroids_coords + delta  # end shape is (batch_size * h * w, side_len, side_len, 2)
            indexed_corr_volume = grid_sample(corr_volume, sampling_coords, align_corners=True, mode="bilinear").view(
                batch_size, h, w, -1
            )
            indexed_pyramid.append(indexed_corr_volume)
            centroids_coords = centroids_coords / 2

        corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()

        expected_output_shape = (batch_size, self.out_channels, h, w)
        if corr_features.shape != expected_output_shape:
            raise ValueError(
                f"Output shape of index pyramid is incorrect. Should be {expected_output_shape}, got {corr_features.shape}"
            )

        return corr_features

    def _compute_corr_volume(self, fmap1:torch.Tensor, fmap2:torch.Tensor):
        batch_size, num_channels, h, w = fmap1.shape
        fmap1 = fmap1.view(batch_size, num_channels, h * w)
        fmap2 = fmap2.view(batch_size, num_channels, h * w)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch_size, h, w, 1, h, w)
        return corr / torch.sqrt(torch.tensor(num_channels))
    

def _pass_through_h(h, _):
    # Declared here for torchscript
    return h



class FlowHead(nn.Module):
    """Flow head, part of the update block.

    Takes the hidden state of the recurrent unit as input, and outputs the predicted "delta flow".
    """

    def __init__(self, *, in_channels:int, hidden_size:int, pooling_size:Tuple[int,int]):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(pooling_size)
        inplanes = hidden_size * pooling_size[0] * pooling_size[1]
        self.conv1 = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = MLPNet(head_dims=[inplanes, 256], sub_dims=[256, 3], activation_func=nn.LeakyReLU(0.1, inplace=True))
        self.pcd = None
        self.camera_info = None
        self.uv0 = None

    
    def initialize(self, pcd:torch.Tensor, camera_info:Optional[Dict]=None):
        self.pcd = pcd
        if camera_info is None:
            assert self.camera_info is not None, "camera_info must be given for the first time."
        else:
            self.camera_info = camera_info
            self.uv0 = project_pc2image(pcd, self.camera_info)  # (B, 2, N)

    def forward(self, hidden_state:torch.Tensor, Tcl:torch.Tensor):
        if self.pcd is None or self.uv0 is None:
            raise ValueError("must call initialize() before forward.")
        x = self.relu(self.conv1(hidden_state))
        x = torch.flatten(self.pooling(x), start_dim=1)  # (B, C, H, W) -> (B, C*H*W)
        x = self.mlp(x)  # predict delta_x
        pcd_tf = se3.transform(se3.exp(x) @ Tcl, self.pcd)
        uv = project_pc2image(pcd_tf, self.camera_info)  # (B, 2, N)
        sparse_flow2d = uv - self.uv0
        return x, sparse_flow2d, self.uv0


class UpdateBlock(nn.Module):
    """The update block which contains the motion encoder, the recurrent block, and the flow head.

    It must expose a ``hidden_state_size`` attribute which is the hidden state size of its recurrent block.
    """

    def __init__(self, *, motion_encoder:MotionEncoder, recurrent_block:RecurrentBlock, flow_head:FlowHead):
        super().__init__()
        self.motion_encoder = motion_encoder
        self.recurrent_block = recurrent_block
        self.flow_head = flow_head
        self.hidden_state_size = recurrent_block.hidden_size
        self.pcd = None
        self.camera_info = None


    def forward(self, hidden_state, context, corr_features, flow, confidence_map, Tcl) -> Tuple[torch.Tensor, torch.Tensor]:
        motion_features = self.motion_encoder(flow, corr_features, confidence_map)
        x = torch.cat([context, motion_features], dim=1)
        hidden_state = self.recurrent_block(hidden_state, x)
        x, sparse_flow, uv = self.flow_head(hidden_state, Tcl)
        return hidden_state, x, sparse_flow, uv
    

class LCCRAFT(nn.Module):
    def __init__(self, # Feature encoder
        feature_encoder_layers=(32, 32, 64, 96, 128),
        feature_encoder_block:Literal['bottleneck','residualblock']='bottleneck',
        feature_encoder_norm_layer:Literal['batchnorm','instancenorm']='batchnorm',
        # Context encoder
        context_encoder_layers=(32, 32, 64, 96, 160),
        context_encoder_block:Literal['bottleneck','residualblock']='bottleneck',
        context_encoder_norm_layer=None,
        # Correlation block
        corr_block_num_levels=4,
        corr_block_radius=3,
        # Motion encoder
        motion_encoder_corr_layers=(96,),
        motion_encoder_flow_layers=(64, 32),
        motion_encoder_out_channels=82,
        # Recurrent block
        recurrent_block_hidden_state_size=96,
        recurrent_block_kernel_size=(3,3),
        recurrent_block_padding=(1,1),
        # Flow head
        flow_head_hidden_size=128,
        depth_gen_pooling_size:int=1,
        depth_gen_max_depth:float=50.0,
        feat_poooling_size:Tuple[int,int]=[2,4],
        fps_num:int=512):
        """
        `LCCRAFT: LCCRAFT: LiDAR and Camera Calibration Using Recurrent All-Pairs Field Transforms Without Precise Initial Guess`_.

        args:
            img_feature_encoder (nn.Module): The image feature encoder. It must downsample the input by 8.
            depth_feature_encoder (nn.Module): The depth map feature encoder. It must downsample the input by 8.
            depth_context_encoder (nn.Module): The depth map feature encoder. It must downsample the input by 8.
                Its input is ``depth map``. Its output will be split into 2 parts:

                - one part will be used as the actual "context", passed to the recurrent unit of the ``update_block``
                - one part will be used to initialize the hidden state of the of the recurrent unit of
                  the ``update_block``

                These 2 parts are split according to the ``hidden_state_size`` of the ``update_block``, so the output
                of the ``context_encoder`` must be strictly greater than ``hidden_state_size``.

            corr_block (nn.Module): The correlation block, which creates a correlation pyramid from the output of the
                ``feature_encoder``, and then indexes from this pyramid to create correlation features. It must expose
                2 methods:

                - a ``build_pyramid`` method that takes ``feature_map_1`` and ``feature_map_2`` as input (these are the
                  output of the ``feature_encoder``).
                - a ``index_pyramid`` method that takes the coordinates of the centroid pixels as input, and returns
                  the correlation features. See paper section 3.2.

                It must expose an ``out_channels`` attribute.

            update_block (nn.Module): The update block, which contains the motion encoder, the recurrent unit, and the
                flow head. It takes as input the hidden state of its recurrent unit, the context, the correlation
                features, and the current predicted flow. It outputs an updated hidden state, and the ``delta_flow``
                prediction (see paper appendix A). It must expose a ``hidden_state_size`` attribute.
        """
        super().__init__()
        feature_encoder_block = BottleneckBlock if feature_encoder_block == 'bottleneck' else ResidualBlock
        feature_encoder_norm_layer = BatchNorm2d if feature_encoder_norm_layer == 'batchnorm' else InstanceNorm2d
        context_encoder_block = BottleneckBlock if context_encoder_block == 'bottleneck' else ResidualBlock
        self.img_feature_encoder = FeatureEncoder(in_chan=3, block=feature_encoder_block, layers=feature_encoder_layers, norm_layer=feature_encoder_norm_layer)
        self.depth_feature_encoder = FeatureEncoder(in_chan=1, block=feature_encoder_block, layers=feature_encoder_layers, norm_layer=feature_encoder_norm_layer)
        self.depth_context_encoder = FeatureEncoder(in_chan=1, block=context_encoder_block, layers=context_encoder_layers, norm_layer=context_encoder_norm_layer)
        self.depth_generator = DepthImgGenerator(depth_gen_pooling_size, depth_gen_max_depth)
        self.corr_block = CorrBlock(num_levels=corr_block_num_levels, radius=corr_block_radius)
        motion_encoder = MotionEncoder(
            in_channels_corr=self.corr_block.out_channels,
            corr_layers=motion_encoder_corr_layers,
            flow_layers=motion_encoder_flow_layers,
            out_channels=motion_encoder_out_channels,
        )

        # See comments in forward pass of RAFT class about why we split the output of the context encoder
        out_channels_context = context_encoder_layers[-1] - recurrent_block_hidden_state_size
        recurrent_block = RecurrentBlock(
            input_size=motion_encoder.out_channels + out_channels_context,
            hidden_size=recurrent_block_hidden_state_size,
            kernel_size=recurrent_block_kernel_size,
            padding=recurrent_block_padding,
        )

        flow_head = FlowHead(in_channels=recurrent_block_hidden_state_size, hidden_size=flow_head_hidden_size, pooling_size=feat_poooling_size)

        self.update_block = UpdateBlock(motion_encoder=motion_encoder, recurrent_block=recurrent_block, flow_head=flow_head)
        self.fps_num = fps_num
        self.buffer = dict()

    def restore_buffer(self, img:torch.Tensor):
        img_fmap = self.img_feature_encoder(img)
        self.buffer['img_fmap'] = img_fmap

    def clear_buffer(self):
        self.buffer.clear()

    def forward(self, img:torch.Tensor, pcd_tf:torch.Tensor, camera_info:Dict, num_flow_updates: int = 12):
        batch_size, _, h, w = img.shape
        if not (h % 8 == 0) and (w % 8 == 0):
            raise ValueError(f"input image H and W should be divisible by 8, insted got {h} (h) and {w} (w)")
        depth = self.depth_generator.project(pcd_tf, camera_info)  # (B, 1, h, w)
        if len(self.buffer.keys()) == 0:
            img_fmap = self.img_feature_encoder(img)
        else:
            img_fmap = self.buffer['img_fmap']
        depth_fmap = self.depth_feature_encoder(depth)
        confidence_map = self.depth_generator.binary_project(pcd_tf, camera_info)
        if img_fmap.shape[-2:] != (h // 8, w // 8):  # tuple equation
            raise ValueError("The image feature encoder should downsample H and W by 8")
        if depth_fmap.shape[-2:] != (h // 8, w // 8):
            raise ValueError("The image feature encoder should downsample H and W by 8")
        context_out = self.depth_context_encoder(depth)
        if context_out.shape[-2:] != (h // 8, w // 8):
            raise ValueError("The context encoder should downsample H and W by 8")
        self.corr_block.build_pyramid(depth_fmap, img_fmap)
        sampled_indices = furthest_point_sampling(pcd_tf.transpose(1,2), self.fps_num, cpp_impl=True) # (B, N)
        pcd_tf_downsampled = batch_indexing(pcd_tf, sampled_indices)
        feat_camera_info = camera_info.copy()
        feat_h, feat_w = img_fmap.shape[-2:]
        kx = feat_w / camera_info['sensor_w']
        ky = feat_h / camera_info['sensor_h']
        feat_camera_info.update({
            'sensor_w': feat_w,
            'sensor_h': feat_h,
            'fx': kx * camera_info['fx'],
            'fy': ky * camera_info['fy'],
            'cx': kx * camera_info['cx'],
            'cy': kx * camera_info['cy']
        })
        confidence_map = F.interpolate(confidence_map, size=(feat_h, feat_w), mode='bilinear')
        self.update_block.flow_head.initialize(pcd_tf_downsampled, feat_camera_info)
        # As in the original paper, the actual output of the context encoder is split in 2 parts:
        # - one part is used to initialize the hidden state of the recurent units of the update block
        # - the rest is the "actual" context.
        hidden_state_size = self.update_block.hidden_state_size
        out_channels_context = context_out.shape[1] - hidden_state_size
        if out_channels_context <= 0:
            raise ValueError(
                f"The context encoder outputs {context_out.shape[1]} channels, but it should have at strictly more than hidden_state={hidden_state_size} channels"
            )
        hidden_state, context = torch.split(context_out, [hidden_state_size, out_channels_context], dim=1)
        hidden_state = torch.tanh(hidden_state)
        context = F.relu(context)
        coords0 = make_coords_grid(batch_size, h // 8, w // 8).to(img_fmap.device)  # (B, 2, H, W)
        coords1 = torch.clone(coords0)  # (B, 2, H, W)
        Tcl = torch.eye(4)[None].repeat(batch_size, 1, 1).to(pcd_tf)
        Tcl_predictions = [Tcl]
        dense_flow = torch.zeros_like(coords0)  # (B, 2, H, W)
        for _ in range(num_flow_updates):
            corr_features = self.corr_block.index_pyramid(centroids_coords=(dense_flow + coords0).detach())
            hidden_state, x, sprase_flow, uv = self.update_block(hidden_state, context, corr_features, dense_flow, confidence_map, Tcl_predictions[-1])
            dense_flow = knn_interpolation(uv, sprase_flow, torch.flatten(coords1, start_dim=-2), k=3).reshape(batch_size, 2, feat_h, feat_w)
            Tcl_predictions.append(se3.exp(x) @ Tcl_predictions[-1])
        return Tcl_predictions[1:]
    
    @staticmethod
    def sequence_loss(x_preds:List[torch.Tensor], x_gt:torch.Tensor, loss_fn:Callable, gamma:float):
        N = len(x_preds)
        loss = 0
        i_weight = 1
        for i in reversed(range(N)):
            loss += loss_fn(x_preds[i], x_gt) * i_weight
            i_weight *= gamma
        return loss