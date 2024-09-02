import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Union, List, Optional, Tuple, Callable
from timm.layers import trunc_normal_
from collections import OrderedDict
from functools import partial

# def scn_input_wrapper(coords: Iterable[torch.Tensor], features: Iterable[torch.Tensor], scale:Union[float, Iterable[float]], bias:float, device=None) -> Tuple[torch.Tensor, torch.Tensor, int]:
#     assert len(coords) == len(features)
#     D = coords[0].shape[-1]  # (B, N, D)
#     if not isinstance(scale, Iterable):
#         scale = [scale] * D
#     device = device if device is not None else coords[0].device
#     coord_params = dict(dtype=torch.int32, device=device)
#     feature_params = dict(dtype=torch.float32, device=device)
#     C = torch.empty([0,D+1], **coord_params)  # (N, D+1) last dim for batch_idx
#     for b, coord in enumerate(coords):
#         batchid_coord = torch.hstack([b*torch.ones([coord.shape[0],1],**coord_params),
#                                       ((coord+bias)*torch.tensor(scale)[None,:].to(device)).to(**coord_params)])
#         C = torch.vstack([C,batchid_coord])
#     F = torch.cat(features, dim=0).to(**feature_params)  # (N, FD)
#     return C.contiguous(), F, len(coords)

# def scn_output_wrapper(tensor:torch.Tensor, batch_idx:torch.Tensor, batch_size:int) -> torch.Tensor:
#     batch = torch.zeros(batch_size, tensor.shape[0] // batch_size, tensor.shape[1], dtype=tensor.dtype, device=tensor.device)
#     for bi in range(batch_size):
#         bi_indices = batch_idx==bi
#         batch[bi, ...] = tensor[bi_indices,:]
#     return batch  # (B, N, FD)

def sparse_tensor_collate(coords:Iterable[torch.Tensor], scale=1.0, bias=0.0):
    D = coords[0].shape[-1]  # (B, N, D)
    coord_params = dict(dtype=torch.int32, device=coords[0].device)
    C = torch.empty([0,D+1], **coord_params)  # (N, D+1) first dim for batch_idx
    for b, coord in enumerate(coords):
        batchid_coord = torch.hstack([b*torch.ones([coord.shape[0],1],**coord_params), ((coord+bias)*scale).to(**coord_params)])
        C = torch.vstack([C,batchid_coord])
    return C.contiguous()

# copied from pytorch (but remove the condition h > 1)
def grid_sample(img: torch.Tensor, absolute_grid: torch.Tensor, mode: str = "bilinear", align_corners: Optional[bool] = None):
    """Same as torch's grid_sample, with absolute pixel coordinates instead of normalized coordinates."""
    h, w = img.shape[-2:]   # (B,H,W,C)

    xgrid, ygrid = absolute_grid.split([1, 1], dim=-1)  # (B,H,W,1), (B,H,W,1)
    xgrid = 2 * xgrid / (w - 1) - 1
    ygrid = 2 * ygrid / (h - 1) - 1
    normalized_grid = torch.cat([xgrid, ygrid], dim=-1)

    return F.grid_sample(img, normalized_grid, mode=mode, align_corners=align_corners)

def conv_norm_relu(in_chan:int, out_chan:int, kszie:int, norm_fn:Union[nn.Module, None], activation:Union[nn.Module, None], **conv_args):
    module_list = [spconv.SubMConv3d(in_chan, out_chan, kszie, **conv_args)]
    if norm_fn is not None:
        module_list.append(norm_fn(out_chan))
    if activation is not None:
        module_list.append(activation())
    return module_list


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional):
            Norm layer that will be stacked on top of the convolution layer.
            If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional):
            Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the
            conv layer. If ``None`` this layer wont be used.
            Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can
            optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = True,
        bias: bool = True,
        norm_first: bool = False,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from
        # the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels

        for hidden_dim in hidden_channels[:-1]:
            if norm_first and norm_layer is not None:
                layers.append(norm_layer(in_dim))

            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))

            if not norm_first and norm_layer is not None:
                layers.append(norm_layer(hidden_dim))

            layers.append(activation_layer(**params))

            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout, **params))

            in_dim = hidden_dim

        if norm_first and norm_layer is not None:
            layers.append(norm_layer(in_dim))

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)

class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, embed_channels, kernel_size=1, bias=False
                ),
                norm_fn(embed_channels),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out

class SpUNetNoSkipBase(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=32,
        channels=(32, 64, 128, 128, 96, 96),
        layers=(2, 3, 4, 4, 2, 2),
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[s],
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(channels[s]),
                    nn.ReLU(),
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                            # if i == 0 else
                            (
                                f"block{i}",
                                block(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )

            # decode num_stages
            self.up.append(
                spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        channels[len(channels) - s - 2],
                        dec_channels,
                        kernel_size=3,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(dec_channels),
                    nn.ReLU(),
                )
            )
            self.dec.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                (
                                    f"block{i}",
                                    block(
                                        dec_channels,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}",
                                    ),
                                )
                                if i == 0
                                else (
                                    f"block{i}",
                                    block(
                                        dec_channels,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}",
                                    ),
                                )
                            )
                            for i in range(layers[len(channels) - s - 1])
                        ]
                    )
                )
            )
            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        self.final = (
            spconv.SubMConv3d(
                channels[-1], out_channels, kernel_size=1, padding=1, bias=True
            )
            if out_channels > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)  # apply initiation through all children

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data):
        grid_coord, feat, batch_size = data
        amax = torch.amax(grid_coord[:,1:], dim=0)
        amin = torch.amin(grid_coord[:,1:],dim=0)
        drange = amax - amin
        grid_coord[:, 1:] -= amin[None, :]
        drange_bias = torch.zeros_like(drange)
        drange_bias[drange % 2 ==0] = 1
        drange += drange_bias
        x = spconv.SparseConvTensor(
            features=feat,
            indices=grid_coord.contiguous(),
            spatial_shape=drange,
            batch_size=batch_size,
        )
        x = self.conv_input(x)
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
            
        x = skips.pop(-1)
        # dec forward
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            # skip = skips.pop(-1)
            # x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[s](x)

        x = self.final(x)
        return x.features

class SInputHead(nn.Module):
    def __init__(self, in_chan:int, out_chan:int, norm_fn:Union[nn.Module, None], kernel_size:int, **conv_args) -> None:
        super().__init__()
        module_list = [spconv.SubMConv3d(in_chan, out_chan, kernel_size=kernel_size, **conv_args)]
        if norm_fn is not None:
            module_list.append(norm_fn(out_chan))
        module_list.append(nn.ReLU(inplace=False))
        self.conv_input = spconv.SparseSequential(
            *module_list
        )

    def forward(self, coord:torch.Tensor, feat:torch.Tensor, batch_size:int):
        amax = torch.amax(coord[:,1:], dim=0)
        amin = torch.amin(coord[:,1:], dim=0)
        drange = amax - amin
        coord[:, 1:] -= amin[None, :]
        drange_bias = torch.zeros_like(drange)
        drange_bias[drange % 2 ==0] = 1 # spatial shape is odd
        drange += drange_bias
        # this spatial shape suppress errors in downsampling with SpConv(k=3, stride=2, padding=1)
        x = spconv.SparseConvTensor(
            features=feat,
            indices=coord.contiguous(),
            spatial_shape=drange,
            batch_size=batch_size,
        )
        x = self.conv_input(x)
        return x

class SCBottleneckBlock(nn.Module):
    def __init__(self, in_chan:int, out_chan:int, norm_fn:Union[nn.Module,None], stride=1) -> None:
        super().__init__()
        self.layer1 = spconv.SparseSequential(
            *conv_norm_relu(in_chan, out_chan // 4, 1, norm_fn, nn.ReLU, stride=1)
        )
        self.layer2 = spconv.SparseSequential(
            *conv_norm_relu(out_chan // 4,  out_chan // 4, 3, norm_fn, nn.ReLU, stride=stride, padding=1)
        )
        self.layer3 = spconv.SparseSequential(
            *conv_norm_relu(out_chan // 4,  out_chan, 1, norm_fn, nn.ReLU, stride=1)
        )
        self.relu = nn.ReLU()
        if stride == 1:
            self.downsample = spconv.Identity()
        else:
            self.downsample = spconv.SparseSequential(
                *conv_norm_relu(in_chan,  out_chan, 1, norm_fn, None, stride=stride)
            )

    def forward(self, x):
        y = x
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        x = self.downsample(x)
        y = y.replace_feature(self.relu(y.features + x.features))  # residual adding
        return y


class SparseFeatureEncoder(nn.Module):
    """The feature encoder, used both as the actual feature encoder, and as the context encoder.

    It must downsample its input by 8.
    """

    def __init__(
        self, *, block=SCBottleneckBlock, in_chan=2, layers=(64, 64, 96, 128, 256), strides=(2, 1, 2, 2), norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
    ):
        super().__init__()
        if len(layers) != 5:
            raise ValueError(f"The expected number of layers is 5, instead got {len(layers)}")

        # See note in ResidualBlock for the reason behind bias=True
        self.convnormrelu = SInputHead(
            in_chan, layers[0], norm_fn=norm_fn, kernel_size=7, stride=strides[0], padding=3
        )

        self.layer1 = self._make_2_blocks(block, layers[0], layers[1], norm_fn=norm_fn, first_stride=strides[1])
        self.layer2 = self._make_2_blocks(block, layers[1], layers[2], norm_fn=norm_fn, first_stride=strides[2])
        self.layer3 = self._make_2_blocks(block, layers[2], layers[3], norm_fn=norm_fn, first_stride=strides[3])

        self.conv = spconv.SparseConv3d(layers[3], layers[4], kernel_size=1, stride=1)

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

    def _make_2_blocks(self, block, in_channels, out_channels, norm_fn, first_stride):
        block1 = block(in_channels, out_channels, norm_fn=norm_fn, stride=first_stride)
        block2 = block(out_channels, out_channels, norm_fn=norm_fn, stride=1)
        return nn.Sequential(block1, block2)

    def forward(self, x):
        x1 = self.convnormrelu(*x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)

        x5 = self.conv(x4)

        return [x1, x2, x3, x4, x5]


class SparseMotionEncoder(nn.Module):
    """The motion encoder, part of the update block.

    Takes the current predicted flow and the correlation features as input and returns an encoded version of these.
    """

    def __init__(self, *, in_channels_corr, corr_layers=(256, 192), flow_layers=(128, 64), out_channels=128, concat_dims=8):
        super().__init__()

        if len(flow_layers) != 2:
            raise ValueError(f"The expected number of flow_layers is 2, instead got {len(flow_layers)}")
        if len(corr_layers) not in (1, 2):
            raise ValueError(f"The number of corr_layers should be 1 or 2, instead got {len(corr_layers)}")

        self.convcorr1 = conv_norm_relu(in_channels_corr, corr_layers[0], norm_fn=None, kernel_size=1)
        if len(corr_layers) == 2:
            self.convcorr2 = conv_norm_relu(corr_layers[0], corr_layers[1], norm_fn=None, kernel_size=3)
        else:
            self.convcorr2 = nn.Identity()

        self.convflow1 = conv_norm_relu(2, flow_layers[0], norm_fn=None, kernel_size=7)
        self.convflow2 = conv_norm_relu(flow_layers[0], flow_layers[1], norm_fn=None, kernel_size=3)

        # out_channels - 2 because we cat the flow (2 channels) at the end
        self.conv = conv_norm_relu(
            corr_layers[-1] + flow_layers[-1], out_channels - concat_dims, norm_fn=None, kernel_size=3
        )

        self.out_channels = out_channels

    def forward(self, extrinsics:torch.Tensor, flow:torch.Tensor, corr_features:torch.Tensor):
        corr = self.convcorr1(corr_features)
        corr = self.convcorr2(corr)

        flow_orig = flow
        flow = self.convflow1(flow)
        flow = self.convflow2(flow)

        corr_flow = torch.cat([corr, flow], dim=1)
        corr_flow = self.conv(corr_flow)
        return torch.cat([corr_flow, flow_orig, extrinsics], dim=1)

class SparseCorrBlock(nn.Module):  # can use extrinsic to constrain flows (extrinsic -> project -> possible flows)
    """The correlation block (Sparse Version).

    Creates a correlation pyramid with ``num_levels`` levels from the outputs of the feature encoder,
    and then indexes from this pyramid to create correlation features.
    The "indexing" of a given centroid pixel x' is done by concatenating its surrounding neighbors that
    are within a ``radius``, according to the infinity norm (see paper section 3.2).
    Note: typo in the paper, it should be infinity norm, not 1-norm.
    """

    def __init__(self, src_feat_dim:int, tgt_feat_dim:int, hidden_dims:Iterable[int], num_levels:int = 4, radius:int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.input_dim = src_feat_dim + tgt_feat_dim
        self.hidden_dims = hidden_dims
        self.out_dim = 1 # has to be 1 currently
        self.mlp = MLP(src_feat_dim+tgt_feat_dim, hidden_dims + [self.out_dim], nn.BatchNorm1d)  # out_dim = 1
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
        fmap1: feat of source SparseTensor (N, D)
        fmap2: target Dense Tensor (B, C, H, W)
        """
        corr_volume = self._compute_corr_volume(fmap1, fmap2)  # (B, M, 1, H, W)
        B, M, _, H, W = corr_volume.shape
        corr_volume = corr_volume.reshape(B*M, 1, H, W)  # for averge pooling, the corr_volumn should be a 4-dimension tensor
        self.corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=2, stride=2)  # (B*M, C, h, w), h, w = (H, W) // 2^k
            self.corr_pyramid.append(corr_volume)

    def index_pyramid(self, centroids_coords:torch.Tensor):
        """Return correlation features by indexing from the pyramid.

        Args:
            centroids_coords (torch.Tensor): (B, N, 2)

        Raises:
            ValueError: Unexpected Shape

        Returns:
            corr_feat: B, C, N
        """
        side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
        di = torch.linspace(-self.radius, self.radius, side_len)
        dj = torch.linspace(-self.radius, self.radius, side_len)
        delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), dim=-1).to(centroids_coords.device)
        delta = delta.view(1, side_len, side_len, 2)

        B, M, _ = centroids_coords.shape  # _ = 2 (B,2,H,W)
        centroids_coords = centroids_coords.reshape(B*M, 2)[:,None,None,:]  # (B*N, 1, 1, 2), unsqueezed dimension for side_len product

        indexed_pyramid = []
        for corr_volume in self.corr_pyramid:  #  corr_volume (B*M, 1, h, w)
            sampling_coords = centroids_coords + delta  # end shape is (batch_size * h * w, side_len, side_len, 2)
            indexed_corr_volume = grid_sample(corr_volume, sampling_coords, align_corners=True, mode="bilinear").view(
                B, M, -1
            )  # 2D sampling, res (B, M, side_len*side_len)
            indexed_pyramid.append(indexed_corr_volume)  # num_levels of (B, M, side_len**2)
            centroids_coords = centroids_coords / 2  # k=2, stride=2

        corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 2, 1).contiguous()  # (B, M, out_chan) -> (B, out_chan, M)  out_chan = num_levels * side_len**2

        expected_output_shape = (B, self.out_channels, M)
        if corr_features.shape != expected_output_shape:
            raise ValueError(
                f"Output shape of index pyramid is incorrect. Should be {expected_output_shape}, got {corr_features.shape}"
            )

        return corr_features

    def _compute_corr_volume(self, fmap1:torch.Tensor, fmap2:torch.Tensor):
        """compute correluation volumn between a SparseTensor (feat) and a DenseTensor

        Args:
            feat1 (torch.Tensor): (B, M, C), features of the source image (sparse)
            fmap2 (torch.Tensor): (B, C, H, W)

        Returns:
            _type_: (M, N, C), N=B*H*W
        """
        assert fmap1.ndim == 3 and fmap2.ndim == 4
        M = fmap1.shape[1]
        B, C, H, W = fmap2.shape
        corr = torch.bmm(fmap1, fmap2.reshape(B, C, H*W))  # B, M, H*W
        return corr.unsqueeze(2).view(B, M, 1, H, W)  # (B, M, H*W) -> (B, M, 1, H*W) -> (B, M, 1, H, W)