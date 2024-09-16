"""
Modified from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List, Tuple, Literal, Dict
from ..tools.core import get_activation_func

class VanillaVAE(nn.Module):
    def __init__(self,
                 img_size:Tuple[int,int],
                 in_chan:int=4,
                 latent_dim:int=128,
                 hidden_dims:List[int]=[32, 64, 128, 256, 512],
                 depthgen_argv:Dict={"pooling_size":1, "max_depth":50.0},
                 activation:Literal['leakyrelu','relu','elu','gelu']='leakyrelu',
                 inplace:bool=True) -> None:
        """VanillaVAE

        Args:
            img_size (Tuple[int, int]): size of the image
            in_chan (int, optional): input channel,  Defaults to 4
            latent_dim (int, optional): latent of mu and var,  Defaults to 128
            hidden_dims (List[int], optional): Channels of conv and convTranspose . Defaults to [32, 64, 128, 256, 512].
            pooling_size (Tuple[int, int], optional): pooling size of the last dimension. Defaults to [2, 4].
            activation (Literal[&#39;leakyrelu&#39;,&#39;relu&#39;,&#39;elu&#39;,&#39;gelu&#39;], optional): activation type. Defaults to 'leakyrelu'.
            inplace (bool, optional): activation inplace. Defaults to True.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.depthgen_argv = depthgen_argv
        activation_fn = get_activation_func(activation, inplace)
        modules = []
        self.downsampled_ratio = 2 ** len(hidden_dims)
        hidden_dims = [in_chan] + hidden_dims
        # Build Encoder
        for in_dim, out_dim in zip(hidden_dims[:-2], hidden_dims[1:-1]):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_dim),
                    activation_fn)
            )
        modules.append(nn.Conv2d(hidden_dims[-2], hidden_dims[-1], kernel_size=3, stride=2, padding=1))
        self.encoder = nn.Sequential(*modules)
        self.final_size = (img_size[0] // self.downsampled_ratio, img_size[1] // self.downsampled_ratio)
        self.final_chan = hidden_dims[-1]
        fc_inplanes = hidden_dims[-1] * self.final_size[0] * self.final_size[1]
        self.fc_mu = nn.Linear(fc_inplanes, latent_dim)
        self.fc_var = nn.Linear(fc_inplanes, latent_dim)
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, fc_inplanes)
        hidden_dims.reverse()
        hidden_dims[-1] = 1  # depth
        for in_dim, out_dim in zip(hidden_dims[:-2], hidden_dims[1:-1]):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_dim,
                                       out_dim,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(out_dim),
                    activation_fn)
            )
        modules.append(nn.ConvTranspose2d(hidden_dims[-2], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoder = nn.Sequential(*modules)
        

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        x = self.encoder(x)  # (B, C, H, W)
        x = torch.flatten(x, start_dim=1)  # (B, C*H*W)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        x = self.decoder_input(z)
        x = x.view(-1, self.final_chan, *self.final_size)
        x = self.decoder(x)
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  self.decode(z), mu, log_var

    @staticmethod
    def loss_function(gt_output: torch.Tensor,
                      output: torch.Tensor,
                      mu: torch.Tensor,
                      log_var: torch.Tensor,
                      kld_weight:float=1.0E-8) -> torch.Tensor:
        """VAE Loss function

        Args:
            kld_weight (float, optional): _description_. Defaults to 1.0E-8.

        Returns:
            dict: 'loss', 'Reconstruction_Loss', 'KLD'
        """

        recons_loss = VanillaVAE.reconstruction_loss(output, gt_output)
        kld_loss = VanillaVAE.kld_loss(mu, log_var)
        loss = recons_loss + kld_weight * kld_loss
        return loss
    
    @staticmethod
    def reconstruction_loss(pred:torch.Tensor, gt:torch.Tensor):
        recon_loss = F.l1_loss(pred, gt, reduction='mean')
        return recon_loss

    @staticmethod
    def kld_loss(mu:torch.Tensor, log_var:torch.Tensor):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return kld_loss

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]