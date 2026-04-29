# UNet noise predictor for the DDPM pixel-space model.
# Used as the network argument of DDPMModel.
#
# Author: Grégory Sainton
# Lab: Observatoire de Paris - PSL University


import torch
import torch.nn as nn


def sinusoidal_embedding(n_steps, emb_dim):
    """
    Sinusoidal positional embeddings for diffusion timesteps.

    Args:
        n_steps (int): Total number of diffusion timesteps.
        emb_dim (int): Embedding dimension (must be even).

    Returns:
        torch.Tensor: Embeddings of shape (n_steps, emb_dim).
    """
    assert emb_dim % 2 == 0, "emb_dim must be even."
    embedding = torch.zeros(n_steps, emb_dim)
    for i in range(emb_dim):
        if i % 2 == 0:
            embedding[:, i] = torch.sin(torch.arange(n_steps) / 10000 ** (i / emb_dim))
        else:
            embedding[:, i] = torch.cos(torch.arange(n_steps) / 10000 ** (i / emb_dim))
    return embedding


class ResBlock(nn.Module):
    """
    Convolutional residual block with optional LayerNorm.

    Args:
        shape (tuple): Spatial shape (C, H, W) for LayerNorm.
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        stride (int): Convolution stride.
        padding (int): Convolution padding.
        activation (nn.Module, optional): Activation function. Default: SiLU.
        normalize (bool): Whether to apply LayerNorm. Default: False.
        num_filters (int, optional): Intermediate channel count. Default: out_c.
    """

    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1,
                 activation=None, normalize=False, num_filters=None):
        super(ResBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        filters = num_filters if num_filters is not None else out_c
        self.conv1 = nn.Conv2d(in_c, filters, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(filters, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize



class UNet(nn.Module):
    """
    UNet noise predictor for 64x64 RGB images.

    Time conditioning is injected at each resolution level via additive
    sinusoidal embeddings projected to the channel dimension.

    Args:
        n_steps (int): Total number of diffusion timesteps.
        time_emb_dim (int): Dimension of the sinusoidal time embedding.
    """

    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()

        # Frozen sinusoidal time embedding table
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # Encoder
        self.te1 = self._make_te(time_emb_dim, 3)
        self.b1 = nn.Sequential(
            ResBlock((3, 64, 64), 3, 16, num_filters=32),
            ResBlock((16, 64, 64), 16, 16),
            ResBlock((16, 64, 64), 16, 16),
        )
        self.down1 = nn.Conv2d(16, 16, 4, 2, 1)   # -> (B, 16, 32, 32)

        self.te2 = self._make_te(time_emb_dim, 16)
        self.b2 = nn.Sequential(
            ResBlock((16, 32, 32), 16, 32, num_filters=64),
            ResBlock((32, 32, 32), 32, 32),
            ResBlock((32, 32, 32), 32, 32),
        )
        self.down2 = nn.Conv2d(32, 32, 4, 2, 1)   # -> (B, 32, 16, 16)

        self.te3 = self._make_te(time_emb_dim, 32)
        self.b3 = nn.Sequential(
            ResBlock((32, 16, 16), 32, 64, num_filters=128),
            ResBlock((64, 16, 16), 64, 64),
            ResBlock((64, 16, 16), 64, 64),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 64, 2, 1, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1),            # -> (B, 64, 8, 8)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 64)
        self.b_mid = nn.Sequential(
            ResBlock((64, 8, 8), 64, 128, num_filters=256),
            ResBlock((128, 8, 8), 128, 128),
            ResBlock((128, 8, 8), 128, 64),
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 64, 2, 1, padding=1),  # -> (B, 64, 16, 16)
        )
        self.te4 = self._make_te(time_emb_dim, 128)
        self.b4 = nn.Sequential(
            ResBlock((128, 16, 16), 128, 64),
            ResBlock((64, 16, 16), 64, 32),
            ResBlock((32, 16, 16), 32, 32),
        )

        self.up2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)    # -> (B, 32, 32, 32)
        self.te5 = self._make_te(time_emb_dim, 64)
        self.b5 = nn.Sequential(
            ResBlock((64, 32, 32), 64, 32),
            ResBlock((32, 32, 32), 32, 16),
            ResBlock((16, 32, 32), 16, 16),
        )

        self.up3 = nn.ConvTranspose2d(16, 16, 4, 2, 1)    # -> (B, 16, 64, 64)
        self.te_out = self._make_te(time_emb_dim, 32)
        self.b_out = nn.Sequential(
            ResBlock((32, 64, 64), 32, 16),
            ResBlock((16, 64, 64), 16, 16),
            ResBlock((16, 64, 64), 16, 16, normalize=False),
        )

        self.conv_out = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, x, t):
        """
        Args:
            x (torch.Tensor): Noisy images (B, 3, 64, 64).
            t (torch.Tensor): Timestep indices (B,).

        Returns:
            torch.Tensor: Predicted noise (B, 3, 64, 64).
        """
        t_emb = self.time_embed(t)
        B = len(x)

        # Encoder
        out1 = self.b1(x + self.te1(t_emb).reshape(B, -1, 1, 1))       # (B, 16, 64, 64)
        out2 = self.b2(self.down1(out1) + self.te2(t_emb).reshape(B, -1, 1, 1))  # (B, 32, 32, 32)
        out3 = self.b3(self.down2(out2) + self.te3(t_emb).reshape(B, -1, 1, 1))  # (B, 64, 16, 16)

        # Bottleneck
        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t_emb).reshape(B, -1, 1, 1))  # (B, 64, 8, 8)

        # Decoder with skip connections
        out4 = self.b4(
            torch.cat([out3, self.up1(out_mid)], dim=1)
            + self.te4(t_emb).reshape(B, -1, 1, 1)
        )  # (B, 32, 16, 16)

        out5 = self.b5(
            torch.cat([out2, self.up2(out4)], dim=1)
            + self.te5(t_emb).reshape(B, -1, 1, 1)
        )  # (B, 16, 32, 32)

        out = self.b_out(
            torch.cat([out1, self.up3(out5)], dim=1)
            + self.te_out(t_emb).reshape(B, -1, 1, 1)
        )  # (B, 16, 64, 64)

        return self.conv_out(out)  # (B, 3, 64, 64)

    def _make_te(self, dim_in, dim_out):
        """Time embedding projection: R^dim_in -> R^dim_out."""
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )
