# UNet noise predictor for the DDPM pixel-space model.
# Refactored following Ho et al. (2020) strictly:
#   - ResBlock with residual connection and 1x1 projection if in_c != out_c
#   - GroupNorm(32) instead of LayerNorm
#   - Time embedding injected inside each ResBlock (between the two convolutions)
#   - Self-attention at the bottleneck (8x8)
#
# Reference: Ho et al., "Denoising Diffusion Probabilistic Models", 2020.
# https://arxiv.org/abs/2006.11239
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
            embedding[:, i] = torch.sin(
                torch.arange(n_steps) / 10000 ** (i / emb_dim)
            )
        else:
            embedding[:, i] = torch.cos(
                torch.arange(n_steps) / 10000 ** (i / emb_dim)
            )
    return embedding


class ResBlock(nn.Module):
    """
    Residual block with GroupNorm and time embedding injection (Ho et al., 2020).

    Time embedding is injected between the two convolutions via an additive
    linear projection, allowing each block to modulate its intermediate
    representation as a function of the diffusion timestep.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        time_emb_dim (int): Dimension of the projected time embedding.
        kernel_size (int): Convolution kernel size. Default: 3.
        stride (int): Convolution stride. Default: 1.
        padding (int): Convolution padding. Default: 1.
        activation (nn.Module, optional): Activation function. Default: SiLU.
    """

    def __init__(self, in_c, out_c, time_emb_dim, kernel_size=3, stride=1,
                 padding=1, activation=None):
        super(ResBlock, self).__init__()
        self.norm1 = nn.GroupNorm(min(32, in_c), in_c)
        self.norm2 = nn.GroupNorm(min(32, out_c), out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.time_proj = nn.Linear(time_emb_dim, out_c)
        self.shortcut = (
            nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        )

    def forward(self, x, t_emb):
        """
        Args:
            x (torch.Tensor): Input feature map (B, in_c, H, W).
            t_emb (torch.Tensor): Projected time embedding (B, time_emb_dim).

        Returns:
            torch.Tensor: Output feature map (B, out_c, H, W).
        """
        out = self.activation(self.conv1(self.norm1(x)))
        out = out + self.time_proj(t_emb).reshape(x.shape[0], -1, 1, 1)
        out = self.activation(self.conv2(self.norm2(out)))
        return out + self.shortcut(x)


class SelfAttention(nn.Module):
    """
    Self-attention block for spatial feature maps (Ho et al., 2020).

    Reshapes (B, C, H, W) to a sequence (H*W, B, C), applies multi-head
    attention, then reshapes back. A residual connection is added.

    Args:
        n_channels (int): Number of input/output channels.
        n_heads (int): Number of attention heads. Default: 8.
    """

    def __init__(self, n_channels, n_heads=8):
        super(SelfAttention, self).__init__()
        self.norm = nn.GroupNorm(32, n_channels)
        self.attn = nn.MultiheadAttention(n_channels, n_heads)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).

        Returns:
            torch.Tensor: Output feature map (B, C, H, W).
        """
        B, C, H, W = x.shape
        out = self.norm(x)
        out = out.flatten(2).permute(2, 0, 1)       # (H*W, B, C)
        out, _ = self.attn(out, out, out)
        out = out.permute(1, 2, 0).reshape(B, C, H, W)
        return x + out


class UNetV2(nn.Module):
    """
    UNet noise predictor for 64x64 RGB images (Ho et al., 2020).

    Improvements over UNetV1:
    - ResBlocks with residual connections and 1x1 projection shortcuts
    - GroupNorm(32) instead of LayerNorm
    - Time embedding injected inside each ResBlock
    - Self-attention at configurable resolutions

    The encoder downsamples 64x64 -> 32x32 -> 16x16 -> 8x8.
    The decoder upsamples 8x8 -> 16x16 -> 32x32 -> 64x64.
    Skip connections concatenate encoder and decoder feature maps at each level.

    Args:
        n_steps (int): Total number of diffusion timesteps. Default: 1000.
        time_emb_dim (int): Sinusoidal embedding dimension. Default: 256.
        attention_resolutions (list): Spatial resolutions at which self-attention
            is applied. Supported values: 8 (bottleneck), 16 (enc3/dec3).
            []      -> no attention (ablation Run 4a)
            [8]     -> bottleneck only (Ho et al. 2020, Run 4b)
            [8, 16] -> bottleneck + 16x16 (Nichol & Dhariwal 2021, Run 3)
            Default: [8, 16].
    """

    def __init__(self, n_steps=1000, time_emb_dim=256, attention_resolutions=None):
        super(UNetV2, self).__init__()

        if attention_resolutions is None:
            attention_resolutions = [8, 16]
        self.attention_resolutions = attention_resolutions

        # Frozen sinusoidal time embedding table
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # Time embedding MLP: projects raw sinusoidal embedding to a richer
        # representation before injection in each ResBlock.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        te = time_emb_dim

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.enc1_a = ResBlock(3, 64, te)
        self.enc1_b = ResBlock(64, 64, te)
        self.down1 = nn.Conv2d(64, 64, 4, 2, 1)     # -> (B, 64, 32, 32)

        self.enc2_a = ResBlock(64, 128, te)
        self.enc2_b = ResBlock(128, 128, te)
        self.down2 = nn.Conv2d(128, 128, 4, 2, 1)   # -> (B, 128, 16, 16)

        self.enc3_a = ResBlock(128, 256, te)
        self.enc3_b = ResBlock(256, 256, te)
        self.attn_enc3 = SelfAttention(256, n_heads=8) if 16 in attention_resolutions else nn.Identity()
        self.down3 = nn.Conv2d(256, 256, 4, 2, 1)   # -> (B, 256, 8, 8)

        # ------------------------------------------------------------------
        # Bottleneck: 8x8 with optional self-attention
        # ------------------------------------------------------------------
        self.mid_a = ResBlock(256, 256, te)
        self.mid_attn = SelfAttention(256, n_heads=8) if 8 in attention_resolutions else nn.Identity()
        self.mid_b = ResBlock(256, 256, te)

        # ------------------------------------------------------------------
        # Decoder (skip connections via concatenation)
        # ------------------------------------------------------------------
        self.up3 = nn.ConvTranspose2d(256, 256, 4, 2, 1)    # -> (B, 256, 16, 16)
        self.dec3_a = ResBlock(512, 256, te)                 # 512 = 256 skip + 256 up
        self.dec3_b = ResBlock(256, 128, te)
        self.attn_dec3 = SelfAttention(128, n_heads=8) if 16 in attention_resolutions else nn.Identity()

        self.up2 = nn.ConvTranspose2d(128, 128, 4, 2, 1)    # -> (B, 128, 32, 32)
        self.dec2_a = ResBlock(256, 128, te)                 # 256 = 128 skip + 128 up
        self.dec2_b = ResBlock(128, 64, te)

        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)      # -> (B, 64, 64, 64)
        self.dec1_a = ResBlock(128, 64, te)                  # 128 = 64 skip + 64 up
        self.dec1_b = ResBlock(64, 64, te)

        # ------------------------------------------------------------------
        # Output projection
        # ------------------------------------------------------------------
        self.out_norm = nn.GroupNorm(32, 64)
        self.out_act = nn.SiLU()
        self.conv_out = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x, t):
        """
        Args:
            x (torch.Tensor): Noisy images (B, 3, 64, 64).
            t (torch.Tensor): Timestep indices (B,).

        Returns:
            torch.Tensor: Predicted noise (B, 3, 64, 64).
        """
        # Time embedding: (B,) -> (B, time_emb_dim)
        t_emb = self.time_mlp(self.time_embed(t))

        # Encoder
        e1 = self.enc1_b(self.enc1_a(x, t_emb), t_emb)     # (B, 64, 64, 64)
        e2 = self.enc2_b(self.enc2_a(self.down1(e1), t_emb), t_emb)   # (B, 128, 32, 32)
        e3 = self.enc3_b(self.enc3_a(self.down2(e2), t_emb), t_emb)   # (B, 256, 16, 16)
        e3 = self.attn_enc3(e3)                            # (B, 256, 16, 16)

        # Bottleneck
        mid = self.mid_a(self.down3(e3), t_emb)             # (B, 256, 8, 8)
        mid = self.mid_attn(mid)
        mid = self.mid_b(mid, t_emb)                        # (B, 256, 8, 8)

        # Decoder with skip connections
        d3 = self.dec3_a(torch.cat([e3, self.up3(mid)], dim=1), t_emb)  # (B, 256, 16, 16)
        d3 = self.dec3_b(d3, t_emb)                                     # (B, 128, 16, 16)
        d3 = self.attn_dec3(d3)                                         # (B, 128, 16, 16)

        d2 = self.dec2_a(torch.cat([e2, self.up2(d3)], dim=1), t_emb)  # (B, 128, 32, 32)
        d2 = self.dec2_b(d2, t_emb)                                      # (B, 64, 32, 32)

        d1 = self.dec1_a(torch.cat([e1, self.up1(d2)], dim=1), t_emb)  # (B, 64, 64, 64)
        d1 = self.dec1_b(d1, t_emb)                                      # (B, 64, 64, 64)

        # Output
        return self.conv_out(self.out_act(self.out_norm(d1)))  # (B, 3, 64, 64)
