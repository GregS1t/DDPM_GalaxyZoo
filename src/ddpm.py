# DDPM model for image generation.
# Reference: Ho et al., "Denoising Diffusion Probabilistic Models", 2020.
# https://arxiv.org/abs/2006.11239
#
# Cosine noise schedule: Nichol & Dhariwal, "Improved DDPMs", 2021.
# https://arxiv.org/abs/2102.09672
#
# Author: Grégory Sainton
# Institution: Observatoire de Paris - PSL University

import os
import sys

import math
import numpy as np
import torch
import torch.nn as nn


_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
_SRC_DIR = os.path.join(_PROJECT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from transform_custom import AsinhStretch



def cosine_alpha_bars(n_steps, s=0.008):
    """
    Cosine noise schedule for alpha_bars (Nichol & Dhariwal, 2021).

    Produces a smoother degradation than the linear beta schedule, avoiding
    too-rapid destruction of image structure at early timesteps. Particularly
    suited to high-contrast images (dark background + bright source).

        alpha_bar(t) = cos^2( (t/T + s) / (1 + s) * pi/2 ) / cos^2( s/(1+s) * pi/2 )

    Args:
        n_steps (int): Number of diffusion timesteps T.
        s (float): Small offset to avoid alpha_bar(0) = 1 exactly. Default: 0.008.

    Returns:
        torch.Tensor: alpha_bar values of shape (n_steps,), in (0, 1].
    """
    steps = torch.arange(n_steps + 1, dtype=torch.float64)
    f = torch.cos((steps / n_steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bars = f / f[0]
    # Derive betas and clamp to avoid instability at large t
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    betas = betas.clamp(1e-8, 0.999)
    # Recompute alpha_bars from clamped betas for consistency
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    # Clamp to avoid alpha_bar = 1.0 exactly at t=0, which causes
    # division by zero in the reverse process (sqrt(1 - alpha_bar_0) = 0)
    alpha_bars = alpha_bars.clamp(1e-8, 1.0 - 1e-8)
    return alpha_bars.float()


def histogram_match(source, reference_quantiles, n_quantiles=256):
    """
    Match the histogram of source images to a reference distribution.

    Applies a per-channel, per-image non-linear mapping such that the output
    distribution matches the reference quantiles. This is the PyTorch
    equivalent of skimage.exposure.match_histograms.

    Args:
        source (torch.Tensor): Generated images (B, C, H, W), any range.
        reference_quantiles (torch.Tensor): Quantiles of the reference
            distribution, shape (C, n_quantiles), in [0, 1].
            Computed once from the real dataset via compute_reference_quantiles().
        n_quantiles (int): Number of quantile levels used.

    Returns:
        torch.Tensor: Remapped images (B, C, H, W), same range as reference.
    """
    B, C, H, W = source.shape
    quantile_levels = torch.linspace(0, 1, n_quantiles, device=source.device)
    out = torch.zeros_like(source)

    for c in range(C):
        src_c = source[:, c]  # (B, H, W)
        ref_q = reference_quantiles[c].to(source.device)  # (n_quantiles,)

        # Process each image independently to avoid searchsorted shape issues
        src_flat = src_c.flatten(1)  # (B, H*W)
        src_q = torch.quantile(src_flat, quantile_levels, dim=1).T  # (B, n_quantiles)

        matched_flat = torch.zeros_like(src_flat)
        for b in range(B):
            pixels = src_flat[b]   # (H*W,)
            src_cdf = src_q[b]     # (n_quantiles,)

            # Find bracketing indices in source CDF
            idx_hi = torch.searchsorted(
                src_cdf.contiguous(), pixels.contiguous()
            ).clamp(1, n_quantiles - 1)   # (H*W,)
            idx_lo = (idx_hi - 1).clamp(0, n_quantiles - 2)

            # Linear interpolation weight between the two bracketing quantiles
            src_lo = src_cdf[idx_lo]
            src_hi = src_cdf[idx_hi]
            denom = (src_hi - src_lo).clamp(min=1e-8)
            weight = (pixels - src_lo) / denom  # in [0, 1]

            # Interpolate between the corresponding reference quantile values
            ref_lo = ref_q[idx_lo]
            ref_hi = ref_q[idx_hi]
            matched_flat[b] = (ref_lo + weight * (ref_hi - ref_lo)).clamp(0, 1)

        out[:, c] = matched_flat.view(B, H, W)

    return out


def compute_reference_quantiles(loader, n_quantiles=256, n_batches=20):
    """
    Compute per-channel quantiles from a sample of real images.

    Call once before inference and pass the result to generate().

    Args:
        loader (DataLoader): Real image data loader (images in [0, 1]).
        n_quantiles (int): Number of quantile levels to compute.
        n_batches (int): Number of batches to sample (more = more accurate).

    Returns:
        torch.Tensor: Reference quantiles of shape (C, n_quantiles).
    """
    import torch
    quantile_levels = torch.linspace(0, 1, n_quantiles)
    all_pixels = [[] for _ in range(3)]  # one list per channel

    for i, batch in enumerate(loader):
        if batch is None or i >= n_batches:
            break
        images = batch[0]  # (B, C, H, W) in [0, 1]
        for c in range(images.shape[1]):
            all_pixels[c].append(images[:, c].flatten())

    ref_quantiles = torch.zeros(3, n_quantiles)
    for c in range(3):
        pixels_c = torch.cat(all_pixels[c])
        ref_quantiles[c] = torch.quantile(pixels_c, quantile_levels)

    return ref_quantiles


class DDPMModel(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020).

    Implements the forward (noising) process q(x_t | x_0) and the reverse
    (denoising) process via a learnable noise predictor network.

    Args:
        network (nn.Module): Noise prediction network (e.g. UNet).
        n_steps (int): Number of diffusion timesteps T.
        min_beta (float): Beta schedule start value (Ho et al.: 1e-4).
        max_beta (float): Beta schedule end value (Ho et al.: 0.02).
        device (torch.device): Computation device.
        image_chw (tuple): Image shape (C, H, W).
        beta_schedule (str): Noise schedule — "linear" (Ho et al., 2020) or
            "cosine" (Nichol & Dhariwal, 2021). Default: "cosine".
    """

    def __init__(self, network, n_steps, min_beta=1e-4, max_beta=0.02,
                 device=None, image_chw=(1, 28, 28), beta_schedule="cosine"):
        super(DDPMModel, self).__init__()
        self.network = network.to(device)
        self.device = device
        self.n_steps = n_steps
        self.image_chw = image_chw

        if beta_schedule == "cosine":
            self.alpha_bars = cosine_alpha_bars(n_steps).to(device)
            self.betas = torch.cat([
                torch.zeros(1, device=device),
                (1 - self.alpha_bars[1:] / self.alpha_bars[:-1]).clamp(0, 0.999)
            ])
        else:
            # Linear schedule (Ho et al., 2020)
            self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
            alphas = 1.0 - self.betas
            self.alpha_bars = torch.cumprod(alphas, dim=0).to(device)

        self.alphas = (1.0 - self.betas).to(device)
        self.beta_schedule = beta_schedule

    def q_sample(self, x_0, t, eps=None):
        """
        Forward diffusion process: q(x_t | x_0).

        Adds noise to x_0 at timestep t:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x_0 (torch.Tensor): Clean images (B, C, H, W).
            t (torch.Tensor): Timestep indices (B,).
            eps (torch.Tensor, optional): Noise. Sampled from N(0,I) if None.

        Returns:
            torch.Tensor: Noisy images x_t (B, C, H, W).
        """
        if eps is None:
            eps = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        return alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * eps

    def predict_eps(self, x_t, t):
        """
        Reverse process: predicts the noise eps_theta(x_t, t) via the network.

        Args:
            x_t (torch.Tensor): Noisy images (B, C, H, W).
            t (torch.Tensor): Timestep indices (B,).

        Returns:
            torch.Tensor: Predicted noise (B, C, H, W).
        """
        return self.network(x_t, t)

    def compute_loss(self, x_0, t, eps=None):
        """
        Training loss: L = ||eps - eps_theta(x_t, t)||^2.

        Args:
            x_0 (torch.Tensor): Clean images (B, C, H, W).
            t (torch.Tensor): Timestep indices (B,).
            eps (torch.Tensor, optional): Noise. Sampled if None.

        Returns:
            torch.Tensor: Scalar MSE loss.
        """
        if eps is None:
            eps = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, eps)
        eps_theta = self.predict_eps(x_t, t)
        return torch.mean((eps - eps_theta) ** 2)

    @torch.no_grad()
    def generate(self, n_samples, c, h, w, reference_quantiles=None, 
                 asinh_stretch=False, asinh_scale=1.0, histogram_matching=False):
        """
        Reverse diffusion sampling: generates images from pure noise.

        Implements Algorithm 2 from Ho et al. (2020):
            x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta)
                      + sigma_t * z,   z ~ N(0, I) if t > 0

        Args:
            n_samples (int): Number of images to generate.
            c (int): Number of image channels.
            h (int): Image height.
            w (int): Image width.

        Args:
            reference_quantiles (torch.Tensor, optional): Per-channel quantiles
                of the real dataset, shape (C, n_quantiles), computed via
                compute_reference_quantiles(). If provided, histogram matching
                is applied to align generated pixel distributions with the real
                data. Recommended for scientific use.

        Returns:
            torch.Tensor: Generated images (n_samples, C, H, W) in [0, 1].
        """
        x_t = torch.randn(n_samples, c, h, w).to(self.device)

        for t in reversed(range(self.n_steps)):
            t_batch = (torch.ones(n_samples, 1) * t).long().to(self.device)
            eps_theta = self.predict_eps(x_t, t_batch)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

            # Guard against division by zero when alpha_bar_t -> 1 (t=0)
            denom = (1 - alpha_bar_t).sqrt().clamp(min=1e-8)
            x_t = (1 / alpha_t.sqrt()) * (
                x_t - (1 - alpha_t) / denom * eps_theta
            )

            if t > 0:
                sigma_t = self.betas[t].sqrt()
                x_t = x_t + sigma_t * torch.randn_like(x_t)

        x_t = x_t.float()
        if asinh_stretch:
            stretcher = AsinhStretch(scale=asinh_scale)
            x_t = stretcher.inverse(x_t)
            

        if reference_quantiles is not None:
            # Histogram matching: align generated pixel distributions with the
            # real dataset distribution, per channel. This is strictly more
            # powerful than global mean/std normalisation and preserves the
            # photometric properties of the reference images.
            x_t = histogram_match(x_t, reference_quantiles)
        else:
            # Fallback: simple percentile rescaling to [0, 1]
            flat = x_t.flatten(1)
            p_low = torch.quantile(flat, 0.001, dim=1).view(-1, 1, 1, 1)
            p_high = torch.quantile(flat, 0.999, dim=1).view(-1, 1, 1, 1)
            x_t = ((x_t - p_low) / (p_high - p_low + 1e-8)).clamp(0, 1)

        return x_t
