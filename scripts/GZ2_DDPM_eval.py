# Evaluation script for the trained DDPMModel.
# Computes FID, pixel distribution comparison, and generates a denoising GIF.
# Reads hyperparameters from param_GZ2.json.
#
# Requires: pip install pytorch-fid imageio
#
# Author: Grégory Sainton
# Institution: Observatoire de Paris - PSL University

import os
import sys
import json
import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import TensorDataset
import pandas as pd
from tqdm import tqdm

from ddpm import DDPMModel
from unet_v2 import UNetV2
from galaxy_zoo_dataset import GalaxyZooDataset, custom_collate
from gpu_utils import setup_device
from viz import make_denoising_gif
from metrics import (build_inception, extract_features, compute_fid,
                     precision_recall, density_coverage)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
N_SAMPLES_FID = 1000
N_SAMPLES_DIST = 500
N_REAL_FID = 1000
BATCH_SIZE_GEN = 16
IMAGE_C, IMAGE_H, IMAGE_W = 3, 64, 64


# ------------------------------------------------------------------
# Pixel distribution
# ------------------------------------------------------------------
def plot_pixel_distribution(real_images, generated_images, save_path=None,
                             show=True):
    """
    Compare the pixel intensity distributions of real vs generated images.

    Plots one histogram per RGB channel overlaid for both sets.

    Args:
        real_images (torch.Tensor): Real images (N, 3, H, W) in [0, 1].
        generated_images (torch.Tensor): Generated images (N, 3, H, W) in [0, 1].
        save_path (str, optional): Path to save the figure.
        show (bool): Whether to call plt.show().
    """
    channel_names = ["Red", "Green", "Blue"]
    colors = ["red", "green", "blue"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for c, (ax, name, color) in enumerate(zip(axes, channel_names, colors)):
        real_vals = real_images[:, c].flatten().numpy()
        gen_vals = generated_images[:, c].flatten().numpy()

        ax.hist(real_vals, bins=64, range=(0, 1), alpha=0.5,
                color=color, label="Real", density=True)
        ax.hist(gen_vals, bins=64, range=(0, 1), alpha=0.5,
                color="gray", label="Generated", density=True)
        ax.set_title(f"{name} channel")
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Pixel intensity distribution — Real vs Generated", fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Pixel distribution plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# FID computation
# ------------------------------------------------------------------
def save_images_to_dir(images, out_dir):
    """
    Save a batch of tensors as PNG files for optional visual inspection.

    Args:
        images (torch.Tensor): Images (N, C, H, W) in [0, 1].
        out_dir (str): Output directory (created if missing).
    """
    from torchvision.utils import save_image
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        save_image(img.clamp(0, 1), os.path.join(out_dir, f"{i:05d}.png"))


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="param_GZ2.json",
                        help="Path to the JSON parameter file.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--gif", action="store_true", default=False,
                        help="Generate denoising GIF (slow, disabled by default).")
    args = parser.parse_args()

    if not os.path.exists(args.params):
        print(f"'{args.params}' not found.")
        sys.exit(1)

    with open(args.params, "r") as f:
        p = json.load(f)

    n_steps = p["n_steps"]
    min_beta = p["beta_start"]
    max_beta = p["beta_end"]
    project_dir = p["project_dir"]
    output_dir = p["output_dir"]
    data_dir = p["data_dir"]
    catalog_file = p["catalog_file"]
    mapping_file = p["mapping_file"]
    precision_recall_metric = p.get("precision_recall", False)
    precision_recall_k = p.get("precision_recall_k", 3)
    density_coverage_k = p.get("density_coverage_k", 5)
    attention_resolutions = p.get("attention_resolutions", [8, 16])
    asinh_stretch = p.get("asinh_stretch", False)
    asinh_scale = p.get("asinh_scale", 0.02)
    beta_schedule = p.get("beta_schedule", "cosine")
    morphology = p.get("morphology", "S")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_dir, output_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = setup_device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    checkpoint_path = args.checkpoint
    ddpm = DDPMModel(UNetV2(n_steps, attention_resolutions=attention_resolutions),
                     n_steps=n_steps, min_beta=min_beta,
                     max_beta=max_beta, device=device,
                     image_chw=(IMAGE_C, IMAGE_H, IMAGE_W),
                     beta_schedule=beta_schedule)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    ddpm.load_state_dict(state)

    if "ema_shadow" in ckpt:
        ema_shadow = ckpt["ema_shadow"]
        for name, param in ddpm.named_parameters():
            if name in ema_shadow:
                param.data.copy_(ema_shadow[name])
        print("EMA weights loaded from checkpoint")
    else:
        print("No EMA weights found in checkpoint, using raw model weights")

    ddpm.eval()
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}  "
          f"val_loss: {ckpt.get('val_loss', 'unknown'):.4f}")

    # --- Real data loader ---
    df = pd.read_csv(os.path.join(data_dir, catalog_file))
    df_mapping = pd.read_csv(os.path.join(data_dir, mapping_file))
    df_spiral = pd.merge(
        df[df["gz2_class"].str.startswith(morphology)],
        df_mapping, left_on="dr7objid", right_on="objid"
    ).sample(n=max(N_REAL_FID, N_SAMPLES_DIST), random_state=42)

    transform = Compose([Resize((IMAGE_H, IMAGE_W)), ToTensor()])
    dataset = GalaxyZooDataset(df_spiral, data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        collate_fn=custom_collate)

    # --- Collect real images ---
    print(f"Collecting {N_REAL_FID} real images...")
    real_images = []
    for batch in tqdm(loader):
        if batch is None:
            continue
        real_images.append(batch[0])
        if sum(b.shape[0] for b in real_images) >= N_REAL_FID:
            break
    real_images = torch.cat(real_images, dim=0)[:N_REAL_FID]

    # --- Generate images ---
    print(f"Generating {N_SAMPLES_FID} images...")
    generated_images = []
    n_generated = 0
    with torch.no_grad():
        while n_generated < N_SAMPLES_FID:
            n_batch = min(BATCH_SIZE_GEN, N_SAMPLES_FID - n_generated)
            batch = ddpm.generate(n_batch, IMAGE_C, IMAGE_H, IMAGE_W)
            generated_images.append(batch.cpu())
            n_generated += n_batch
            print(f"  {n_generated}/{N_SAMPLES_FID}", end="\r")
    generated_images = torch.cat(generated_images, dim=0)
    print()

    # --- Pixel distribution ---
    print("Computing pixel distribution...")
    dist_path = os.path.join(out_dir, f"pixel_distribution_{timestamp}.png")
    plot_pixel_distribution(
        real_images[:N_SAMPLES_DIST].cpu(),
        generated_images[:N_SAMPLES_DIST].cpu(),
        save_path=dist_path, show=False
    )

    # --- Extract features once, reuse for all metrics ---
    print("Extracting InceptionV3 features...")
    inception = build_inception(device)
    phi_r = extract_features(loader, inception, device, max_samples=N_REAL_FID)
    gen_dataset = TensorDataset(generated_images)
    gen_loader = DataLoader(gen_dataset, batch_size=64, shuffle=False)
    phi_g = extract_features(gen_loader, inception, device, max_samples=N_SAMPLES_FID)

    # FID
    print("Computing FID...")
    fid_score = compute_fid(phi_r, phi_g)
    print(f"FID score: {fid_score:.2f}")

    # Precision-Recall
    prec, rec = None, None
    if precision_recall_metric:
        print("Computing Precision-Recall...")
        prec, rec = precision_recall(phi_r, phi_g, k=precision_recall_k)
        print(f"Precision: {prec:.4f}  Recall: {rec:.4f}")

    # Density-Coverage
    print("Computing Density-Coverage...")
    dens, cov = density_coverage(phi_r, phi_g, k=density_coverage_k)
    print(f"Density:   {dens:.4f}  Coverage: {cov:.4f}")

    # --- Save all results to a single file ---
    results_path = os.path.join(out_dir, f"results_{timestamp}.txt")
    with open(results_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DDPM Evaluation Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("[Run configuration]\n")
        f.write(f"checkpoint:            {checkpoint_path}\n")
        f.write(f"timestamp:             {timestamp}\n")
        f.write(f"morphology:            {morphology}\n")
        f.write(f"beta_schedule:         {beta_schedule}\n")
        f.write(f"attention_resolutions: {attention_resolutions}\n")
        f.write(f"asinh_stretch:         {asinh_stretch}\n")
        if asinh_stretch:
            f.write(f"asinh_scale:           {asinh_scale}\n")
        f.write(f"n_steps:               {n_steps}\n")
        f.write(f"beta_start:            {min_beta}\n")
        f.write(f"beta_end:              {max_beta}\n")
        f.write(f"ema_weights:           {'yes' if 'ema_shadow' in ckpt else 'no'}\n")
        f.write("\n")

        f.write("[Checkpoint info]\n")
        f.write(f"epoch:                 {ckpt.get('epoch', 'unknown')}\n")
        f.write(f"val_loss:              {ckpt.get('val_loss', 'unknown')}\n")
        f.write("\n")

        f.write("[Evaluation settings]\n")
        f.write(f"n_real:                {N_REAL_FID}\n")
        f.write(f"n_generated:           {N_SAMPLES_FID}\n")
        f.write(f"precision_recall_k:    {precision_recall_k}\n")
        f.write(f"density_coverage_k:    {density_coverage_k}\n")
        f.write("\n")

        f.write("[Metrics]\n")
        f.write(f"FID:                   {fid_score:.4f}\n")
        if prec is not None:
            f.write(f"Precision (k={precision_recall_k}):         {prec:.4f}\n")
            f.write(f"Recall (k={precision_recall_k}):            {rec:.4f}\n")
        else:
            f.write("Precision:             N/A (precision_recall=false)\n")
            f.write("Recall:                N/A (precision_recall=false)\n")
        f.write(f"Density (k={density_coverage_k}):          {dens:.4f}\n")
        f.write(f"Coverage (k={density_coverage_k}):         {cov:.4f}\n")
        f.write("\n")

        f.write("[Full param file]\n")
        f.write(json.dumps(p, indent=4))
        f.write("\n")

    print(f"Results saved to {results_path}")

    # --- Denoising GIF (optional) ---
    if args.gif:
        print("Generating denoising GIF...")
        gif_path = os.path.join(out_dir, f"denoising_{timestamp}.gif")
        make_denoising_gif(ddpm, loader, device, n_frames=30,
                           gif_path=gif_path, fps=10)
        print(f"GIF saved to {gif_path}")
    else:
        print("Denoising GIF skipped (use --gif to enable).")
