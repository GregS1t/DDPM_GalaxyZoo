# Inference script: generates images from a trained DDPMModel checkpoint.
# Reads hyperparameters from param_GZ2.json.
#
# Author: Grégory Sainton
# Institution: Observatoire de Paris - PSL University

import os
import json
import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from ddpm import DDPMModel, compute_reference_quantiles
from unet_v2 import UNetV2
from viz import show_images, show_noising_sequence
from galaxy_zoo_dataset import GalaxyZooDataset, custom_collate
from gpu_utils import setup_device
from transform_custom import AsinhStretch
# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
PARAMETERS_FILE = "param_GZ2.json"
N_SAMPLES = 16
IMAGE_C, IMAGE_H, IMAGE_W = 3, 64, 64

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
    args = parser.parse_args()

    with open(args.params, "r") as f:
        p = json.load(f)

    # Read parameters from json file
    n_steps = p["n_steps"]
    min_beta = p["beta_start"]
    max_beta = p["beta_end"]
    project_dir = p["project_dir"]
    output_dir = p["output_dir"]
    data_dir = p["data_dir"]
    catalog_file = p["catalog_file"]
    mapping_file = p["mapping_file"]
    asinh_scale = p["asinh_scale"]
    asinh_stretch = p.get("asinh_stretch", False)
    histogram_matching = p.get("histogram_matching", False)
    attention_resolutions = p.get("attention_resolutions", [8, 16])
    beta_schedule = p.get("beta_schedule", "cosine")


    device = setup_device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Real galaxy dataset (used for reference quantiles and noising sequence) ---
    df = pd.read_csv(os.path.join(data_dir, catalog_file))
    df_mapping = pd.read_csv(os.path.join(data_dir, mapping_file))

    # For now we keep all galaxies regardless of morphology.
    df_spiral = pd.merge(
        df[df["gz2_class"].str.startswith("S")],
        df_mapping, left_on="dr7objid", right_on="objid"
    ).sample(n=1280, random_state=42)

    base_transforms = [AsinhStretch(asinh_scale)] if asinh_stretch else []

    transform = Compose([
        Resize((IMAGE_H, IMAGE_W)),
        ToTensor(),
        *base_transforms,])

    dataset = GalaxyZooDataset(df_spiral, data_dir, transform=transform)

    ref_loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            collate_fn=custom_collate)
    noising_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                collate_fn=custom_collate)


    if histogram_matching:
        reference_quantiles = compute_reference_quantiles(ref_loader,
                                                          n_quantiles=256,
                                                          n_batches=20)
        print("Reference quantiles computed for histogram matching")
    else:
        reference_quantiles = None

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
    # Apply EMA weights for smoother inference if available in the checkpoint
    if "ema_shadow" in ckpt:
        ema_shadow = ckpt["ema_shadow"]
        for name, param in ddpm.named_parameters():
            if name in ema_shadow:
                param.data.copy_(ema_shadow[name])
        print("EMA weights loaded from checkpoint")
    else:
        print("No EMA weights found in checkpoint, using raw model weights")

    ddpm.eval()
    print(f"Checkpoint loaded from {checkpoint_path}")

    # --- Generate ---
    images = ddpm.generate(N_SAMPLES, IMAGE_C, IMAGE_H, IMAGE_W,
                       reference_quantiles=reference_quantiles)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(project_dir, output_dir), exist_ok=True)

    save_path = os.path.join(project_dir, output_dir,
                             f"generated_GZ2_{timestamp}.png")
    show_images(images, suptitle="Generated Galaxy Zoo 2 images",
                save_path=save_path, show=True)
    print(f"Saved to {save_path}")

    # --- Noising sequence on one real galaxy ---
    # noise_path = os.path.join(project_dir, output_dir,
    #                           f"noising_sequence_{timestamp}.png")
    # show_noising_sequence(ddpm, noising_loader, device, n_steps=10,
    #                       save_path=noise_path, show=True)
    #print(f"Noising sequence saved to {noise_path}")
