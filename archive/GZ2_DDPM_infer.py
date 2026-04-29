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
from archive.unet import UNet

from viz import show_images, show_noising_sequence
from galaxy_zoo_dataset import GalaxyZooDataset, custom_collate
from gpu_utils import setup_device

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
    with open(PARAMETERS_FILE, "r") as f:
        p = json.load(f)

    n_steps = p["n_steps"]
    min_beta = p["beta_start"]
    max_beta = p["beta_end"]
    project_dir = p["project_dir"]
    output_dir = p["output_dir"]
    output_model_name = p["output_model"]
    data_dir = p["data_dir"]
    catalog_file = p["catalog_file"]
    mapping_file = p["mapping_file"]

    device = setup_device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Real galaxy dataset (used for reference quantiles and noising sequence) ---
    df = pd.read_csv(os.path.join(data_dir, catalog_file))
    df_mapping = pd.read_csv(os.path.join(data_dir, mapping_file))

    # For now we keep all galaxies regardless of morphology.
    df_spiral = pd.merge(
        df[df["gz2_class"].str.startswith("S")],
        df_mapping, left_on="dr7objid", right_on="objid"
    ).sample(n=1280, random_state=42)

    transform = Compose([Resize((IMAGE_H, IMAGE_W)), ToTensor()])
    dataset = GalaxyZooDataset(df_spiral, data_dir, transform=transform)

    ref_loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            collate_fn=custom_collate)
    noising_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                collate_fn=custom_collate)

    # --- Reference quantiles for histogram matching ---
    # Disabled: histogram matching is sensitive to the quality of the
    # generative model. Re-enable once the model produces a distribution
    # close to the real data (i.e. after retraining without RandomRotation).
    # reference_quantiles = compute_reference_quantiles(ref_loader,
    #                                                   n_quantiles=256,
    #                                                   n_batches=20)

    # --- Load model ---
    checkpoint_path = os.path.join(project_dir, output_dir, output_model_name)
    beta_schedule = p.get("beta_schedule", "linear")
    ddpm = DDPMModel(UNet(n_steps), n_steps=n_steps, min_beta=min_beta,
                     max_beta=max_beta, device=device,
                     image_chw=(IMAGE_C, IMAGE_H, IMAGE_W),
                     beta_schedule=beta_schedule)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    ddpm.load_state_dict(state)
    ddpm.eval()
    print(f"Checkpoint loaded from {checkpoint_path}")

    # --- Generate ---
    images = ddpm.generate(N_SAMPLES, IMAGE_C, IMAGE_H, IMAGE_W)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(project_dir, output_dir), exist_ok=True)

    save_path = os.path.join(project_dir, output_dir,
                             f"generated_GZ2_{timestamp}.png")
    show_images(images, suptitle="Generated Galaxy Zoo 2 images",
                save_path=save_path, show=True)
    print(f"Saved to {save_path}")

    # --- Noising sequence on one real galaxy ---
    noise_path = os.path.join(project_dir, output_dir,
                              f"noising_sequence_{timestamp}.png")
    show_noising_sequence(ddpm, noising_loader, device, n_steps=10,
                          save_path=noise_path, show=True)
    print(f"Noising sequence saved to {noise_path}")
