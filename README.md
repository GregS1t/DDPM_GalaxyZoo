# DDPM — Galaxy Zoo 2

Denoising Diffusion Probabilistic Model (DDPM) trained to generate images of
spiral galaxies from the [Galaxy Zoo 2](https://data.galaxyzoo.org/) dataset.

This project is part of a broader research effort at the Observatoire de Paris –
PSL University exploring generative deep learning models applied to galaxy
morphology datasets.

---

## Model

The model follows Ho et al. (2020) with the following design choices:

- **Noise predictor**: UNet with sinusoidal time embeddings and SiLU activations,
  trained on 64×64 RGB images.
- **Noise schedule**: cosine schedule (Nichol & Dhariwal, 2021), better suited to
  high-contrast astronomical images than the original linear schedule.
- **Training**: AdamW optimizer, cosine LR annealing with linear warmup,
  mixed-precision (AMP), gradient clipping.

---

## Project structure

```
ddpm.py                         # DDPMModel: forward/reverse diffusion, cosine schedule
unet.py                         # UNet noise predictor (ResBlock + sinusoidal embeddings)
galaxy_zoo_dataset.py           # GalaxyZooDataset and custom collate function
gpu_utils.py                    # GPU selection utilities
viz.py                          # Visualisation: image grids, noising sequence, denoising GIF

GZ2_DDPM_training_v3.py        # Training script
GZ2_DDPM_infer.py              # Inference: generates images from a saved checkpoint
GZ2_DDPM_eval.py               # Evaluation: FID score, pixel distribution, denoising GIF

param_GZ2.json                  # All hyperparameters and paths
```

---

## Requirements

```bash
pip install torch torchvision tqdm tensorboard pytorch-fid imageio
```

---

## Configuration

All paths and hyperparameters are set in `param_GZ2.json`:

| Key | Description | Default |
|---|---|---|
| `data_dir` | Path to the Galaxy Zoo 2 image directory | — |
| `project_dir` | Project root (outputs written here) | — |
| `n_galaxies` | Number of spiral galaxies to use for training | 141000 |
| `n_steps` | Number of diffusion timesteps T | 1000 |
| `beta_start` / `beta_end` | Beta schedule bounds (used only for linear schedule) | 1e-4 / 0.02 |
| `beta_schedule` | `"cosine"` or `"linear"` | `"cosine"` |
| `lr` | Learning rate | 2e-4 |
| `warmup_epochs` | Linear warmup duration before cosine annealing | 10 |
| `n_epochs` | Maximum number of training epochs | 300 |
| `patience` | Early stopping patience (in epochs) | 20 |
| `batch_size` | Training batch size | 64 |

---

## Usage

**Train:**
```bash
python GZ2_DDPM_training_v3.py
```

**Generate images from a checkpoint:**
```bash
python GZ2_DDPM_infer.py
```

**Evaluate (FID, pixel distribution, denoising GIF):**
```bash
python GZ2_DDPM_eval.py
```

**Monitor training:**
```bash
tensorboard --logdir=<project_dir>/tensorboard_logs
```

---

## Data

The Galaxy Zoo 2 dataset is available at [https://data.galaxyzoo.org/](https://data.galaxyzoo.org/).
The catalog file (`gz2_hart16.csv`) and the filename mapping (`gz2_filename_mapping.csv`)
must be placed in `data_dir`. Only spiral galaxies (classes starting with `S`) are used.

---

## References

- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
  https://arxiv.org/abs/2006.11239
- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML 2021.
  https://arxiv.org/abs/2102.09672
- Hart et al., *Galaxy Zoo 2: detailed morphological classifications*, MNRAS 2016.
  https://doi.org/10.1093/mnras/stw1588
