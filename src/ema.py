# Exponential Moving Average (EMA) of model weights.
# Used at inference to produce smoother, higher-quality generated images
# by averaging out the per-batch fluctuations of the training weights.
#
# Reference: Ho et al., "Denoising Diffusion Probabilistic Models", 2020.
# https://arxiv.org/abs/2006.11239
#
# Inspired by: https://medium.com/@heyamit10/exponential-moving-average-ema-in-pytorch-eb8b6f1718eb
#
# Author: Grégory Sainton
# Lab: Observatoire de Paris - PSL University


class EMA:
    """
    Exponential Moving Average of model weights (Ho et al., 2020).

    Maintains a shadow copy of the model parameters updated after each
    optimizer step via:
        shadow = decay * shadow + (1 - decay) * current_weights

    Use apply() before inference and restore() to resume training.

    Args:
        model (nn.Module): Model to track.
        decay (float): EMA decay factor. Default: 0.9999.
    """

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights after each optimizer step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply(self):
        """Replace model weights with EMA weights before inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore training weights after inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}
