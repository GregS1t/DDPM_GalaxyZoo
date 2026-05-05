# Custom transforms for GZ2 dataset
#
# Author: Grégory Sainton
# Lab: Observatoire de Paris - PSL University


import torch
import random

from torchvision.transforms import functional as TF


class RandomDiscreteRotation:
    """
    Rotate by a uniformly chosen multiple of 90 degrees: 0, 90, 180, or 270
    to preserve the PSF.
    """

    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle)


class AsinhStretch:
    """
    Apply an arcsinh stretch to enhance faint features in astronomical images.
    To be applyed after ToTensor()
    """

    def __init__(self, scale=0.02):
        self.scale = scale

    def __call__(self, img):

        img_stretched = torch.asinh(img / self.scale) / torch.asinh(torch.tensor(1.0 / self.scale))
        return img_stretched.clamp(0,1)
    
    def inverse(self, img_stretched):
        img = torch.sinh(img_stretched * torch.asinh(torch.tensor(1.0 / self.scale))) * self.scale
        return img.clamp(0,1)