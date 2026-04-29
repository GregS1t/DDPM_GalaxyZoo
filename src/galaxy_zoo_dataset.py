# Galaxy Zoo 2 dataset loader.
#
# Author: Grégory Sainton
# Institution: Observatoire de Paris - PSL University

import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from PIL import Image


class GalaxyZooDataset(Dataset):
    """
    Dataset for Galaxy Zoo 2 images.

    Args:
        df (pd.DataFrame): Metadata table. Must contain columns 'asset_id'
            (image filename without extension) and 'gz2_class'.
        data_dir (str): Root directory. Images are expected at
            data_dir/images/<asset_id>.jpg.
        transform (callable, optional): Transform applied to each image.
    """

    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, img_path, gz2_class) or None if file is missing.
        """
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, "images",
                                str(row["asset_id"]) + ".jpg")
        gz2_class = str(row["gz2_class"])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_path, gz2_class
        except FileNotFoundError:
            return None


def custom_collate(batch):
    """
    Collate function that silently drops None entries (missing image files).

    Args:
        batch (list): Raw batch from the dataset.

    Returns:
        Collated batch, or None if all entries were missing.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)
