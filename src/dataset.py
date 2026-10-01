import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

LABELS = ["epidural", "intraparenchymal", "intraventricular",
          "subarachnoid", "subdural", "any"]


class WindowLevelJitter:
    """Domain-specific CT augmentation: simulates re-reading the same slice at a
    slightly different window level/width, the single most common source of
    real-world appearance variation for a fixed CT volume. Implemented as a
    bounded brightness/contrast jitter rather than generic ColorJitter so the
    range stays within what a radiologist's window adjustment would produce."""

    def __init__(self, brightness=0.08, contrast=0.08, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        img = TF.adjust_brightness(img, b)
        img = TF.adjust_contrast(img, c)
        return img


class GaussianNoise:
    """Simulates scanner/reconstruction noise. Applied post-ToTensor."""

    def __init__(self, std=0.02, p=0.3):
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std


class ICHDataset(Dataset):
    def __init__(self, csv_path, img_dir, train=True, split=None):
        """
        csv_path: labels.csv (optionally containing a 'split' column, as
                  written by prep_from_dicom.py)
        split: if the csv has a 'split' column ('train'/'val'/'test'), filter
               to just that split. Ignored if the column doesn't exist.
        """
        df = pd.read_csv(csv_path)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        self.df = df
        self.img_dir = img_dir

        # CT-specific augmentation choices:
        # - NO vertical flip: CT slices have a fixed anatomical up/down
        #   (skull vs. neck), and flipping vertically produces an anatomically
        #   impossible image.
        # - Horizontal flip IS valid: the brain is broadly left-right
        #   symmetric, and hemorrhage location isn't tied to a fixed side.
        # - Small rotation only: real patient positioning varies by a few
        #   degrees, not tens of degrees.
        # - Window-level/width jitter: reproduces the actual variability of
        #   reading the same HU data at a different clinical window.
        if train:
            self.pil_tf = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                WindowLevelJitter(brightness=0.08, contrast=0.08, p=0.5),
            ])
            self.tensor_tf = T.Compose([
                T.ToTensor(),
                GaussianNoise(std=0.02, p=0.3),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.pil_tf = T.Compose([])
            self.tensor_tf = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, f"{row['ImageID']}.png")).convert("RGB")
        img = self.pil_tf(img)
        img = self.tensor_tf(img)
        labels = torch.tensor(row[LABELS].values.astype("float32"))
        return img, labels
