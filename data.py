#!/usr/bin/env python
"""Dataset and data-related utilities."""

import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Any, Tuple, Optional
from utils import load_obj


class PatchClassificationDataset(Dataset):
    """
    Custom dataset for patch classification.

    Args:
        data (Any): DataFrame or path to CSV file containing filenames/labels.
        image_dir (str): Path to the directory containing images.
        transforms (Any): Albumentations transforms.
        image_col (str): Column name for image filenames in the CSV/dataframe.
    """

    def __init__(
        self,
        data: Any,
        image_dir: str,
        transforms: Optional[Any] = None,
        image_col: str = "filename"
    ) -> None:
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data

        valid_rows = []
        for idx, row in df.iterrows():
            primary_path = os.path.join(image_dir, row[image_col])
            if os.path.exists(primary_path):
                valid_rows.append(row)
            else:
                print(f"Warning: Image not found for row {idx}: {primary_path}. Skipping.")

        self.df = pd.DataFrame(valid_rows)
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_col = image_col

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get one sample of the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Any, int]: (image tensor, label).
        """
        row = self.df.iloc[idx]
        primary_path = os.path.join(self.image_dir, row[self.image_col])
        if os.path.exists(primary_path):
            image_path = primary_path
        else:
            raise FileNotFoundError(f"Image not found: {primary_path}")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        label = int(row["label"])
        return image, label
