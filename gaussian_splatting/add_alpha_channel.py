#!/bin/python3

"""Script that adds an alpha channel based on binary masks in both .npy and .png formats."""

import argparse
import os

from pathlib import Path

import numpy as np

from PIL import Image
from tqdm import tqdm


def load_mask(mask_path):
    """Loads a mask, whether it's in .npy or .png format."""
    if mask_path.suffix == ".npy":
        mask = np.load(mask_path)
    elif mask_path.suffix == ".png":
        mask = np.asarray(Image.open(mask_path).convert("L"))
    else:
        raise ValueError(f"Unsupported mask format: {mask_path.suffix}")
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="The path to the folder containing the images.",
    )
    parser.add_argument(
        "--masks",
        type=str,
        required=True,
        help="The path to the folder containing the binary masks.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="The path to the folder where the images with alpha channel are written to.",
    )
    args = parser.parse_args()
    image_dir_path = Path(args.images)
    mask_dir_path = Path(args.masks)
    out_dir_path = Path(args.out)

    image_files = sorted(list(image_dir_path.glob("*.png")))
    mask_files = sorted(
        list(mask_dir_path.glob("*.npy")) + list(mask_dir_path.glob("*.png"))
    )

    assert len(image_files) > 0, f"No images found in {image_dir_path}"
    assert len(mask_files) > 0, f"No masks found in {mask_dir_path}"
    assert len(image_files) == len(mask_files), "Number of images and masks must match."

    out_dir_path.mkdir(exist_ok=True)

    for img_path, mask_path in tqdm(
        zip(image_files, mask_files),
        total=len(image_files),
        desc="Adding alpha channel",
    ):
        img = np.asarray(Image.open(img_path).convert("RGB"))
        mask = load_mask(mask_path)

        # Ensure the mask has the correct shape for concatenation
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]

        img_w_alpha = np.concatenate((img, mask), axis=-1)

        out_path = out_dir_path / img_path.name
        img_w_alpha_pil = Image.fromarray(img_w_alpha)
        img_w_alpha_pil.save(out_path)


if __name__ == "__main__":
    main()
