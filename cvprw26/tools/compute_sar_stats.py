"""Compute global mean and std for SAR images using Welford's online algorithm.

Usage::

    # Compute on train split only (recommended to avoid data leakage)
    python tools/compute_sar_stats.py --image-dir data/BRIGHT/post-event --split data/BRIGHT/splits/train_set.txt

    # Compute on all images
    python tools/compute_sar_stats.py --image-dir data/BRIGHT/post-event --split ''
"""

import argparse
import glob
import os

import numpy as np
import rasterio


def compute_stats(image_dir: str, split_file: str = None) -> dict:
    """Compute global mean/std using Welford's online algorithm."""
    if split_file:
        with open(split_file) as f:
            sample_ids = [line.strip() for line in f if line.strip()]
        tif_files = []
        for sid in sample_ids:
            path = os.path.join(image_dir, f"{sid}_post_disaster.tif")
            if os.path.isfile(path):
                tif_files.append(path)
        tif_files.sort()
        print(f"Split file: {split_file}")
        print(f"Sample IDs: {len(sample_ids)}, matched tif: {len(tif_files)}")
    else:
        tif_files = sorted(glob.glob(os.path.join(image_dir, "*.tif")))

    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {image_dir}")

    n = 0
    mean = 0.0
    m2 = 0.0

    for i, path in enumerate(tif_files):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float64).ravel()

        batch_n = data.size
        batch_mean = data.mean()
        batch_var = data.var()

        delta = batch_mean - mean
        new_n = n + batch_n
        mean = mean + delta * batch_n / new_n
        m2 = m2 + batch_var * batch_n + delta ** 2 * n * batch_n / new_n
        n = new_n

        if (i + 1) % 100 == 0 or (i + 1) == len(tif_files):
            print(f"[{i + 1}/{len(tif_files)}] running mean={mean:.4f}, std={np.sqrt(m2 / n):.4f}")

    std = np.sqrt(m2 / n)

    return {
        "num_images": len(tif_files),
        "num_pixels": n,
        "pixel_mean_raw": round(float(mean), 4),
        "pixel_std_raw": round(float(std), 4),
        "pixel_mean_norm": round(float(mean / 255.0), 6),
        "pixel_std_norm": round(float(std / 255.0), 6),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute SAR image global mean/std")
    parser.add_argument("--image-dir", default="data/BRIGHT/post-event",
                        help="Directory containing SAR .tif files")
    parser.add_argument("--split", default="data/BRIGHT/splits/train_set.txt",
                        help="Split file with sample IDs. Defaults to train split to avoid data leakage. "
                             "Pass --split '' to use all images.")
    args = parser.parse_args()

    split = args.split if args.split else None
    stats = compute_stats(args.image_dir, split)

    split_label = f" (split: {os.path.basename(args.split)})" if split else " (all images)"
    print(f"\n========== SAR Normalization Stats{split_label} ==========")
    print(f"Images:          {stats['num_images']}")
    print(f"Total pixels:    {stats['num_pixels']:,}")
    print(f"Mean (raw):      {stats['pixel_mean_raw']}")
    print(f"Std  (raw):      {stats['pixel_std_raw']}")
    print(f"Mean (/255):     {stats['pixel_mean_norm']}")
    print(f"Std  (/255):     {stats['pixel_std_norm']}")
    print("=" * (46 + len(split_label)))
    print()
    print("# Fill these values into config/disaster.yaml:")
    print(f"pixel_mean: {stats['pixel_mean_norm']}")
    print(f"pixel_std: {stats['pixel_std_norm']}")


if __name__ == "__main__":
    main()
