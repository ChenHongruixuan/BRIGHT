"""BRIGHT dataset for pre-event RGB + post-event SAR fusion."""

import os

import numpy as np
import rasterio
import torch
import torch.utils.data
from pycocotools.coco import COCO


class BRIGHTDataset(torch.utils.data.Dataset):
    """Dataset for BRIGHT building damage instance segmentation (fusion mode).

    Each sample concatenates pre-event RGB (3 bands) and post-event SAR (1 band)
    into a 4-channel tensor [R, G, B, SAR].

    Args:
        ann_file:      Path to the unified COCO JSON (output of merge_coco_json.py).
        image_dir:     Directory containing post-event SAR tif images.
        pre_event_dir: Directory containing pre-event RGB tif images.
        transforms:    Optional callable for data augmentation.

    __getitem__ returns:
        image:  FloatTensor [4, H, W]
        target: dict with keys boxes, labels, masks, image_id, area, iscrowd
    """

    def __init__(
        self,
        ann_file: str,
        image_dir: str,
        pre_event_dir: str,
        transforms=None,
    ):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.pre_event_dir = pre_event_dir
        self.transforms = transforms

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self._validate_image_pairs()

    def _validate_image_pairs(self, max_examples: int = 5) -> None:
        """Fail fast if any sample in the annotation file is missing an input image."""
        missing = []
        for img_id in self.ids:
            fname = self.coco.imgs[img_id]["file_name"]
            post_path = os.path.join(self.image_dir, fname)
            pre_fname = fname.replace("_post_disaster.tif", "_pre_disaster.tif")
            pre_path = os.path.join(self.pre_event_dir, pre_fname)

            missing_parts = []
            if not os.path.isfile(post_path):
                missing_parts.append(f"post-event SAR: {post_path}")
            if not os.path.isfile(pre_path):
                missing_parts.append(f"pre-event RGB: {pre_path}")

            if missing_parts:
                missing.append((fname, missing_parts))

        if not missing:
            return

        example_lines = []
        for file_name, missing_parts in missing[:max_examples]:
            example_lines.append(f"  - {file_name}: {', '.join(missing_parts)}")

        remaining = len(missing) - len(example_lines)
        if remaining > 0:
            example_lines.append(f"  ... and {remaining} more samples")

        raise FileNotFoundError(
            f"Found {len(missing)} samples in {self.ann_file} with missing input files.\n"
            "BRIGHTDataset requires both post-event SAR and pre-event RGB for every annotation entry.\n"
            "Fix the dataset layout or regenerate annotations for the available subset.\n"
            + "\n".join(example_lines)
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple:
        coco = self.coco
        img_id = self.ids[idx]
        img_info = coco.imgs[img_id]
        file_name = img_info["file_name"]

        # Post-event SAR (1 band)
        img_path = os.path.join(self.image_dir, file_name)
        with rasterio.open(img_path) as src:
            sar = src.read(1).astype(np.float32) / 255.0  # [H, W]

        # Pre-event RGB (3 bands)
        pre_fname = file_name.replace("_post_disaster.tif", "_pre_disaster.tif")
        pre_path = os.path.join(self.pre_event_dir, pre_fname)
        with rasterio.open(pre_path) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0  # [3, H, W]

        image_np = np.concatenate([rgb, sar[np.newaxis]], axis=0)  # [4, H, W]
        image = torch.as_tensor(image_np, dtype=torch.float32)

        # Load annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            mask = coco.annToMask(ann)
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            masks.append(mask)
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) > 0:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
                "image_id": img_id,
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            }
        else:
            h_img, w_img = image_np.shape[-2], image_np.shape[-1]
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, h_img, w_img), dtype=torch.uint8),
                "image_id": img_id,
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class RandomVerticalFlip:
    """Randomly flip image and target vertically."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1).item() < self.prob:
            image = image.flip(-2)
            height = image.shape[-2]

            if target["boxes"].numel() > 0:
                boxes = target["boxes"]
                boxes_flipped = boxes.clone()
                boxes_flipped[:, 1] = height - boxes[:, 3]
                boxes_flipped[:, 3] = height - boxes[:, 1]
                target["boxes"] = boxes_flipped

            if target["masks"].numel() > 0:
                target["masks"] = target["masks"].flip(-2)

        return image, target


def get_transforms(train: bool):
    """Return transforms for training or validation."""
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
    return Compose(transforms)


class Compose:
    """Compose transforms that operate on (image, target) pairs."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Randomly flip image and target horizontally."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1).item() < self.prob:
            image = image.flip(-1)
            width = image.shape[-1]

            if target["boxes"].numel() > 0:
                boxes = target["boxes"]
                boxes_flipped = boxes.clone()
                boxes_flipped[:, 0] = width - boxes[:, 2]
                boxes_flipped[:, 2] = width - boxes[:, 0]
                target["boxes"] = boxes_flipped

            if target["masks"].numel() > 0:
                target["masks"] = target["masks"].flip(-1)

        return image, target
