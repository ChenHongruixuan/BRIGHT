"""Dataset annotation statistics for COCO-format BRIGHT splits.

Usage::

    python -m src.annotation_stats --config config/disaster.yaml --split val
    python -m src.annotation_stats --ann-file data/instance_annotations/val.json
"""

import argparse
import json
from collections import Counter

from src.utils import load_config


def resolve_annotation_file(config_path: str, split: str) -> str:
    """Resolve an annotation JSON path from the YAML config and split name."""
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    key = f"{split}_ann"
    if key not in data_cfg:
        raise KeyError(f"Split '{split}' is not defined in {config_path}")
    return data_cfg[key]


def compute_instance_stats(ann_file: str) -> dict:
    """Compute per-image instance-count statistics from a COCO annotation file."""
    with open(ann_file, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    counts = Counter()
    for ann in annotations:
        counts[ann["image_id"]] += 1

    image_records = []
    for img in images:
        num_instances = counts.get(img["id"], 0)
        image_records.append(
            {
                "image_id": img["id"],
                "file_name": img.get("file_name", ""),
                "num_instances": num_instances,
            }
        )

    max_instances = max((record["num_instances"] for record in image_records), default=0)
    max_images = [record for record in image_records if record["num_instances"] == max_instances]

    return {
        "ann_file": ann_file,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "max_instances_per_image": max_instances,
        "num_images_with_max_instances": len(max_images),
        "max_instance_images": max_images,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute instance-count statistics for a COCO annotation file")
    parser.add_argument("--config", default="config/disaster.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Labeled dataset split to inspect when --ann-file is not provided",
    )
    parser.add_argument("--ann-file", default=None, help="Explicit COCO annotation JSON path")
    args = parser.parse_args()

    ann_file = args.ann_file or resolve_annotation_file(args.config, args.split)
    stats = compute_instance_stats(ann_file)

    print(f"Annotation file: {stats['ann_file']}")
    print(f"Images: {stats['num_images']}")
    print(f"Annotations: {stats['num_annotations']}")
    print(f"Max instances in one image: {stats['max_instances_per_image']}")
    print(f"Images reaching that max: {stats['num_images_with_max_instances']}")
    for record in stats["max_instance_images"]:
        print(
            f"  image_id={record['image_id']}  "
            f"instances={record['num_instances']}  "
            f"file_name={record['file_name']}"
        )


if __name__ == "__main__":
    main()
