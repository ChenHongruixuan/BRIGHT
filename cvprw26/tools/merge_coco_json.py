"""Prepare unified COCO files for public train/val labels and unlabeled holdout inference.

Workflow:
    1. Read sample_id lists from splits/{train,val,holdout}_set.txt
    2. For train/val, load the corresponding per-image JSON from
       target_instance_level/{sample_id}_instance_damage.json
    3. For holdout, default to an images-only COCO manifest with empty annotations
       so participants can run inference without private holdout labels
    4. Optionally check that matching post-event SAR and pre-event RGB tifs exist
    5. Reassign globally unique image_id and annotation_id
    6. Set file_name = {sample_id}_post_disaster.tif
    7. Preserve images with empty annotations (negative samples)
    8. Write unified COCO JSONs to annotations/{train,val,holdout}.json

Usage:
    # Competition-style prep: train/val with labels, holdout as images-only manifest
    python tools/merge_coco_json.py

    # Server/internal prep: also merge private holdout labels if available
    python tools/merge_coco_json.py --holdout-mode annotations

    # Local subset debugging (only keep samples with a matching pre/post image pair)
    python tools/merge_coco_json.py --check-images
"""

import argparse
import json
import os

import rasterio


CATEGORIES = [
    {"id": 1, "name": "intact", "supercategory": "building_damage"},
    {"id": 2, "name": "damaged", "supercategory": "building_damage"},
    {"id": 3, "name": "destroyed", "supercategory": "building_damage"},
]

SPLITS = ["train", "val", "holdout"]


def read_split_ids(splits_dir: str, split_name: str) -> list[str]:
    """Read sample IDs from a split file, one ID per line."""
    path = os.path.join(splits_dir, f"{split_name}_set.txt")
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def merge_labeled_split(
    sample_ids: list[str],
    json_dir: str,
    image_dir: str,
    pre_event_dir: str,
    check_images: bool,
    global_image_id: int,
    global_ann_id: int,
) -> tuple[dict, int, int, dict]:
    """Merge per-image JSONs for a single split into a unified COCO dict.

    Returns:
        (coco_dict, next_global_image_id, next_global_ann_id, stats_dict)
    """
    images = []
    annotations = []
    stats = {
        "total": len(sample_ids),
        "processed": 0,
        "skipped_no_json": 0,
        "skipped_no_post_image": 0,
        "skipped_no_pre_image": 0,
    }

    for sample_id in sample_ids:
        json_name = f"{sample_id}_instance_damage.json"
        json_path = os.path.join(json_dir, json_name)

        # Skip if the expected JSON does not exist
        if not os.path.isfile(json_path):
            stats["skipped_no_json"] += 1
            continue

        # Optionally check that the expected pre/post image pair exists
        if check_images:
            post_name = f"{sample_id}_post_disaster.tif"
            post_path = os.path.join(image_dir, post_name)
            pre_name = f"{sample_id}_pre_disaster.tif"
            pre_path = os.path.join(pre_event_dir, pre_name)
            missing_post = not os.path.isfile(post_path)
            missing_pre = not os.path.isfile(pre_path)
            if missing_post or missing_pre:
                if missing_post:
                    stats["skipped_no_post_image"] += 1
                if missing_pre:
                    stats["skipped_no_pre_image"] += 1
                continue

        # Load per-image JSON (handle UTF-8 BOM)
        with open(json_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        # Each per-image JSON has exactly one image entry
        src_image = data["images"][0]
        new_image = {
            "id": global_image_id,
            "file_name": f"{sample_id}_post_disaster.tif",
            "width": src_image["width"],
            "height": src_image["height"],
        }
        images.append(new_image)

        # Map original image_id to new global image_id for annotations
        old_image_id = src_image["id"]
        for ann in data.get("annotations", []):
            if ann["image_id"] != old_image_id:
                continue
            new_ann = {
                "id": global_ann_id,
                "image_id": global_image_id,
                "category_id": ann["category_id"],
                "segmentation": ann["segmentation"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann.get("iscrowd", 0),
            }
            annotations.append(new_ann)
            global_ann_id += 1

        global_image_id += 1
        stats["processed"] += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    return coco_dict, global_image_id, global_ann_id, stats


def build_holdout_manifest(
    sample_ids: list[str],
    image_dir: str,
    pre_event_dir: str,
    global_image_id: int,
) -> tuple[dict, int, dict]:
    """Build an images-only COCO manifest for unlabeled holdout inference."""
    images = []
    stats = {
        "total": len(sample_ids),
        "processed": 0,
        "skipped_no_json": 0,
        "skipped_no_post_image": 0,
        "skipped_no_pre_image": 0,
    }

    for sample_id in sample_ids:
        post_name = f"{sample_id}_post_disaster.tif"
        post_path = os.path.join(image_dir, post_name)
        pre_name = f"{sample_id}_pre_disaster.tif"
        pre_path = os.path.join(pre_event_dir, pre_name)

        if not os.path.isfile(post_path):
            stats["skipped_no_post_image"] += 1
            continue
        if not os.path.isfile(pre_path):
            stats["skipped_no_pre_image"] += 1
            continue

        with rasterio.open(post_path) as src:
            width = src.width
            height = src.height

        images.append(
            {
                "id": global_image_id,
                "file_name": post_name,
                "width": width,
                "height": height,
            }
        )
        global_image_id += 1
        stats["processed"] += 1

    coco_dict = {
        "images": images,
        "annotations": [],
        "categories": CATEGORIES,
    }
    return coco_dict, global_image_id, stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare labeled train/val COCO files and a public holdout inference manifest."
    )
    parser.add_argument(
        "--json-dir",
        default="/data/ggeoinfo/datasets/BRIGHT/target_instance_level",
        help="Directory containing per-image *_instance_damage.json files",
    )
    parser.add_argument(
        "--splits-dir",
        default="/home/chenhrx/project/cvprw26/data/splits",
        help="Directory containing {train,val,holdout}_set.txt split files",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/chenhrx/project/cvprw26/data/instance_annotations",
        help="Output directory for merged COCO JSON files",
    )
    parser.add_argument(
        "--image-dir",
        default="/data/ggeoinfo/datasets/BRIGHT/post-event",
        help="Post-event SAR image directory, used with --check-images",
    )
    parser.add_argument(
        "--pre-event-dir",
        default="/data/ggeoinfo/datasets/BRIGHT/pre-event",
        help="Pre-event RGB image directory, used with --check-images",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Skip samples without the expected pre/post image pair (for local subset debugging)",
    )
    parser.add_argument(
        "--holdout-mode",
        choices=("manifest", "annotations"),
        default="manifest",
        help=(
            "How to prepare the holdout split: "
            "'manifest' creates an images-only COCO file for public inference; "
            "'annotations' merges private holdout labels for internal/server evaluation"
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    global_image_id = 1
    global_ann_id = 1

    for split in SPLITS:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")

        sample_ids = read_split_ids(args.splits_dir, split)
        print(f"  Sample IDs in split file: {len(sample_ids)}")

        if split == "holdout" and args.holdout_mode == "manifest":
            coco_dict, global_image_id, stats = build_holdout_manifest(
                sample_ids=sample_ids,
                image_dir=args.image_dir,
                pre_event_dir=args.pre_event_dir,
                global_image_id=global_image_id,
            )
        else:
            coco_dict, global_image_id, global_ann_id, stats = merge_labeled_split(
                sample_ids=sample_ids,
                json_dir=args.json_dir,
                image_dir=args.image_dir,
                pre_event_dir=args.pre_event_dir,
                check_images=args.check_images,
                global_image_id=global_image_id,
                global_ann_id=global_ann_id,
            )

        output_path = os.path.join(args.output_dir, f"{split}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f)

        n_images = len(coco_dict["images"])
        n_anns = len(coco_dict["annotations"])
        image_ids_with_anns = {a["image_id"] for a in coco_dict["annotations"]}
        n_empty = sum(1 for img in coco_dict["images"] if img["id"] not in image_ids_with_anns)

        print(f"  Processed:       {stats['processed']}")
        if stats["skipped_no_json"] > 0:
            print(f"  Skipped (no JSON): {stats['skipped_no_json']}")
        if stats["skipped_no_post_image"] > 0:
            print(f"  Skipped (no post-event tif): {stats['skipped_no_post_image']}")
        if stats["skipped_no_pre_image"] > 0:
            print(f"  Skipped (no pre-event tif):  {stats['skipped_no_pre_image']}")
        print(f"  Images:          {n_images}")
        print(f"  Annotations:     {n_anns}")
        print(f"  Empty images:    {n_empty}")
        if split == "holdout" and args.holdout_mode == "manifest":
            print("  Note: holdout.json was written as an images-only COCO manifest.")
        print(f"  Saved to: {output_path}")

    print(f"\nDone. Total images: {global_image_id - 1}, total annotations: {global_ann_id - 1}")


if __name__ == "__main__":
    main()
