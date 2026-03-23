"""Shared helper utilities for inference-style commands."""

from __future__ import annotations

import gzip
import json
import os
import time

import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader

from src.dataset.classes import CATEGORIES
from src.dataset.data import BRIGHTDataset, get_transforms
from src.model.mask_rcnn import build_model
from src.utils import collate_fn


def resolve_data_dirs(data_cfg: dict) -> tuple[str, str]:
    """Return absolute dataset directories for post-event and pre-event inputs."""
    image_dir = os.path.join(data_cfg["root"], data_cfg["images_dir"])
    pre_event_dir = os.path.join(data_cfg["root"], data_cfg["pre_event_dir"])
    return image_dir, pre_event_dir


def build_inference_dataset(data_cfg: dict, ann_file: str) -> BRIGHTDataset:
    """Construct the dataset used by inference-style commands."""
    image_dir, pre_event_dir = resolve_data_dirs(data_cfg)
    return BRIGHTDataset(
        ann_file=ann_file,
        image_dir=image_dir,
        pre_event_dir=pre_event_dir,
        transforms=get_transforms(train=False),
    )


def build_inference_loader(dataset: BRIGHTDataset, num_workers: int) -> DataLoader:
    """Construct a deterministic dataloader for inference."""
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def build_inference_model(model_cfg: dict, data_cfg: dict):
    """Build the Mask R-CNN model for checkpoint-based inference."""
    return build_model(
        num_classes=model_cfg["num_classes"],
        pretrained=False,
        pixel_mean=data_cfg["pixel_mean"],
        pixel_std=data_cfg["pixel_std"],
        box_detections_per_img=model_cfg.get("box_detections_per_img", 1500),
        rpn_pre_nms_top_n_test=model_cfg.get("rpn_pre_nms_top_n_test", 1500),
        rpn_post_nms_top_n_test=model_cfg.get("rpn_post_nms_top_n_test", 1500),
    )


def load_checkpoint(model, checkpoint_path: str, device: torch.device) -> None:
    """Load either a raw model state dict or a full training checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        if "epoch" in checkpoint:
            print(
                f"  Checkpoint from epoch {checkpoint['epoch']}, "
                f"best_ap={checkpoint.get('best_ap', 'N/A')}"
            )
    else:
        model.load_state_dict(checkpoint)


def save_coco_results(coco_results: list[dict], output_path: str) -> None:
    """Write COCO-format predictions to disk."""
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with _open_json_file(output_path, "wt") as f:
        json.dump(coco_results, f)


def load_json_file(path: str):
    """Load either a plain JSON file or a gzip-compressed JSON file."""
    with _open_json_file(path, "rt") as f:
        return json.load(f)


def _open_json_file(path: str, mode: str):
    """Open plain `.json` or gzip-compressed `.json.gz` files."""
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def run_inference(
    model,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: str | None = None,
    visualize: bool = False,
    vis_score_thr: float | None = None,
    vis_max: int = 0,
) -> tuple[list[dict], dict[str, object]]:
    """Run model inference and optionally dump visualization images."""
    dataset = data_loader.dataset
    num_images = len(dataset)
    coco_results = []
    start_time = time.time()

    vis_dir = None
    vis_count = 0
    vis_annotators = None
    if visualize:
        import supervision as sv

        base_dir = output_dir or "."
        vis_dir = os.path.join(base_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        vis_annotators = (
            sv.MaskAnnotator(opacity=0.4),
            sv.BoxAnnotator(thickness=2),
            sv.LabelAnnotator(text_scale=0.5, text_padding=4),
        )

    model.eval()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                img_id = target["image_id"]
                image_id = img_id.item() if isinstance(img_id, torch.Tensor) else int(img_id)

                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy()
                masks = output["masks"].detach().cpu().numpy()

                for i in range(len(scores)):
                    x1, y1, x2, y2 = boxes[i].tolist()
                    bbox_coco = [x1, y1, x2 - x1, y2 - y1]

                    mask_bin = (masks[i, 0] > 0.5).astype(np.uint8)
                    rle = mask_util.encode(np.asfortranarray(mask_bin))
                    rle["counts"] = rle["counts"].decode("utf-8")

                    coco_results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(labels[i]),
                            "bbox": bbox_coco,
                            "score": float(scores[i]),
                            "segmentation": rle,
                        }
                    )

                if vis_dir is not None and (vis_max == 0 or vis_count < vis_max):
                    _visualize_one(
                        dataset=dataset,
                        image_id=image_id,
                        boxes=boxes,
                        scores=scores,
                        labels=labels,
                        masks=masks[:, 0],
                        vis_dir=vis_dir,
                        vis_idx=vis_count,
                        annotators=vis_annotators,
                        vis_score_thr=vis_score_thr,
                    )
                    vis_count += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == num_images:
                print(
                    f"  Processed {idx + 1}/{num_images} images, "
                    f"{len(coco_results)} detections so far"
                )

    elapsed = time.time() - start_time
    summary = {
        "num_images": num_images,
        "num_detections": len(coco_results),
        "elapsed_sec": elapsed,
        "seconds_per_image": elapsed / max(num_images, 1),
        "vis_count": vis_count,
        "vis_dir": vis_dir,
    }
    return coco_results, summary


def _visualize_one(
    dataset: BRIGHTDataset,
    image_id: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    vis_dir: str,
    vis_idx: int,
    annotators,
    vis_score_thr: float | None,
) -> None:
    """Write a single visualization image to disk."""
    import cv2
    import rasterio
    import supervision as sv

    mask_ann, box_ann, label_ann = annotators

    def _read_tif_as_bgr(path: str):
        with rasterio.open(path) as src:
            if src.count >= 3:
                rgb = src.read([1, 2, 3])
                return rgb.transpose(1, 2, 0)[:, :, ::-1].copy()
            gray = src.read(1)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    img_info = dataset.coco.imgs[image_id]
    vis_file_name = img_info["file_name"]

    pre_fname = vis_file_name.replace("_post_disaster.tif", "_pre_disaster.tif")
    pre_path = os.path.join(dataset.pre_event_dir, pre_fname)
    img = None
    if os.path.isfile(pre_path):
        img = _read_tif_as_bgr(pre_path)

    if img is None:
        tif_path = os.path.join(dataset.image_dir, vis_file_name)
        if os.path.isfile(tif_path):
            img = _read_tif_as_bgr(tif_path)

    if img is None:
        return

    if vis_score_thr is not None:
        keep = scores >= vis_score_thr
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        masks = masks[keep]

    if len(boxes) == 0:
        cv2.imwrite(os.path.join(vis_dir, f"{vis_idx:04d}.png"), img)
        return

    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=labels,
        mask=masks > 0.5,
    )

    labels_text = [
        f"{CATEGORIES.get(class_id, '?')} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    annotated = img.copy()
    annotated = mask_ann.annotate(annotated, detections)
    annotated = box_ann.annotate(annotated, detections)
    annotated = label_ann.annotate(annotated, detections, labels=labels_text)

    cv2.imwrite(os.path.join(vis_dir, f"{vis_idx:04d}.png"), annotated)
