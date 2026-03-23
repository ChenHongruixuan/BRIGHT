"""Training and evaluation engine.

Loosely follows the conventions of ``torchvision/references/detection/engine.py``
but is self-contained so the project has no hidden dependencies on that reference
code.
"""

import json
import os
import tempfile
import warnings

import torch
import numpy as np

# Use faster-coco-eval (C++ backend, ~30x faster) if available, else fall back to pycocotools
try:
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
except ImportError:
    pass

from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util

from src.utils import MetricLogger, SmoothedValue


EVAL_IOU_TYPES = ("bbox", "segm")
EVAL_MAX_DETS = (1, 100, 1500)


def _sanitize_category_name(name: str) -> str:
    """Convert a category label into a metric-friendly suffix."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def build_empty_coco_metrics(coco_gt, iou_types=EVAL_IOU_TYPES, max_dets=EVAL_MAX_DETS) -> dict[str, float]:
    """Return a zero-filled metric dict with the same keys as regular COCO evaluation."""
    metrics = {}
    cat_ids = list(coco_gt.getCatIds())

    for iou_type in iou_types:
        metrics.update(
            {
                f"{iou_type}_AP": 0.0,
                f"{iou_type}_AP50": 0.0,
                f"{iou_type}_AP75": 0.0,
                f"{iou_type}_APs": 0.0,
                f"{iou_type}_APm": 0.0,
                f"{iou_type}_APl": 0.0,
                f"{iou_type}_AR{max_dets[0]}": 0.0,
                f"{iou_type}_AR{max_dets[1]}": 0.0,
                f"{iou_type}_AR{max_dets[2]}": 0.0,
                f"{iou_type}_ARs": 0.0,
                f"{iou_type}_ARm": 0.0,
                f"{iou_type}_ARl": 0.0,
            }
        )

        for cat_id in cat_ids:
            cat_name = coco_gt.cats[cat_id]["name"]
            metrics[f"{iou_type}_AP_{_sanitize_category_name(cat_name)}"] = 0.0

    return metrics


def summarize_coco_eval(coco_eval, iou_type: str) -> dict[str, float]:
    """Convert a COCOeval object into a flat metric dict."""
    stats = coco_eval.stats
    max_dets = coco_eval.params.maxDets
    metrics = {
        f"{iou_type}_AP": float(stats[0]),
        f"{iou_type}_AP50": float(stats[1]),
        f"{iou_type}_AP75": float(stats[2]),
        f"{iou_type}_APs": float(stats[3]),
        f"{iou_type}_APm": float(stats[4]),
        f"{iou_type}_APl": float(stats[5]),
        f"{iou_type}_AR{max_dets[0]}": float(stats[6]),
        f"{iou_type}_AR{max_dets[1]}": float(stats[7]),
        f"{iou_type}_AR{max_dets[2]}": float(stats[8]),
        f"{iou_type}_ARs": float(stats[9]),
        f"{iou_type}_ARm": float(stats[10]),
        f"{iou_type}_ARl": float(stats[11]),
    }

    precisions = coco_eval.eval.get("precision")
    if precisions is None:
        return metrics

    for cat_idx, cat_id in enumerate(coco_eval.params.catIds):
        precision = precisions[:, :, cat_idx, 0, -1]
        precision = precision[precision > -1]
        cat_name = coco_eval.cocoGt.cats[cat_id]["name"]
        metrics[f"{iou_type}_AP_{_sanitize_category_name(cat_name)}"] = (
            float(precision.mean()) if precision.size else 0.0
        )

    return metrics


def format_metrics_report(metrics: dict[str, float]) -> list[str]:
    """Format flat metric dict into readable log lines."""
    lines = []

    for iou_type in EVAL_IOU_TYPES:
        prefix = f"{iou_type}_"
        if f"{prefix}AP" not in metrics:
            continue

        lines.append(
            (
                f"{iou_type}: "
                f"AP={metrics[f'{prefix}AP']:.4f}  "
                f"AP50={metrics[f'{prefix}AP50']:.4f}  "
                f"AP75={metrics[f'{prefix}AP75']:.4f}  "
                f"APs={metrics[f'{prefix}APs']:.4f}  "
                f"APm={metrics[f'{prefix}APm']:.4f}  "
                f"APl={metrics[f'{prefix}APl']:.4f}"
            )
        )

        ar_keys = [key for key in metrics if key.startswith(f"{prefix}AR") and key[7:].isdigit()]
        if ar_keys:
            ar_keys = sorted(ar_keys, key=lambda key: int(key[7:]))
            ar_parts = [f"{key[5:]}={metrics[key]:.4f}" for key in ar_keys]
            lines.append(f"{iou_type} recall: " + "  ".join(ar_parts))

        class_keys = [key for key in metrics if key.startswith(f"{prefix}AP_")]
        if class_keys:
            class_parts = []
            for key in sorted(class_keys):
                class_name = key[len(f"{prefix}AP_") :]
                class_parts.append(f"{class_name}={metrics[key]:.4f}")
            lines.append(f"{iou_type} per-class AP: " + "  ".join(class_parts))

    return lines


def evaluate_coco_results(
    coco_gt,
    results_path: str,
    iou_types=EVAL_IOU_TYPES,
    max_dets=EVAL_MAX_DETS,
) -> dict[str, float]:
    """Evaluate a saved COCO prediction JSON and return a flat metric dict."""
    coco_dt = coco_gt.loadRes(results_path)
    metrics = {}

    for iou_type in iou_types:
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.params.maxDets = list(max_dets)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics.update(summarize_coco_eval(coco_eval, iou_type))

    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    print_freq=10,
    scaler=None,
    warmup_iters=0,
    warmup_factor=0.001,
    log_file=None,
):
    """Run one full training epoch.

    Args:
        model: Mask R-CNN model.
        optimizer: Optimiser instance.
        data_loader: Training data loader.
        device: ``torch.device``.
        epoch: Current epoch index (for logging).
        print_freq: How often (in iterations) to print metrics.
        scaler: Optional ``torch.cuda.amp.GradScaler`` for mixed-precision.
        warmup_iters: Number of warmup iterations (only active in epoch 0).
        warmup_factor: Initial lr multiplier for warmup (ramps linearly to 1.0).
        log_file: Optional path to write training logs.

    Returns:
        A :class:`MetricLogger` with recorded losses.
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ", log_file=log_file)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # LinearLR warmup (only epoch 0, capped to fit within the epoch)
    lr_warmup_scheduler = None
    if epoch == 0 and warmup_iters > 0:
        warmup_iters = min(warmup_iters, len(data_loader) - 1)
        lr_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_factor,
            total_iters=warmup_iters,
        )

    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Reduce losses for logging (value on this device is fine for single-GPU)
        loss_value = losses.item()

        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            raise SystemExit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_warmup_scheduler is not None and i < warmup_iters:
            # Suppress false-positive warning: LinearLR.__init__ internally bumps
            # _step_count before any optimizer.step(), but our call order is correct.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*lr_scheduler.step.*optimizer.step.*")
                lr_warmup_scheduler.step()

        metric_logger.update(loss=loss_value, **{k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, data_loader, device, output_dir=None, epoch=None, log_file=None):
    """Run COCO-style evaluation for bbox and segm AP.

    The function expects ``data_loader.dataset`` to expose a ``.coco``
    attribute that is a :class:`pycocotools.coco.COCO` object (this is set
    by :class:`BRIGHTDataset`).

    Args:
        model: Mask R-CNN model (will be set to eval mode).
        data_loader: Validation / test data loader.
        device: ``torch.device``.
        output_dir: If provided, save COCO results JSON to this directory.
        epoch: Current epoch index (used in the saved filename).

    Returns:
        A flat dict with COCO summary metrics for ``bbox`` and ``segm``,
        plus per-category AP entries such as ``segm_AP_intact``.
    """
    import logging
    _logger = logging.getLogger("train")
    if not _logger.handlers:
        _logger.addHandler(logging.StreamHandler())
    if log_file and not any(isinstance(h, logging.FileHandler) for h in _logger.handlers):
        _logger.addHandler(logging.FileHandler(log_file, mode="a"))

    model.eval()
    coco_gt = data_loader.dataset.coco  # pycocotools COCO object

    coco_results = []
    num_images = len(data_loader)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]

            outputs = model(images)

            for target, output in zip(targets, outputs):
                img_id = target["image_id"]
                image_id = img_id.item() if isinstance(img_id, torch.Tensor) else int(img_id)

                boxes = output["boxes"].cpu().numpy()    # [N, 4] xyxy
                scores = output["scores"].cpu().numpy()  # [N]
                labels = output["labels"].cpu().numpy()   # [N]
                masks = output["masks"].cpu().numpy()    # [N, 1, H, W]

                # No score filtering here — COCOeval handles ranking internally.
                for i in range(len(scores)):
                    # Convert bbox from [x1, y1, x2, y2] to COCO [x, y, w, h]
                    x1, y1, x2, y2 = boxes[i].tolist()
                    bbox_coco = [x1, y1, x2 - x1, y2 - y1]

                    # Encode mask to RLE
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

            if (idx + 1) % 50 == 0 or (idx + 1) == num_images:
                _logger.info(f"  Eval: [{idx + 1}/{num_images}] {len(coco_results)} detections")

    # If there are no predictions at all, return zeros
    if len(coco_results) == 0:
        _logger.info("No predictions produced -- returning zero metrics.")
        return build_empty_coco_metrics(coco_gt)

    # Save results JSON — to output_dir if provided, otherwise a temp file
    cleanup_results_path = False
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        tag = f"_epoch{epoch:03d}" if epoch is not None else ""
        results_path = os.path.join(output_dir, f"eval_results{tag}.json")
        with open(results_path, "w") as f:
            json.dump(coco_results, f)
        _logger.info(f"  Saved {len(coco_results)} eval predictions to {results_path}")
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(coco_results, f)
            results_path = f.name
            cleanup_results_path = True

    try:
        return evaluate_coco_results(coco_gt, results_path)
    finally:
        if cleanup_results_path and os.path.exists(results_path):
            os.remove(results_path)
