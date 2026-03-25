"""Competition inference entry-point: generate a submission JSON from a checkpoint."""

import argparse
import os

import torch

from src.infer_utils import (
    build_inference_dataset,
    build_inference_loader,
    build_inference_model,
    load_checkpoint,
    run_inference,
    save_coco_results,
)
from src.utils import load_config, set_seed


def _normalize_output_stem(output_path: str) -> str:
    """Return the output stem without JSON or compression suffixes."""
    if output_path.endswith(".json.zip"):
        return output_path[:-9]
    if output_path.endswith(".zip"):
        return output_path[:-4]
    if output_path.endswith(".json.gz"):
        return output_path[:-8]
    if output_path.endswith(".gz"):
        return output_path[:-3]
    if output_path.endswith(".json"):
        return output_path[:-5]
    return output_path


def _resolve_output_path(output_path: str, use_zip: bool) -> str:
    output_stem = _normalize_output_stem(output_path)
    return output_stem + ".zip" if use_zip else output_stem + ".json"


def main():
    parser = argparse.ArgumentParser(description="Run Mask R-CNN inference and export a submission JSON/ZIP")
    parser.add_argument("--config", default="config/disaster.yaml", help="Path to YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path from config")
    parser.add_argument(
        "--ann-file",
        default=None,
        help="COCO image manifest or annotation JSON used for inference",
    )
    parser.add_argument("--output", default=None, help="Override output JSON/ZIP path from config")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument(
        "--zip",
        "--gzip",
        dest="zip_output",
        action="store_true",
        help="Write a zip-compressed submission file (`.zip`); `--gzip` is kept as a legacy alias",
    )
    parser.add_argument(
        "--no-zip",
        "--no-gzip",
        dest="zip_output",
        action="store_false",
        help="Disable archive compression and write a plain `.json` file",
    )
    parser.add_argument("--visualize", dest="visualize", action="store_true",
                        help="Save instance-segmentation visualization images")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false",
                        help="Disable visualization even if config enables it")
    parser.add_argument("--vis-score-thr", type=float, default=None,
                        help="Score threshold for visualization only")
    parser.add_argument("--vis-max", type=int, default=None,
                        help="Max images to visualize (0 = all)")
    parser.set_defaults(visualize=None, zip_output=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    infer_cfg = cfg.get("infer", {})
    legacy_test_cfg = cfg.get("test", {})

    checkpoint_path = args.checkpoint or infer_cfg.get("checkpoint") or legacy_test_cfg.get("checkpoint")
    ann_file = (
        args.ann_file
        or infer_cfg.get("ann_file")
        or data_cfg.get("holdout_manifest")
    )
    output_path = args.output or infer_cfg.get("output_json") or legacy_test_cfg.get("output_json")
    if checkpoint_path is None:
        raise ValueError("No checkpoint path provided. Set infer.checkpoint or pass --checkpoint.")
    if ann_file is None:
        raise ValueError("No inference annotation file provided. Set infer.ann_file or pass --ann-file.")
    if output_path is None:
        raise ValueError("No output path provided. Set infer.output_json or pass --output.")
    output_path = _resolve_output_path(output_path, use_zip=args.zip_output)

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else infer_cfg.get("num_workers", legacy_test_cfg.get("num_workers", train_cfg.get("num_workers", 2)))
    )
    visualize = infer_cfg.get("visualize", False) if args.visualize is None else args.visualize
    vis_score_thr = args.vis_score_thr if args.vis_score_thr is not None else infer_cfg.get("vis_score_thr")
    vis_max = args.vis_max if args.vis_max is not None else infer_cfg.get("vis_max", 0)

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    set_seed(train_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Inference manifest: {ann_file}")

    dataset = build_inference_dataset(data_cfg, ann_file)
    data_loader = build_inference_loader(dataset, num_workers=num_workers)

    model = build_inference_model(model_cfg, data_cfg)
    load_checkpoint(model, checkpoint_path, device)
    model.to(device)

    coco_results, summary = run_inference(
        model,
        data_loader,
        device,
        output_dir=output_dir,
        visualize=visualize,
        vis_score_thr=vis_score_thr,
        vis_max=vis_max,
    )

    print(
        f"Inference complete: {summary['num_images']} images in {summary['elapsed_sec']:.1f}s "
        f"({summary['seconds_per_image']:.3f} s/img), "
        f"{summary['num_detections']} total detections"
    )

    save_coco_results(coco_results, output_path)
    print(f"Saved predictions to {output_path}")

    if summary["vis_dir"] is not None:
        print(f"Saved {summary['vis_count']} visualizations to {summary['vis_dir']}/")


if __name__ == "__main__":
    main()
