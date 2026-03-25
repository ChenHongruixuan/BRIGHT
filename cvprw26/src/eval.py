"""Self-contained server-side evaluation script for holdout predictions."""

import argparse
from contextlib import contextmanager
import gzip
import io
import json
import os
import tempfile
import zipfile

try:
    import faster_coco_eval

    faster_coco_eval.init_as_pycocotools()
except ImportError:
    pass

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


CATEGORIES = {
    1: "intact",
    2: "damaged",
    3: "destroyed",
}


def _zip_member_name(path: str) -> str:
    archive_name = os.path.basename(path)
    if archive_name.endswith(".json.zip"):
        return archive_name[:-4]
    if archive_name.endswith(".zip"):
        return archive_name[:-4] + ".json"
    return archive_name + ".json"


def _resolve_zip_member_name(archive: zipfile.ZipFile, path: str) -> str:
    expected_name = _zip_member_name(path)
    members = [info.filename for info in archive.infolist() if not info.is_dir()]
    if expected_name in members:
        return expected_name

    json_members = [name for name in members if name.lower().endswith(".json")]
    if json_members:
        return json_members[0]
    if len(members) == 1:
        return members[0]
    raise FileNotFoundError(f"No JSON payload found in ZIP archive: {path}")


@contextmanager
def _open_json_file(path: str, mode: str):
    """Open plain JSON, gzip JSON, or zip-packaged JSON with UTF-8 encoding."""
    if path.endswith(".gz"):
        with gzip.open(path, mode, encoding="utf-8") as f:
            yield f
        return
    if path.endswith(".zip"):
        if "b" in mode:
            raise ValueError("Binary ZIP access is not supported for JSON helpers.")

        zip_mode = "r" if "r" in mode else "w"
        member_mode = "r" if "r" in mode else "w"
        with zipfile.ZipFile(path, zip_mode, compression=zipfile.ZIP_DEFLATED) as archive:
            member_name = _resolve_zip_member_name(archive, path) if "r" in mode else _zip_member_name(path)
            with archive.open(member_name, member_mode) as raw_file:
                with io.TextIOWrapper(raw_file, encoding="utf-8") as text_file:
                    yield text_file
        return
    with open(path, mode, encoding="utf-8") as f:
        yield f


def _load_json_file(path: str):
    with _open_json_file(path, "rt") as f:
        return json.load(f)


def _default_metrics_path(prediction_json: str) -> str:
    if prediction_json.endswith(".json.zip"):
        return prediction_json[:-9] + "_metrics.json"
    if prediction_json.endswith(".json.gz"):
        return prediction_json[:-8] + "_metrics.json"
    if prediction_json.endswith(".zip"):
        return prediction_json[:-4] + "_metrics.json"
    if prediction_json.endswith(".gz"):
        return prediction_json[:-3] + "_metrics.json"
    if prediction_json.endswith(".json"):
        return prediction_json[:-5] + "_metrics.json"
    return prediction_json + ".metrics.json"


def _load_coco_gt(gt_path: str) -> tuple[COCO, str | None]:
    """Load GT into a COCO object, expanding compressed input into a temp file if needed."""
    if not gt_path.endswith((".gz", ".zip")):
        return COCO(gt_path), None

    gt_dict = _load_json_file(gt_path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(gt_dict, f)
        temp_path = f.name
    return COCO(temp_path), temp_path


def _build_zero_report(coco_gt: COCO) -> dict[str, float]:
    """Return the compact competition report for the no-prediction case."""
    report = {
        "mAP": 0.0,
        "AP50": 0.0,
        "AP75": 0.0,
    }
    for cat_id in coco_gt.getCatIds():
        report[coco_gt.cats[cat_id]["name"]] = 0.0
    return report


def _build_server_report(coco_eval: COCOeval) -> dict[str, float]:
    """Keep only the competition-facing segmentation metrics."""
    stats = coco_eval.stats
    report = {
        "mAP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
    }

    precisions = coco_eval.eval.get("precision")
    if precisions is None:
        for cat_id in coco_eval.params.catIds:
            report[coco_eval.cocoGt.cats[cat_id]["name"]] = 0.0
        return report

    for cat_idx, cat_id in enumerate(coco_eval.params.catIds):
        precision = precisions[:, :, cat_idx, 0, -1]
        precision = precision[precision > -1]
        category_name = coco_eval.cocoGt.cats[cat_id]["name"]
        report[category_name] = float(precision.mean()) if precision.size else 0.0

    return report


def _evaluate_segmentation(coco_gt: COCO, prediction_list: list[dict]) -> dict[str, float]:
    """Run COCO segmentation evaluation and return the compact report."""
    if not prediction_list:
        return _build_zero_report(coco_gt)

    coco_dt = coco_gt.loadRes(prediction_list)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.maxDets = [1, 100, 1500]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return _build_server_report(coco_eval)


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction JSON against server-side ground truth")
    parser.add_argument("--gt", required=True, help="Server-side ground-truth COCO annotation JSON")
    parser.add_argument("--predictions", required=True, help="Prediction JSON/ZIP to evaluate")
    parser.add_argument("--metrics-output", default=None, help="Optional path to save the compact metrics JSON")
    args = parser.parse_args()

    if not os.path.isfile(args.gt):
        raise FileNotFoundError(f"Ground-truth annotation file not found: {args.gt}")
    if not os.path.isfile(args.predictions):
        raise FileNotFoundError(f"Prediction JSON not found: {args.predictions}")

    gt_dict = _load_json_file(args.gt)
    if not gt_dict.get("annotations"):
        raise ValueError("Ground-truth file contains no annotations.")

    prediction_list = _load_json_file(args.predictions)
    if not isinstance(prediction_list, list):
        raise ValueError(f"Prediction JSON must contain a list of COCO detections: {args.predictions}")

    print(f"Ground-truth annotations: {args.gt}")
    print(f"Prediction JSON: {args.predictions}")

    coco_gt, temp_gt_path = _load_coco_gt(args.gt)
    try:
        report = _evaluate_segmentation(coco_gt, prediction_list)
    finally:
        if temp_gt_path and os.path.exists(temp_gt_path):
            os.remove(temp_gt_path)

    print("Evaluation results:")
    print(f"  mAP: {report['mAP']:.4f}")
    print(f"  AP50: {report['AP50']:.4f}")
    print(f"  AP75: {report['AP75']:.4f}")
    for category_name in CATEGORIES.values():
        print(f"  {category_name}: {report.get(category_name, 0.0):.4f}")

    metrics_output = args.metrics_output or _default_metrics_path(args.predictions)
    metrics_dir = os.path.dirname(metrics_output)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved metrics to {metrics_output}")


if __name__ == "__main__":
    main()
