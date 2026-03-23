<div align="center">
<h3>BRIGHT Challenge: Advancing All-Weather Building Damage Mapping to Instance-Level</h3>
</div>

Mask R-CNN baseline for BRIGHT building damage instance segmentation.

- Input: pre-event RGB + post-event SAR
- Classes: `intact`, `damaged`, `destroyed`
- Main metric: `segm_AP`

This repo follows the public competition setting:

- participants receive all `train` / `val` / `holdout` images
- participants receive labels only for `train` and `val`
- `holdout` is used only for inference submission
- final `holdout` scoring is done on the server with private GT

## Setup

Install:

```bash
conda create -n bright_cvprw26 python=3.10 -y
conda activate bright_cvprw26
pip install -e .
```

Expected dataset layout:

```text
<BRIGHT_ROOT>/
├── post-event/
├── pre-event/
└── target_instance_level/
```

Repo-side files:

```text
data/
├── splits/
│   ├── train_set.txt
│   ├── val_set.txt
│   └── holdout_set.txt
└── instance_annotations/
    ├── train.json
    ├── val.json
    └── holdout.json
```

`holdout.json` is a public images-only COCO manifest. It does not contain public holdout labels.

Before running, check these fields in [config/disaster.yaml](/home/chenhrx/project/cvprw26/config/disaster.yaml):

- `data.root`
- `data.images_dir`
- `data.pre_event_dir`
- `train.output_dir`
- `infer.checkpoint`
- `infer.output_json`

If your release package already includes merged `train.json`, `val.json`, and `holdout.json`, you can skip annotation preparation. Otherwise run:

```bash
python tools/merge_coco_json.py \
  --json-dir <BRIGHT_ROOT>/target_instance_level \
  --image-dir <BRIGHT_ROOT>/post-event \
  --pre-event-dir <BRIGHT_ROOT>/pre-event \
  --splits-dir data/splits \
  --output-dir data/instance_annotations
```

This creates:

- labeled `train.json`
- labeled `val.json`
- images-only `holdout.json`

If organizers need labeled holdout annotations for server evaluation:

```bash
python tools/merge_coco_json.py \
  --json-dir <BRIGHT_ROOT>/target_instance_level \
  --image-dir <BRIGHT_ROOT>/post-event \
  --pre-event-dir <BRIGHT_ROOT>/pre-event \
  --splits-dir data/splits \
  --output-dir data/instance_annotations \
  --holdout-mode annotations
```

## Workflow

Train:

```bash
python -m src.train --config config/disaster.yaml
```

Or use the cluster wrapper:

```bash
bash train.sh
```

Run inference on `holdout` and generate the submission file:

```bash
python -m src.infer --config config/disaster.yaml
```

Or:

```bash
bash infer.sh
```

Notes:

- gzip is enabled by default, so the output is usually `predictions.json.gz`
- use `--no-gzip` if you need plain JSON
- use `--visualize` to save visualization images

Example:

```bash
python -m src.infer --config config/disaster.yaml --visualize --vis-score-thr 0.5
```

<!-- For local infer+eval on a labeled split, run:

```bash
python -m src.test --config config/disaster.yaml
``` -->

By default this evaluates on `val.json`.

## Evaluation

Participant side:

- upload the `holdout` prediction file to the challenge server
- no public `holdout` metric is available locally

Server side:

Use the self-contained script [src/eval.py](/home/chenhrx/project/cvprw26/src/eval.py):

```bash
python src/eval.py \
  --gt /path/to/private_holdout_gt.json \
  --predictions outputs/infer/predictions.json.gz
```

It reports: `mAP`, `AP50`, `AP75`, `intact`, `damaged`, `destroyed`.

## Outputs

Training:

- `outputs/latest.pth`
- `outputs/best_model.pth`
- `outputs/train.log`
- `outputs/eval/eval_results_epochXXX.json`

Inference:

- `outputs/infer/predictions.json.gz` by default
- `outputs/infer/vis/` if visualization is enabled

Local test:

- `outputs/test/predictions.json`
- `outputs/test/predictions_metrics.json`

Main files:

- [src/train.py](/home/chenhrx/project/cvprw26/src/train.py): training
- [src/infer.py](/home/chenhrx/project/cvprw26/src/infer.py): submission inference
<!-- - [src/test.py](/home/chenhrx/project/cvprw26/src/test.py): local labeled evaluation -->
- [src/eval.py](/home/chenhrx/project/cvprw26/src/eval.py): server-side holdout evaluation
- [tools/merge_coco_json.py](/home/chenhrx/project/cvprw26/tools/merge_coco_json.py): split annotation preparation

## 📜Reference
If this dataset or code contributes to your research, please kindly consider citing our paper and give this repo ⭐️ :)

```bibtex
@Article{Chen2025Bright,
    AUTHOR = {Chen, H. and Song, J. and Dietrich, O. and Broni-Bediako, C. and Xuan, W. and Wang, J. and Shao, X. and Wei, Y. and Xia, J. and Lan, C. and Schindler, K. and Yokoya, N.},
    TITLE = {\textsc{Bright}: a globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response},
    JOURNAL = {Earth System Science Data},
    VOLUME = {17},
    YEAR = {2025},
    NUMBER = {11},
    PAGES = {6217--6253},
    DOI = {10.5194/essd-17-6217-2025}
}
```
