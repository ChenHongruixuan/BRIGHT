#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gs-container_g1_dev
#$ -ac d=aip-cuda-12.0.1-blender-2,d_shm=64G
#$ -N MaskRCNN_bright_infer

set -euo pipefail

. ~/net.sh

PYTHON_BIN="${PYTHON_BIN:-/home/chenhrx/anaconda3/envs/bright_cvprw2026/bin/python}"

"${PYTHON_BIN}" -m src.infer --config config/disaster.yaml "$@"
