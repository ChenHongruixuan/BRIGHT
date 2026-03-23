#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gtn-container_g1.24h
#$ -ac d=aip-cuda-12.0.1-blender-2,d_shm=121G
#$ -N MaskRCNN_bright_newsplit

set -euo pipefail

. ~/net.sh

PYTHON_BIN="${PYTHON_BIN:-/home/chenhrx/anaconda3/envs/bright_cvprw2026/bin/python}"

"${PYTHON_BIN}" -m src.train --config config/disaster.yaml "$@"
