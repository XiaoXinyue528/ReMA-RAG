#!/usr/bin/env bash
set -euo pipefail
conda create -n rema-mappo python=3.10 -y
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
python -m pip install --upgrade pip
pip install -r /hy-tmp/rema_mappo_v0_tiny_20260615/requirements-mappo-v0.txt
cd /hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory
pip install -e ".[metrics]"
