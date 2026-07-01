#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
cd /hy-tmp/rema_mappo_v0_tiny_20260615
python patches/apply_plantuml_ppo_v0.py
