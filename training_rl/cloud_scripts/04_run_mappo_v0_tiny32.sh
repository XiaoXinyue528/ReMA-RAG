#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
cd /hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory
export REMA_PLANTUML_PPO=1
export REMA_FUSION_DIR=/hy-tmp/rema_mappo_v0_tiny_20260615/fusion
export REMA_PLANTUML_REWARD_MAP=/hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory/data/rema_plantuml_mappo_reward_map.jsonl
export REMA_PLANTUML_REWARD_MODE=relation_focus
export REMA_PLANTUML_MAX_NEW_TOKENS=512
llamafactory-cli train examples/train_lora/rema_plantuml_mappo_v0_tiny32.yaml
