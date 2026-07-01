#!/usr/bin/env bash
set -euo pipefail
ROOT=/hy-tmp/rema_mappo_v0_tiny_20260615
bash "$ROOT/cloud_scripts/02_prepare_full_sft_adapter.sh"
bash "$ROOT/cloud_scripts/03_patch_llamafactory.sh"
bash "$ROOT/cloud_scripts/04_run_mappo_v0_tiny32.sh" 2>&1 | tee /hy-tmp/rema_mappo_v0_tiny32.log
bash "$ROOT/cloud_scripts/05_eval_mappo_v0_tiny32.sh" 2>&1 | tee /hy-tmp/rema_mappo_v0_tiny32_eval.log
tar -czf /hy-tmp/rema_mappo_v0_tiny32_results_$(date +%Y%m%d_%H%M%S).tar.gz \
  -C "$ROOT" LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/mappo_v0_tiny32 \
  -C "$ROOT" outputs/mappo_v0_tiny32 \
  -C /hy-tmp rema_mappo_v0_tiny32.log \
  -C /hy-tmp rema_mappo_v0_tiny32_eval.log
