# ReMA-RAG MAPPO-v0 tiny package

This package is a minimal PPO/value-head bridge for PlantUML reward training.

Scope:
- Warm-start actor from the Full-SFT v4 LoRA adapter.
- Use 32 RL-candidate PlantUML samples by default.
- Generate PlantUML directly from requirement prompts.
- Score generated PlantUML with the local normalized rule reward.
- Run PPO updates through LLaMA-Factory/TRL and save a new adapter/value head.

This is intentionally not the final full multi-agent MAPPO implementation.
It is the first executable RL bridge. If it is stable, expand from tiny32 to
tiny64 and then to the 298 RL candidates.

Required companion file on cloud:
- `/hy-tmp/rema_full_sft_v4_results_20260614_185318.tar.gz`

Recommended cloud flow:

```bash
cd /hy-tmp/rema_mappo_v0_tiny_20260615
bash cloud_scripts/00_setup_env.sh
bash cloud_scripts/01_download_llama3_modelscope.sh
tmux new -s rema-mappo
bash cloud_scripts/run_all_mappo_v0_tiny.sh
```

Monitor:

```bash
tail -f /hy-tmp/rema_mappo_v0_tiny32.log
nvidia-smi
```

Key files:
- `LLaMA-Factory/data/rema_plantuml_mappo_tiny32.json`
- `LLaMA-Factory/data/rema_plantuml_mappo_reward_map.jsonl`
- `patches/apply_plantuml_ppo_v0.py`
- `LLaMA-Factory/examples/train_lora/rema_plantuml_mappo_v0_tiny32.yaml`

Local source summary:
- RL candidates: 298
- tiny32: 32
- tiny64: 64
- full RL candidate pool: 298
