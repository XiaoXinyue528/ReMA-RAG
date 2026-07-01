# ReMA-RAG Agent-Aware SFT v0 Package

This package trains an **agent-aware/component-level credit assignment v0** dataset.
It is intentionally not named full agent-level MAPPO because the current trajectory
does not contain independent PPO action logs for all agents.

## What this package does

- Starts from the existing `Coder-SFT-v5` LoRA adapter.
- Builds a tiny64 multi-task SFT dataset from 1127 MA-RAG-UML trajectories.
- Uses component metrics to select and oversample role-specific examples:
  - planner: class/entity coverage
  - step_definer: relation pair/label/multiplicity
  - extractor_coder: attributes/methods/local snippets
  - final_coder: final gold PlantUML structure
- Evaluates on the fixed PlantUCD test142 split.
- Compares against Coder-SFT-v5, MAPPO-v3 tiny64, and Full-SFT-v4 when those files are present.

## Local build summary

```json
{
  "trajectories": "D:\\Temp\\rema_rag_fusion\\outputs\\trajectories\\plantuml_1127_promptv2_trajectories.jsonl",
  "eval": "D:\\Temp\\rema_rag_fusion\\outputs\\eval\\plantuml_1127_promptv2_normalized_eval.jsonl",
  "test142": "D:\\Temp\\marag_local\\marag_trajectory_cloud_20260610\\data_splits\\plantucd_test_142.json",
  "batch_infer": "D:\\Temp\\rema_full_sft_4090_20260614\\evaluation\\batch_infer.py",
  "eval_normalized": "D:\\Temp\\rema_full_sft_4090_20260614\\evaluation\\evaluate_predictions_normalized.py",
  "selected_ids": 64,
  "sft_examples_total": 364,
  "train_examples": 326,
  "eval_examples": 38,
  "role_counts": {
    "extractor_coder": 118,
    "final_coder": 64,
    "planner": 64,
    "step_definer": 118
  },
  "selected_id_preview": [
    "plantucd_test_10",
    "plantucd_test_17",
    "plantucd_test_21",
    "plantucd_test_23",
    "plantucd_test_43",
    "plantucd_test_100",
    "plantucd_test_101",
    "plantucd_test_120",
    "plantucd_test_123",
    "plantucd_test_132",
    "plantucd_test_136",
    "plantucd_test_138",
    "plantucd_test_151",
    "plantucd_test_153",
    "plantucd_test_165",
    "plantucd_test_167",
    "plantucd_test_177",
    "plantucd_test_180",
    "plantucd_test_182",
    "plantucd_test_183"
  ],
  "selection_policy": {
    "limit": 64,
    "relation_first": "relation-capable samples from total>=0.45",
    "attribute_next": "attribute/method capable samples from total>=0.45",
    "overall_fill": "highest total reward samples"
  }
}
```

## Cloud usage

Upload and unzip this package on the same cloud machine that has:

- `/hy-tmp/rema_mappo_v0_tiny_20260615`
- `/hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/coder_sft_v5`
- `/hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct`

Then run:

```bash
cd /hy-tmp/rema_agent_aware_sft_v0_package_20260628
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
python apply_agent_aware_sft_v0.py

cd /hy-tmp/rema_mappo_v0_tiny_20260615
tmux new-session -d -s rema-agent-aware "bash -lc 'cd /hy-tmp/rema_mappo_v0_tiny_20260615 && bash cloud_scripts/31_run_agent_aware_sft_v0.sh 2>&1 | tee /hy-tmp/rema_agent_aware_sft_v0.log'"
tail -f /hy-tmp/rema_agent_aware_sft_v0.log

bash cloud_scripts/32_eval_agent_aware_sft_v0.sh 2>&1 | tee /hy-tmp/rema_agent_aware_sft_v0_eval.log
bash cloud_scripts/33_pairwise_agent_aware_sft_v0.sh 2>&1 | tee /hy-tmp/rema_agent_aware_sft_v0_pairwise.log
bash cloud_scripts/34_pack_agent_aware_sft_v0.sh
```

## How to report this experiment

Use the label:

> Agent-aware SFT v0 / component-level credit assignment prototype

Do not claim it is complete four-agent MAPPO. It uses intermediate agent outputs
for role-aware SFT and evaluates whether component credit helps beyond Coder-SFT-v5.
