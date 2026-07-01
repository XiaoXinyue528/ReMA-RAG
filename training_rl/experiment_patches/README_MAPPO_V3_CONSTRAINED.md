# ReMA-RAG MAPPO-v3 constrained patch

This patch targets an existing cloud workspace:
`/hy-tmp/rema_mappo_v0_tiny_20260615`.

Compared with v1/v2:
- Starts PPO from `coder_sft_v5`, not from `full_sft_v4`.
- Builds the RL baseline map from `coder_sft_v5`.
- Uses a backbone-constrained reward:
  class and relation-pair are treated as hard structure constraints;
  attribute, method, relation-label, and multiplicity receive bonus only
  when the class/relation-pair backbone does not regress.

Cloud order:

```bash
cd /hy-tmp/rema_mappo_v3_constrained_patch_20260616
python apply_mappo_v3_constrained.py

cd /hy-tmp/rema_mappo_v0_tiny_20260615
bash cloud_scripts/22_build_mappo_v3_coder_sft_v5_baseline.sh 2>&1 | tee /hy-tmp/rema_mappo_v3_coder_sft_v5_baseline.log

tmux new-session -d -s rema-mappo-v3 "bash -lc 'cd /hy-tmp/rema_mappo_v0_tiny_20260615 && bash cloud_scripts/23_run_mappo_v3_constrained_tiny64.sh 2>&1 | tee /hy-tmp/rema_mappo_v3_constrained_tiny64.log'"
tail -f /hy-tmp/rema_mappo_v3_constrained_tiny64.log

bash cloud_scripts/24_eval_mappo_v3_constrained_tiny64.sh 2>&1 | tee /hy-tmp/rema_mappo_v3_constrained_tiny64_eval.log
bash cloud_scripts/25_pairwise_mappo_v3_constrained_tiny64.sh 2>&1 | tee /hy-tmp/rema_mappo_v3_constrained_tiny64_pairwise.log
bash cloud_scripts/26_pack_mappo_v3_constrained_tiny64.sh
```
