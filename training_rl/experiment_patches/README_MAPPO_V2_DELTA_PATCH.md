# ReMA-RAG MAPPO-v2 content-guarded delta patch

This patch targets an existing cloud workspace:
`/hy-tmp/rema_mappo_v0_tiny_20260615`.

It keeps Full-SFT v4 fixed and changes only the PPO training reward.

Reward-v2 principles:
- Class is the highest-priority structural skeleton.
- Attribute is protected as core class content.
- Relation and multiplicity can be rewarded only when class/attribute do not regress.
- PPO reward is baseline-aware: current output is compared against Full-SFT on the same RL sample.
- Final evaluation metrics are not changed.

Cloud order:

```bash
cd /hy-tmp/rema_mappo_v2_delta_patch_20260615
python apply_mappo_v2_content_guarded_delta.py

cd /hy-tmp/rema_mappo_v0_tiny_20260615
bash cloud_scripts/09_build_mappo_v2_tiny64_baseline.sh 2>&1 | tee /hy-tmp/rema_mappo_v2_tiny64_baseline.log

tmux new-session -d -s rema-mappo-v2-64 "bash -lc 'cd /hy-tmp/rema_mappo_v0_tiny_20260615 && bash cloud_scripts/10_run_mappo_v2_tiny64.sh 2>&1 | tee /hy-tmp/rema_mappo_v2_tiny64.log'"
tail -f /hy-tmp/rema_mappo_v2_tiny64.log

bash cloud_scripts/11_eval_mappo_v2_tiny64.sh 2>&1 | tee /hy-tmp/rema_mappo_v2_tiny64_eval.log
bash cloud_scripts/12_pairwise_mappo_v2_tiny64.sh 2>&1 | tee /hy-tmp/rema_mappo_v2_tiny64_pairwise.log
bash cloud_scripts/13_pack_mappo_v2_tiny64.sh
```
