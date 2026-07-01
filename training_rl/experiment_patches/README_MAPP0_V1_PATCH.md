# ReMA-RAG MAPPO-v1 conservative patch

This is a small patch for an existing `/hy-tmp/rema_mappo_v0_tiny_20260615`
cloud workspace.

Changes:
- Adds `conservative_v1` reward mode.
- Creates `rema_plantuml_mappo_v1_tiny32.yaml`.
- Uses learning rate `1e-6`, `max_steps=10`, `save_steps=1000`.
- Writes run/eval/pack scripts for `mappo_v1_tiny32`.

Cloud usage:

```bash
cd /hy-tmp/rema_mappo_v1_patch_20260615
python apply_mappo_v1_conservative.py
tmux new-session -d -s rema-mappo-v1 "bash -lc 'source /usr/local/miniconda3/etc/profile.d/conda.sh && conda activate rema-mappo && cd /hy-tmp/rema_mappo_v0_tiny_20260615 && bash cloud_scripts/06_run_mappo_v1_tiny32.sh 2>&1 | tee /hy-tmp/rema_mappo_v1_tiny32.log'"
tail -f /hy-tmp/rema_mappo_v1_tiny32.log
```
