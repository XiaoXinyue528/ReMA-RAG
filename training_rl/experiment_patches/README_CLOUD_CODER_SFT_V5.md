# ReMA-RAG Coder-SFT-v5 Cloud Package

This package continues from the existing Full-SFT v4 LoRA adapter and trains a
targeted Coder/Final-Coder repair SFT dataset.

It does **not** rerun the old Full-SFT from scratch.

Cloud usage:

```bash
cd /hy-tmp/rema_coder_sft_v5_package_20260616
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
python apply_coder_sft_v5.py

cd /hy-tmp/rema_mappo_v0_tiny_20260615
tmux new-session -d -s rema-coder-sft-v5 "bash -lc 'cd /hy-tmp/rema_mappo_v0_tiny_20260615 && bash cloud_scripts/14_run_coder_sft_v5.sh 2>&1 | tee /hy-tmp/rema_coder_sft_v5.log'"
tail -f /hy-tmp/rema_coder_sft_v5.log

bash cloud_scripts/15_eval_coder_sft_v5.sh 2>&1 | tee /hy-tmp/rema_coder_sft_v5_eval.log
bash cloud_scripts/16_pairwise_coder_sft_v5.sh 2>&1 | tee /hy-tmp/rema_coder_sft_v5_pairwise.log
bash cloud_scripts/17_pack_coder_sft_v5.sh
```
