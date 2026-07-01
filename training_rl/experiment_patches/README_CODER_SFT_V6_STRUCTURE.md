# Coder-SFT-v6 Structure Package

This package builds a conservative structure-preserving Coder-SFT dataset on the cloud.
It starts from `full_sft_v4`, not from `coder_sft_v5`, because v5 improved details but
reduced class/relation-pair metrics.

## Policy

- Keep high-confidence step-coder examples as backbone anchors.
- Keep focused repair examples unless v5 degraded class_f1 or relation_pair_f1.
- Add Full-SFT-v4 prediction anchors for v5-degraded samples when the cloud pairwise CSV exists.
- Add safe full-repair examples only when v5 improved details without hurting class/relation-pair.

## Cloud Usage

```bash
cd /hy-tmp
unzip -o /hy-tmp/rema_coder_sft_v6_structure_package_20260616.zip -d /hy-tmp
cd /hy-tmp/rema_coder_sft_v6_structure_package_20260616
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
python apply_coder_sft_v6_structure.py
cd /hy-tmp/rema_mappo_v0_tiny_20260615
bash cloud_scripts/18_run_coder_sft_v6_structure.sh 2>&1 | tee /hy-tmp/rema_coder_sft_v6_structure.log
bash cloud_scripts/19_eval_coder_sft_v6_structure.sh 2>&1 | tee /hy-tmp/rema_coder_sft_v6_structure_eval.log
bash cloud_scripts/20_pairwise_coder_sft_v6_structure.sh 2>&1 | tee /hy-tmp/rema_coder_sft_v6_structure_pairwise.log
bash cloud_scripts/21_pack_coder_sft_v6_structure.sh
```
