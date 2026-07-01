#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
mkdir -p /hy-tmp/cache/modelscope /hy-tmp/cache/huggingface
export MODELSCOPE_CACHE=/hy-tmp/cache/modelscope
export HF_HOME=/hy-tmp/cache/huggingface
export USE_MODELSCOPE_HUB=1
python - <<'PY'
from modelscope import snapshot_download
path = snapshot_download("LLM-Research/Meta-Llama-3-8B-Instruct", cache_dir="/hy-tmp/cache/modelscope")
print("MODEL_PATH=", path)
PY
