from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


ROOT = Path(os.environ.get("REMA_ROOT", "/hy-tmp/rema_mappo_v0_tiny_20260615"))
PACKAGE = Path(__file__).resolve().parent
FACTORY = ROOT / "LLaMA-Factory"
DATA_DIR = FACTORY / "data"
EXAMPLES = FACTORY / "examples" / "train_lora"


def merge_dataset_info() -> None:
    info_path = DATA_DIR / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    patch = json.loads((PACKAGE / "data" / "dataset_info_patch.json").read_text(encoding="utf-8"))
    info.update(patch)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {info_path}")


def copy_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "coder_sft_v5_mixed.json",
        "coder_final_repair_focus.json",
        "coder_step_sft_positive.json",
        "coder_final_repair_all.json",
    ]:
        src = PACKAGE / "data" / name
        dst = DATA_DIR / name
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")


def write_yaml() -> None:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    yaml = """### model
model_name_or_path: /hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
adapter_name_or_path: /hy-tmp/rema_mappo_v0_tiny_20260615/adapters/full_sft_v4
quantization_bit: 4
quantization_method: bitsandbytes
upcast_layernorm: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: rema_plantuml_coder_sft_v5_mixed
eval_dataset: rema_plantuml_full_eval
dataset_dir: data
template: llama3
cutoff_len: 3072
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: saves/llama3-8b/lora/rema_plantuml/coder_sft_v5
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
seed: 42
ddp_timeout: 180000000

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
"""
    path = EXAMPLES / "rema_plantuml_coder_sft_v5_4090.yaml"
    path.write_text(yaml, encoding="utf-8")
    print(f"Wrote {path}")


def write_scripts() -> None:
    scripts = ROOT / "cloud_scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "14_run_coder_sft_v5.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
cd /hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory
llamafactory-cli train examples/train_lora/rema_plantuml_coder_sft_v5_4090.yaml
""",
        encoding="utf-8",
    )
    (scripts / "15_eval_coder_sft_v5.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
ROOT=/hy-tmp/rema_mappo_v0_tiny_20260615
BASE=/hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
ADAPTER="$ROOT/LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/coder_sft_v5"
DATA="$ROOT/LLaMA-Factory/data/plantucd_test_142.json"
OUT_DIR="$ROOT/outputs/coder_sft_v5"
mkdir -p "$OUT_DIR"
python "$ROOT/evaluation/batch_infer.py" \
  --base-model "$BASE" \
  --adapter "$ADAPTER" \
  --data "$DATA" \
  --output "$OUT_DIR/test142_predictions.jsonl"
python "$ROOT/evaluation/evaluate_predictions_normalized.py" \
  --input "$OUT_DIR/test142_predictions.jsonl" \
  --output "$OUT_DIR/test142_eval.jsonl" \
  --summary "$OUT_DIR/test142_summary.json" \
  --csv "$OUT_DIR/test142_review.csv"
cat "$OUT_DIR/test142_summary.json"
""",
        encoding="utf-8",
    )
    (scripts / "16_pairwise_coder_sft_v5.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import csv
import json
from pathlib import Path

base = Path("/hy-tmp/rema_full_sft_4090_20260614/outputs/full_sft_v4_test142_eval.jsonl")
new = Path("/hy-tmp/rema_mappo_v0_tiny_20260615/outputs/coder_sft_v5/test142_eval.jsonl")
out = Path("/hy-tmp/rema_coder_sft_v5_vs_full_sft_pairwise.csv")
summary = Path("/hy-tmp/rema_coder_sft_v5_vs_full_sft_pairwise_summary.json")

def load(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[item["id"]] = item
    return rows

b, n = load(base), load(new)
metrics = ["total", "class_f1", "attribute_f1", "method_f1", "relation_pair_f1", "relation_label_f1", "multiplicity_f1"]
rows = []
for key in sorted(set(b) & set(n)):
    br, nr = b[key]["normalized_reward"], n[key]["normalized_reward"]
    row = {
        "id": key,
        "base_total": br["total"],
        "coder_sft_v5_total": nr["total"],
        "delta_total": nr["total"] - br["total"],
        "requirement": n[key].get("requirement", ""),
        "base_prediction": b[key].get("prediction", ""),
        "coder_sft_v5_prediction": n[key].get("prediction", ""),
        "gold": n[key].get("gold_plantuml", ""),
    }
    for metric in metrics[1:]:
        row[f"delta_{metric}"] = nr[metric] - br[metric]
    rows.append(row)

wins = sum(row["delta_total"] > 1e-6 for row in rows)
losses = sum(row["delta_total"] < -1e-6 for row in rows)
ties = len(rows) - wins - losses
avg_delta = {"total": sum(row["delta_total"] for row in rows) / len(rows)}
for metric in metrics[1:]:
    avg_delta[metric] = sum(row[f"delta_{metric}"] for row in rows) / len(rows)

summary_obj = {
    "baseline": str(base),
    "coder_sft_v5": str(new),
    "count": len(rows),
    "wins": wins,
    "losses": losses,
    "ties": ties,
    "avg_delta": avg_delta,
    "top_improved_ids": [row["id"] for row in sorted(rows, key=lambda item: item["delta_total"], reverse=True)[:10]],
    "top_degraded_ids": [row["id"] for row in sorted(rows, key=lambda item: item["delta_total"])[:10]],
}
with out.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda item: item["delta_total"]))
summary.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary_obj, ensure_ascii=False, indent=2))
PY
""",
        encoding="utf-8",
    )
    (scripts / "17_pack_coder_sft_v5.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
tar -czf /hy-tmp/rema_coder_sft_v5_results_$(date +%Y%m%d_%H%M%S).tar.gz \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/coder_sft_v5 \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 outputs/coder_sft_v5 \
  -C /hy-tmp rema_coder_sft_v5.log \
  -C /hy-tmp rema_coder_sft_v5_eval.log \
  -C /hy-tmp rema_coder_sft_v5_pairwise.log \
  -C /hy-tmp rema_coder_sft_v5_vs_full_sft_pairwise.csv \
  -C /hy-tmp rema_coder_sft_v5_vs_full_sft_pairwise_summary.json
ls -lh /hy-tmp/rema_coder_sft_v5_results_*.tar.gz
""",
        encoding="utf-8",
    )
    for script in scripts.glob("1*_coder_sft_v5.sh"):
        script.chmod(0o755)
    print(f"Wrote scripts in {scripts}")


def main() -> None:
    copy_data()
    merge_dataset_info()
    write_yaml()
    write_scripts()
    print("Coder-SFT-v5 package applied.")


if __name__ == "__main__":
    main()
