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
SCRIPTS = ROOT / "cloud_scripts"


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def copy_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "rema_agent_aware_sft_v0_train.json",
        "rema_agent_aware_sft_v0_eval.json",
        "plantucd_test_142.json",
    ]:
        src = PACKAGE / "data" / name
        require(src, name)
        dst = DATA_DIR / name
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")


def copy_evaluation_scripts() -> None:
    eval_dir = ROOT / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for name in ["batch_infer.py", "evaluate_predictions_normalized.py"]:
        src = PACKAGE / "evaluation" / name
        require(src, name)
        dst = eval_dir / name
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")


def merge_dataset_info() -> None:
    info_path = DATA_DIR / "dataset_info.json"
    require(info_path, "dataset_info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    patch = json.loads((PACKAGE / "data" / "dataset_info_patch.json").read_text(encoding="utf-8"))
    info.update(patch)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {info_path}")


def write_yaml() -> None:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    coder_v5 = ROOT / "LLaMA-Factory" / "saves" / "llama3-8b" / "lora" / "rema_plantuml" / "coder_sft_v5"
    require(coder_v5 / "adapter_config.json", "Coder-SFT-v5 adapter_config.json")
    yaml = f"""### model
model_name_or_path: /hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
adapter_name_or_path: {coder_v5.as_posix()}
quantization_bit: 4
quantization_method: bitsandbytes
upcast_layernorm: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: rema_agent_aware_sft_v0_train
eval_dataset: rema_agent_aware_sft_v0_eval
dataset_dir: data
template: llama3
cutoff_len: 3072
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: saves/llama3-8b/lora/rema_plantuml/agent_aware_sft_v0_tiny64
logging_steps: 2
save_steps: 50
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
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
eval_steps: 50
"""
    path = EXAMPLES / "rema_agent_aware_sft_v0_tiny64.yaml"
    path.write_text(yaml, encoding="utf-8")
    print(f"Wrote {path}")


def write_scripts() -> None:
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    (SCRIPTS / "31_run_agent_aware_sft_v0.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
cd /hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory
llamafactory-cli train examples/train_lora/rema_agent_aware_sft_v0_tiny64.yaml
""", encoding="utf-8")

    (SCRIPTS / "32_eval_agent_aware_sft_v0.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
ROOT=/hy-tmp/rema_mappo_v0_tiny_20260615
BASE=/hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
ADAPTER="$ROOT/LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/agent_aware_sft_v0_tiny64"
DATA="$ROOT/LLaMA-Factory/data/plantucd_test_142.json"
OUT_DIR="$ROOT/outputs/agent_aware_sft_v0_tiny64"
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
""", encoding="utf-8")

    (SCRIPTS / "33_pairwise_agent_aware_sft_v0.sh").write_text(r"""#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import csv
import json
from pathlib import Path

root = Path("/hy-tmp/rema_mappo_v0_tiny_20260615")
new = root / "outputs/agent_aware_sft_v0_tiny64/test142_eval.jsonl"
comparisons = {
    "coder_sft_v5": root / "outputs/coder_sft_v5/test142_eval.jsonl",
    "mappo_v3_tiny64": root / "outputs/mappo_v3_constrained_tiny64/test142_eval.jsonl",
    "full_sft_v4": Path("/hy-tmp/rema_full_sft_4090_20260614/outputs/full_sft_v4_test142_eval.jsonl"),
}
out_json = Path("/hy-tmp/rema_agent_aware_sft_v0_pairwise_summary.json")
out_csv = Path("/hy-tmp/rema_agent_aware_sft_v0_pairwise.csv")
metrics = ["total", "class_f1", "attribute_f1", "method_f1", "relation_pair_f1", "relation_label_f1", "multiplicity_f1"]

def load(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[item["id"]] = item
    return rows

new_rows = load(new)
summary = {"agent_aware_sft_v0": str(new), "comparisons": {}}
csv_rows = []
for name, path in comparisons.items():
    if not path.exists():
        summary["comparisons"][name] = {"missing": str(path)}
        continue
    base_rows = load(path)
    common = sorted(set(base_rows) & set(new_rows))
    deltas = []
    for key in common:
        br = base_rows[key]["normalized_reward"]
        nr = new_rows[key]["normalized_reward"]
        row = {
            "comparison": name,
            "id": key,
            "baseline_total": br["total"],
            "agent_total": nr["total"],
            "delta_total": nr["total"] - br["total"],
            "requirement": new_rows[key].get("requirement", ""),
            "baseline_prediction": base_rows[key].get("prediction", ""),
            "agent_prediction": new_rows[key].get("prediction", ""),
            "gold": new_rows[key].get("gold_plantuml", ""),
        }
        for metric in metrics[1:]:
            row[f"delta_{metric}"] = nr[metric] - br[metric]
        csv_rows.append(row)
        deltas.append(row)
    wins = sum(item["delta_total"] > 1e-6 for item in deltas)
    losses = sum(item["delta_total"] < -1e-6 for item in deltas)
    ties = len(deltas) - wins - losses
    avg_delta = {"total": sum(item["delta_total"] for item in deltas) / len(deltas)}
    for metric in metrics[1:]:
        avg_delta[metric] = sum(item[f"delta_{metric}"] for item in deltas) / len(deltas)
    summary["comparisons"][name] = {
        "baseline": str(path),
        "count": len(deltas),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "avg_delta": avg_delta,
        "top_improved_ids": [item["id"] for item in sorted(deltas, key=lambda item: item["delta_total"], reverse=True)[:10]],
        "top_degraded_ids": [item["id"] for item in sorted(deltas, key=lambda item: item["delta_total"])[:10]],
    }

if csv_rows:
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(csv_rows, key=lambda item: (item["comparison"], item["delta_total"])))
summary["csv"] = str(out_csv)
out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
""", encoding="utf-8")

    (SCRIPTS / "34_pack_agent_aware_sft_v0.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
tar -czf /hy-tmp/rema_agent_aware_sft_v0_tiny64_results_$(date +%Y%m%d_%H%M%S).tar.gz \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/agent_aware_sft_v0_tiny64 \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 outputs/agent_aware_sft_v0_tiny64 \
  -C /hy-tmp rema_agent_aware_sft_v0.log \
  -C /hy-tmp rema_agent_aware_sft_v0_eval.log \
  -C /hy-tmp rema_agent_aware_sft_v0_pairwise.log \
  -C /hy-tmp rema_agent_aware_sft_v0_pairwise.csv \
  -C /hy-tmp rema_agent_aware_sft_v0_pairwise_summary.json
ls -lh /hy-tmp/rema_agent_aware_sft_v0_tiny64_results_*.tar.gz
""", encoding="utf-8")

    for script in SCRIPTS.glob("3*_agent_aware_sft_v0.sh"):
        script.chmod(0o755)
    print(f"Wrote scripts in {SCRIPTS}")


def main() -> None:
    require(FACTORY, "LLaMA-Factory root")
    copy_data()
    copy_evaluation_scripts()
    merge_dataset_info()
    write_yaml()
    write_scripts()
    print("Agent-aware SFT v0 package applied.")


if __name__ == "__main__":
    main()
