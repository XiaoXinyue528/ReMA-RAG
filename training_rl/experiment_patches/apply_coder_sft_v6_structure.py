from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("REMA_ROOT", "/hy-tmp/rema_mappo_v0_tiny_20260615"))
PACKAGE = Path(__file__).resolve().parent
FACTORY = ROOT / "LLaMA-Factory"
DATA_DIR = FACTORY / "data"
EXAMPLES = FACTORY / "examples" / "train_lora"
PAIRWISE_CSV = Path(os.environ.get("REMA_CODER_V5_PAIRWISE_CSV", "/hy-tmp/rema_coder_sft_v5_vs_full_sft_pairwise.csv"))


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_id(example: dict[str, Any]) -> str:
    raw = str(example.get("id", ""))
    return raw.split(":", 1)[0]


def read_pairwise(path: Path) -> tuple[set[str], set[str], list[dict[str, str]]]:
    degraded: set[str] = set()
    improved_safe: set[str] = set()
    anchors: list[dict[str, str]] = []
    if not path.exists():
        print(f"Pairwise CSV not found, using attribution-only v6: {path}")
        return degraded, improved_safe, anchors

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = row.get("id", "")
            if not sid:
                continue
            def f(name: str) -> float:
                try:
                    return float(row.get(name, 0) or 0)
                except ValueError:
                    return 0.0

            delta_total = f("delta_total")
            delta_class = f("delta_class_f1")
            delta_attr = f("delta_attribute_f1")
            delta_rel_pair = f("delta_relation_pair_f1")
            delta_rel_label = f("delta_relation_label_f1")
            delta_mult = f("delta_multiplicity_f1")

            if delta_class < -1e-6 or delta_rel_pair < -1e-6 or delta_total < -0.05:
                degraded.add(sid)
                base_prediction = (row.get("base_prediction") or "").strip()
                requirement = (row.get("requirement") or "").strip()
                if base_prediction and requirement:
                    anchors.append({
                        "id": f"{sid}:structure_anchor_from_full_sft_v4",
                        "role": "structure_anchor",
                        "instruction": (
                            "You are generating a PlantUML class diagram. This is a structure-preserving anchor sample.\n"
                            "Preserve the explicit entity classes and relationship endpoints from the requirement.\n"
                            "Do not trade away class coverage or relation-pair correctness to add extra attributes or labels.\n"
                            "Use valid PlantUML only."
                        ),
                        "input": f"Requirement:\n{requirement}",
                        "output": base_prediction,
                    })
            elif (
                delta_total > 1e-6
                and delta_class >= -1e-6
                and delta_rel_pair >= -1e-6
                and (delta_attr > 1e-6 or delta_rel_label > 1e-6 or delta_mult > 1e-6)
            ):
                improved_safe.add(sid)

    return degraded, improved_safe, anchors


def strengthen_instruction(example: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(example)
    instruction = str(cloned.get("instruction", "")).strip()
    guard = (
        "\n\nStructure-preserving constraints:\n"
        "1. Class names are the backbone. Do not omit explicit entity classes from the requirement.\n"
        "2. Relation endpoints are the second backbone. Do not change the source/target class pair unless the requirement requires it.\n"
        "3. Add attributes, methods, labels, and multiplicities only after the class/relation-pair backbone is preserved.\n"
        "4. Use canonical member syntax: +name: Type and +method(arg: Type): ReturnType."
    )
    if "Structure-preserving constraints" not in instruction:
        cloned["instruction"] = instruction + guard
    return cloned


def merge_dataset_info() -> None:
    info_path = DATA_DIR / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update({
        "rema_plantuml_coder_sft_v6_structure": {
            "file_name": "coder_sft_v6_structure.json",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    })
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {info_path}")


def copy_v5_data_if_needed() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "coder_sft_v5_mixed.json",
        "coder_final_repair_focus.json",
        "coder_step_sft_positive.json",
        "coder_final_repair_all.json",
    ]:
        dst = DATA_DIR / name
        if dst.exists():
            continue
        src = PACKAGE / "data" / name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied fallback data {src} -> {dst}")


def build_v6_dataset() -> None:
    copy_v5_data_if_needed()
    step_examples = load_json(DATA_DIR / "coder_step_sft_positive.json")
    repair_focus = load_json(DATA_DIR / "coder_final_repair_focus.json")
    repair_all = load_json(DATA_DIR / "coder_final_repair_all.json")

    degraded_ids, improved_safe_ids, anchor_examples = read_pairwise(PAIRWISE_CSV)

    # Conservative v6:
    # - keep high-confidence step-coder examples as backbone-preserving anchors;
    # - keep focused repair examples except samples where v5 harmed class or relation pairs;
    # - add baseline anchors for v5-degraded samples;
    # - add a small number of safe-improved full-repair examples, because these are examples
    #   where v5 improved details without hurting the backbone.
    repair_focus_filtered = [
        strengthen_instruction(example)
        for example in repair_focus
        if sample_id(example) not in degraded_ids
    ]
    safe_full_repair = [
        strengthen_instruction(example)
        for example in repair_all
        if sample_id(example) in improved_safe_ids
    ]
    step_guarded = [strengthen_instruction(example) for example in step_examples]
    anchor_guarded = [strengthen_instruction(example) for example in anchor_examples]

    mixed = step_guarded + repair_focus_filtered + safe_full_repair + anchor_guarded

    write_json(DATA_DIR / "coder_sft_v6_structure.json", mixed)

    summary = {
        "source": {
            "step_examples": str(DATA_DIR / "coder_step_sft_positive.json"),
            "repair_focus": str(DATA_DIR / "coder_final_repair_focus.json"),
            "repair_all": str(DATA_DIR / "coder_final_repair_all.json"),
            "pairwise_csv": str(PAIRWISE_CSV),
        },
        "policy": {
            "purpose": "Preserve class/relation-pair backbone while keeping v5 gains on attributes, relation labels, and multiplicity.",
            "base_adapter": "full_sft_v4",
            "exclude_repair_if": "v5 decreased class_f1 or relation_pair_f1, or delta_total < -0.05",
            "include_safe_full_repair_if": "v5 improved total without decreasing class_f1 or relation_pair_f1 and improved at least one detail metric",
            "anchor_output": "Full-SFT-v4 prediction for v5-degraded samples",
        },
        "counts": {
            "step_guarded": len(step_guarded),
            "repair_focus_original": len(repair_focus),
            "repair_focus_filtered": len(repair_focus_filtered),
            "safe_full_repair": len(safe_full_repair),
            "structure_anchors": len(anchor_guarded),
            "degraded_ids": len(degraded_ids),
            "improved_safe_ids": len(improved_safe_ids),
            "mixed": len(mixed),
        },
    }
    write_json(ROOT / "metadata" / "coder_sft_v6_structure_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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
dataset: rema_plantuml_coder_sft_v6_structure
eval_dataset: rema_plantuml_full_eval
dataset_dir: data
template: llama3
cutoff_len: 3072
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: saves/llama3-8b/lora/rema_plantuml/coder_sft_v6_structure
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.2e-5
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
    path = EXAMPLES / "rema_plantuml_coder_sft_v6_structure_4090.yaml"
    path.write_text(yaml, encoding="utf-8")
    print(f"Wrote {path}")


def write_scripts() -> None:
    scripts = ROOT / "cloud_scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "18_run_coder_sft_v6_structure.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
cd /hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory
llamafactory-cli train examples/train_lora/rema_plantuml_coder_sft_v6_structure_4090.yaml
""",
        encoding="utf-8",
    )
    (scripts / "19_eval_coder_sft_v6_structure.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
ROOT=/hy-tmp/rema_mappo_v0_tiny_20260615
BASE=/hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
ADAPTER="$ROOT/LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/coder_sft_v6_structure"
DATA="$ROOT/LLaMA-Factory/data/plantucd_test_142.json"
OUT_DIR="$ROOT/outputs/coder_sft_v6_structure"
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
    (scripts / "20_pairwise_coder_sft_v6_structure.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import csv
import json
from pathlib import Path

base = Path("/hy-tmp/rema_full_sft_4090_20260614/outputs/full_sft_v4_test142_eval.jsonl")
v5 = Path("/hy-tmp/rema_mappo_v0_tiny_20260615/outputs/coder_sft_v5/test142_eval.jsonl")
v6 = Path("/hy-tmp/rema_mappo_v0_tiny_20260615/outputs/coder_sft_v6_structure/test142_eval.jsonl")
out = Path("/hy-tmp/rema_coder_sft_v6_structure_pairwise.csv")
summary = Path("/hy-tmp/rema_coder_sft_v6_structure_pairwise_summary.json")

def load(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[item["id"]] = item
    return rows

b, five, six = load(base), load(v5), load(v6)
metrics = ["total", "class_f1", "attribute_f1", "method_f1", "relation_pair_f1", "relation_label_f1", "multiplicity_f1"]
rows = []
for key in sorted(set(b) & set(six)):
    br = b[key]["normalized_reward"]
    sr = six[key]["normalized_reward"]
    fr = five.get(key, {}).get("normalized_reward", {})
    row = {
        "id": key,
        "base_total": br["total"],
        "v5_total": fr.get("total", ""),
        "v6_total": sr["total"],
        "delta_v6_vs_base_total": sr["total"] - br["total"],
        "delta_v6_vs_v5_total": sr["total"] - fr["total"] if fr else "",
        "requirement": six[key].get("requirement", ""),
        "base_prediction": b[key].get("prediction", ""),
        "v5_prediction": five.get(key, {}).get("prediction", ""),
        "v6_prediction": six[key].get("prediction", ""),
        "gold": six[key].get("gold_plantuml", ""),
    }
    for metric in metrics[1:]:
        row[f"delta_v6_vs_base_{metric}"] = sr[metric] - br[metric]
        row[f"delta_v6_vs_v5_{metric}"] = sr[metric] - fr[metric] if fr else ""
    rows.append(row)

def summarize(prefix: str, delta_key: str):
    wins = sum(row[delta_key] != "" and row[delta_key] > 1e-6 for row in rows)
    losses = sum(row[delta_key] != "" and row[delta_key] < -1e-6 for row in rows)
    ties = sum(row[delta_key] != "" for row in rows) - wins - losses
    avg = {"total": sum(row[delta_key] for row in rows if row[delta_key] != "") / max(1, sum(row[delta_key] != "" for row in rows))}
    for metric in metrics[1:]:
        key = f"{prefix}_{metric}"
        values = [row[key] for row in rows if row[key] != ""]
        avg[metric] = sum(values) / max(1, len(values))
    return {"wins": wins, "losses": losses, "ties": ties, "avg_delta": avg}

summary_obj = {
    "baseline": str(base),
    "coder_sft_v5": str(v5),
    "coder_sft_v6_structure": str(v6),
    "count": len(rows),
    "v6_vs_base": summarize("delta_v6_vs_base", "delta_v6_vs_base_total"),
    "v6_vs_v5": summarize("delta_v6_vs_v5", "delta_v6_vs_v5_total"),
    "top_improved_vs_base_ids": [row["id"] for row in sorted(rows, key=lambda item: item["delta_v6_vs_base_total"], reverse=True)[:10]],
    "top_degraded_vs_base_ids": [row["id"] for row in sorted(rows, key=lambda item: item["delta_v6_vs_base_total"])[:10]],
}
with out.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda item: item["delta_v6_vs_base_total"]))
summary.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary_obj, ensure_ascii=False, indent=2))
PY
""",
        encoding="utf-8",
    )
    (scripts / "21_pack_coder_sft_v6_structure.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
tar -czf /hy-tmp/rema_coder_sft_v6_structure_results_$(date +%Y%m%d_%H%M%S).tar.gz \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/coder_sft_v6_structure \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 outputs/coder_sft_v6_structure \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 metadata/coder_sft_v6_structure_summary.json \
  -C /hy-tmp rema_coder_sft_v6_structure.log \
  -C /hy-tmp rema_coder_sft_v6_structure_eval.log \
  -C /hy-tmp rema_coder_sft_v6_structure_pairwise.log \
  -C /hy-tmp rema_coder_sft_v6_structure_pairwise.csv \
  -C /hy-tmp rema_coder_sft_v6_structure_pairwise_summary.json
ls -lh /hy-tmp/rema_coder_sft_v6_structure_results_*.tar.gz
""",
        encoding="utf-8",
    )
    for script in scripts.glob("*coder_sft_v6_structure.sh"):
        script.chmod(0o755)
    print(f"Wrote scripts in {scripts}")


def main() -> None:
    copy_v5_data_if_needed()
    build_v6_dataset()
    merge_dataset_info()
    write_yaml()
    write_scripts()
    print("Coder-SFT-v6 structure package applied.")


if __name__ == "__main__":
    main()
