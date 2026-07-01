from __future__ import annotations

import os
import re
from pathlib import Path


ROOT = Path(os.environ.get("REMA_ROOT", "/hy-tmp/rema_mappo_v0_tiny_20260615"))
TRAINER = ROOT / "LLaMA-Factory" / "src" / "llamafactory" / "train" / "ppo" / "trainer_qr_s_g.py"
V0_PATCH = ROOT / "patches" / "apply_plantuml_ppo_v0.py"
V3_YAML = ROOT / "LLaMA-Factory" / "examples" / "train_lora" / "rema_plantuml_mappo_v3_constrained_tiny64.yaml"


V3_BRANCH = """                elif reward_mode == "backbone_constrained_v3":
                    stripped_prediction = prediction.strip().lower()
                    is_plantuml_block = stripped_prediction.startswith("@startuml") and stripped_prediction.endswith("@enduml")

                    class_f1 = float(normalized["class_f1"])
                    attr_f1 = float(normalized["attribute_f1"])
                    method_f1 = float(normalized["method_f1"])
                    rel_pair_f1 = float(normalized["relation_pair_f1"])
                    rel_label_f1 = float(normalized["relation_label_f1"])
                    mult_f1 = float(normalized["multiplicity_f1"])
                    fmt = float(normalized["format_score"])
                    syntax = float(normalized["syntax_score"])

                    current_backbone = 0.56 * class_f1 + 0.44 * rel_pair_f1
                    current_details = (
                        0.42 * attr_f1
                        + 0.20 * method_f1
                        + 0.18 * rel_label_f1
                        + 0.20 * mult_f1
                    )
                    current_content_score = (
                        0.04 * fmt
                        + 0.06 * syntax
                        + 0.34 * class_f1
                        + 0.24 * rel_pair_f1
                        + 0.18 * attr_f1
                        + 0.06 * method_f1
                        + 0.04 * rel_label_f1
                        + 0.04 * mult_f1
                    )

                    baseline_item = baseline_map.get(sample_id, {})
                    baseline = baseline_item.get("normalized_reward", {})
                    if baseline:
                        base_class = float(baseline.get("class_f1", 0.0))
                        base_attr = float(baseline.get("attribute_f1", 0.0))
                        base_method = float(baseline.get("method_f1", 0.0))
                        base_rel_pair = float(baseline.get("relation_pair_f1", 0.0))
                        base_rel_label = float(baseline.get("relation_label_f1", 0.0))
                        base_mult = float(baseline.get("multiplicity_f1", 0.0))
                        base_fmt = float(baseline.get("format_score", 0.0))
                        base_syntax = float(baseline.get("syntax_score", 0.0))

                        baseline_backbone = 0.56 * base_class + 0.44 * base_rel_pair
                        baseline_details = (
                            0.42 * base_attr
                            + 0.20 * base_method
                            + 0.18 * base_rel_label
                            + 0.20 * base_mult
                        )
                        baseline_content_score = (
                            0.04 * base_fmt
                            + 0.06 * base_syntax
                            + 0.34 * base_class
                            + 0.24 * base_rel_pair
                            + 0.18 * base_attr
                            + 0.06 * base_method
                            + 0.04 * base_rel_label
                            + 0.04 * base_mult
                        )

                        class_drop = max(0.0, base_class - class_f1 - 0.005)
                        rel_pair_drop = max(0.0, base_rel_pair - rel_pair_f1 - 0.005)
                        attr_drop = max(0.0, base_attr - attr_f1 - 0.010)
                        backbone_drop = max(0.0, baseline_backbone - current_backbone)
                        detail_gain = max(-0.15, min(0.15, current_details - baseline_details))
                        content_delta = max(-0.20, min(0.20, current_content_score - baseline_content_score))

                        detail_bonus = 0.0
                        if class_drop <= 0.0 and rel_pair_drop <= 0.0:
                            detail_bonus = max(0.0, 0.50 * detail_gain)

                        score = (
                            current_content_score
                            + 0.55 * content_delta
                            + detail_bonus
                            - 2.40 * class_drop
                            - 2.00 * rel_pair_drop
                            - 0.60 * attr_drop
                            - 1.20 * backbone_drop
                        )

                        if class_drop > 0.08 or rel_pair_drop > 0.08:
                            score = min(score, baseline_content_score - 0.08)
                    else:
                        score = current_content_score

                    if not is_plantuml_block:
                        score = min(score, 0.05)
                    if fmt < 1.0 or syntax < 1.0:
                        score = min(score, 0.20)
                    if class_f1 < 0.50:
                        score = min(score, 0.35)
                    elif class_f1 < 0.75:
                        score = min(score, 0.55)
                    if rel_pair_f1 < 0.35:
                        score = min(score, 0.50)
"""


def ensure_v0_patch() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    if "def plantuml_ppo_train" in text and "REMA_PLANTUML_PPO" in text:
        return
    if not V0_PATCH.exists():
        raise SystemExit(f"PlantUML PPO-v0 patch is missing: {V0_PATCH}")
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(V0_PATCH.read_text(encoding="utf-8"), namespace)


def patch_baseline_loader() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    if "baseline_map = {}" in text and "REMA_PLANTUML_BASELINE_MAP" in text:
        print("Baseline map loader already exists.")
        return
    old = """        reward_map = self._plantuml_load_reward_map()
        trace_path = os.path.join(self.args.output_dir, "plantuml_reward_trace.jsonl")
"""
    new = """        reward_map = self._plantuml_load_reward_map()
        baseline_map = {}
        baseline_map_path = os.environ.get("REMA_PLANTUML_BASELINE_MAP")
        if baseline_map_path:
            with open(baseline_map_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        item = json.loads(line)
                        baseline_map[item["id"]] = item
        trace_path = os.path.join(self.args.output_dir, "plantuml_reward_trace.jsonl")
"""
    if old not in text:
        raise SystemExit("Could not find reward_map loader block.")
    TRAINER.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("Patched baseline map loader.")


def patch_reward_mode() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    if 'reward_mode == "backbone_constrained_v3"' in text:
        print("backbone_constrained_v3 reward mode already exists.")
        return
    needle = """                else:
                    score = float(normalized["total"])

                if float(normalized["format_score"]) < 1.0 or float(normalized["syntax_score"]) < 1.0:
"""
    replacement = V3_BRANCH + """                else:
                    score = float(normalized["total"])

                if float(normalized["format_score"]) < 1.0 or float(normalized["syntax_score"]) < 1.0:
"""
    if needle not in text:
        raise SystemExit("Could not find final reward fallback block for v3 insertion.")
    TRAINER.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    print("Patched backbone_constrained_v3 reward mode.")


def write_yaml() -> None:
    yaml = """### model
model_name_or_path: /hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
adapter_name_or_path: /hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/coder_sft_v5
quantization_bit: 4
quantization_method: bitsandbytes
upcast_layernorm: true
reward_model_type: api
reward_model: http://127.0.0.1:9/unused

### method
stage: ppo
do_train: true
finetuning_type: lora
lora_target: all
ppo_buffer_size: 1
ppo_epochs: 1
ppo_target: 4.0

### dataset
dataset: rema_plantuml_mappo_tiny64
dataset_dir: data
template: llama3
cutoff_len: 3072
max_samples: 64
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: saves/llama3-8b/lora/rema_plantuml/mappo_v3_constrained_tiny64
logging_steps: 1
save_steps: 1000
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 5.0e-7
max_steps: 20
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000

### generate
max_new_tokens: 512
top_k: 0
top_p: 0.9
"""
    V3_YAML.write_text(yaml, encoding="utf-8")
    print(f"Wrote {V3_YAML}")


def write_helper_scripts() -> None:
    evaluation_dir = ROOT / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "batch_infer_preserve_id.py").write_text(
        """from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def extract_plantuml(text: str) -> str:
    value = (text or "").strip()
    lower = value.lower()
    start = lower.find("@startuml")
    end = lower.find("@enduml", start + 1) if start >= 0 else -1
    if start >= 0 and end >= 0:
        return value[start : end + len("@enduml")].strip()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.limit > 0:
        rows = rows[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            requirement = row["HumanLang"]
            prompt = "Combine all step snippets into final PlantUML for requirement:\\n" + requirement
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=768,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            prediction = extract_plantuml(tokenizer.decode(new_tokens, skip_special_tokens=True))
            item = {
                "id": row.get("id", f"plantucd_test_{index - 1}"),
                "requirement": requirement,
                "gold_plantuml": row["PlantUML"],
                "prediction": prediction,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\\n")
            print(f"[{index}/{len(rows)}] {item['id']}", flush=True)


if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    scripts = ROOT / "cloud_scripts"
    scripts.mkdir(parents=True, exist_ok=True)

    (scripts / "22_build_mappo_v3_coder_sft_v5_baseline.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
ROOT=/hy-tmp/rema_mappo_v0_tiny_20260615
BASE=/hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
ADAPTER="$ROOT/LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/coder_sft_v5"
OUT_DIR="$ROOT/outputs/mappo_v3_coder_sft_v5_baseline"
mkdir -p "$OUT_DIR"

python - <<'PY'
import json
import re
from pathlib import Path

root = Path("/hy-tmp/rema_mappo_v0_tiny_20260615")
data = json.loads((root / "LLaMA-Factory/data/rema_plantuml_mappo_tiny64.json").read_text(encoding="utf-8"))
reward_rows = {}
for line in (root / "LLaMA-Factory/data/rema_plantuml_mappo_reward_map.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        item = json.loads(line)
        reward_rows[item["id"]] = item

rows = []
for item in data:
    sample_id = item.get("id")
    if not sample_id:
        match = re.search(r"Sample ID:\\s*([^\\n]+)", item.get("instruction", ""))
        sample_id = match.group(1).strip()
    reward = reward_rows[sample_id]
    rows.append({"id": sample_id, "HumanLang": reward["requirement"], "PlantUML": reward["gold_plantuml"]})

out = root / "outputs/mappo_v3_coder_sft_v5_baseline/baseline_data_tiny64.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote", out, "count", len(rows))
PY

python "$ROOT/evaluation/batch_infer_preserve_id.py" \
  --base-model "$BASE" \
  --adapter "$ADAPTER" \
  --data "$OUT_DIR/baseline_data_tiny64.json" \
  --output "$OUT_DIR/coder_sft_v5_baseline_predictions.jsonl"

python "$ROOT/evaluation/evaluate_predictions_normalized.py" \
  --input "$OUT_DIR/coder_sft_v5_baseline_predictions.jsonl" \
  --output "$OUT_DIR/coder_sft_v5_baseline_eval.jsonl" \
  --summary "$OUT_DIR/coder_sft_v5_baseline_summary.json" \
  --csv "$OUT_DIR/coder_sft_v5_baseline_review.csv"

python - <<'PY'
import json
from pathlib import Path

root = Path("/hy-tmp/rema_mappo_v0_tiny_20260615")
src = root / "outputs/mappo_v3_coder_sft_v5_baseline/coder_sft_v5_baseline_eval.jsonl"
dst = root / "LLaMA-Factory/data/rema_plantuml_mappo_baseline_coder_sft_v5_tiny64.jsonl"
with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        out.write(json.dumps({
            "id": item["id"],
            "normalized_reward": item["normalized_reward"],
            "strict_reward": item["strict_reward"],
            "prediction": item["prediction"],
        }, ensure_ascii=False) + "\\n")
print("Wrote", dst)
PY

cat "$OUT_DIR/coder_sft_v5_baseline_summary.json"
""",
        encoding="utf-8",
    )

    (scripts / "23_run_mappo_v3_constrained_tiny64.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
cd /hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory
export REMA_PLANTUML_PPO=1
export REMA_FUSION_DIR=/hy-tmp/rema_mappo_v0_tiny_20260615/fusion
export REMA_PLANTUML_REWARD_MAP=/hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory/data/rema_plantuml_mappo_reward_map.jsonl
export REMA_PLANTUML_BASELINE_MAP=/hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory/data/rema_plantuml_mappo_baseline_coder_sft_v5_tiny64.jsonl
export REMA_PLANTUML_REWARD_MODE=backbone_constrained_v3
export REMA_PLANTUML_MAX_NEW_TOKENS=512
llamafactory-cli train examples/train_lora/rema_plantuml_mappo_v3_constrained_tiny64.yaml
""",
        encoding="utf-8",
    )

    (scripts / "24_eval_mappo_v3_constrained_tiny64.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
ROOT=/hy-tmp/rema_mappo_v0_tiny_20260615
BASE=/hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
ADAPTER="$ROOT/LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/mappo_v3_constrained_tiny64"
DATA="$ROOT/LLaMA-Factory/data/plantucd_test_142.json"
OUT_DIR="$ROOT/outputs/mappo_v3_constrained_tiny64"
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

    (scripts / "25_pairwise_mappo_v3_constrained_tiny64.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import csv
import json
from pathlib import Path

base = Path("/hy-tmp/rema_full_sft_4090_20260614/outputs/full_sft_v4_test142_eval.jsonl")
v5 = Path("/hy-tmp/rema_mappo_v0_tiny_20260615/outputs/coder_sft_v5/test142_eval.jsonl")
v3 = Path("/hy-tmp/rema_mappo_v0_tiny_20260615/outputs/mappo_v3_constrained_tiny64/test142_eval.jsonl")
out = Path("/hy-tmp/rema_mappo_v3_constrained_tiny64_pairwise.csv")
summary = Path("/hy-tmp/rema_mappo_v3_constrained_tiny64_pairwise_summary.json")

def load(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[item["id"]] = item
    return rows

b, five, three = load(base), load(v5), load(v3)
metrics = ["total", "class_f1", "attribute_f1", "method_f1", "relation_pair_f1", "relation_label_f1", "multiplicity_f1"]
rows = []
for key in sorted(set(b) & set(three)):
    br = b[key]["normalized_reward"]
    tr = three[key]["normalized_reward"]
    fr = five.get(key, {}).get("normalized_reward", {})
    row = {
        "id": key,
        "base_total": br["total"],
        "coder_sft_v5_total": fr.get("total", ""),
        "mappo_v3_total": tr["total"],
        "delta_v3_vs_base_total": tr["total"] - br["total"],
        "delta_v3_vs_v5_total": tr["total"] - fr["total"] if fr else "",
        "requirement": three[key].get("requirement", ""),
        "base_prediction": b[key].get("prediction", ""),
        "coder_sft_v5_prediction": five.get(key, {}).get("prediction", ""),
        "mappo_v3_prediction": three[key].get("prediction", ""),
        "gold": three[key].get("gold_plantuml", ""),
    }
    for metric in metrics[1:]:
        row[f"delta_v3_vs_base_{metric}"] = tr[metric] - br[metric]
        row[f"delta_v3_vs_v5_{metric}"] = tr[metric] - fr[metric] if fr else ""
    rows.append(row)

def summarize(prefix, delta_key):
    values = [row[delta_key] for row in rows if row[delta_key] != ""]
    wins = sum(value > 1e-6 for value in values)
    losses = sum(value < -1e-6 for value in values)
    ties = len(values) - wins - losses
    avg = {"total": sum(values) / max(1, len(values))}
    for metric in metrics[1:]:
        key = f"{prefix}_{metric}"
        vals = [row[key] for row in rows if row[key] != ""]
        avg[metric] = sum(vals) / max(1, len(vals))
    return {"wins": wins, "losses": losses, "ties": ties, "avg_delta": avg}

summary_obj = {
    "baseline": str(base),
    "coder_sft_v5": str(v5),
    "mappo_v3_constrained": str(v3),
    "count": len(rows),
    "v3_vs_base": summarize("delta_v3_vs_base", "delta_v3_vs_base_total"),
    "v3_vs_v5": summarize("delta_v3_vs_v5", "delta_v3_vs_v5_total"),
    "top_improved_vs_base_ids": [row["id"] for row in sorted(rows, key=lambda item: item["delta_v3_vs_base_total"], reverse=True)[:10]],
    "top_degraded_vs_base_ids": [row["id"] for row in sorted(rows, key=lambda item: item["delta_v3_vs_base_total"])[:10]],
}
with out.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda item: item["delta_v3_vs_base_total"]))
summary.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary_obj, ensure_ascii=False, indent=2))
PY
""",
        encoding="utf-8",
    )

    (scripts / "26_pack_mappo_v3_constrained_tiny64.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
tar -czf /hy-tmp/rema_mappo_v3_constrained_tiny64_results_$(date +%Y%m%d_%H%M%S).tar.gz \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/mappo_v3_constrained_tiny64 \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 outputs/mappo_v3_constrained_tiny64 \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 outputs/mappo_v3_coder_sft_v5_baseline \
  -C /hy-tmp rema_mappo_v3_constrained_tiny64.log \
  -C /hy-tmp rema_mappo_v3_constrained_tiny64_eval.log \
  -C /hy-tmp rema_mappo_v3_coder_sft_v5_baseline.log \
  -C /hy-tmp rema_mappo_v3_constrained_tiny64_pairwise.log \
  -C /hy-tmp rema_mappo_v3_constrained_tiny64_pairwise.csv \
  -C /hy-tmp rema_mappo_v3_constrained_tiny64_pairwise_summary.json
ls -lh /hy-tmp/rema_mappo_v3_constrained_tiny64_results_*.tar.gz
""",
        encoding="utf-8",
    )

    for script in scripts.glob("*mappo_v3*"):
        script.chmod(0o755)
    print("Wrote v3 helper scripts.")


def main() -> None:
    if not TRAINER.exists():
        raise SystemExit(f"Trainer not found: {TRAINER}")
    ensure_v0_patch()
    patch_baseline_loader()
    patch_reward_mode()
    write_yaml()
    write_helper_scripts()
    print("MAPPO-v3 constrained-from-coder-sft-v5 patch is ready.")


if __name__ == "__main__":
    main()
