from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ.get("REMA_ROOT", "/hy-tmp/rema_mappo_v0_tiny_20260615"))
TRAINER = ROOT / "LLaMA-Factory" / "src" / "llamafactory" / "train" / "ppo" / "trainer_qr_s_g.py"
V0_PATCH = ROOT / "patches" / "apply_plantuml_ppo_v0.py"
V2_YAML = ROOT / "LLaMA-Factory" / "examples" / "train_lora" / "rema_plantuml_mappo_v2_tiny64.yaml"
TINY64_YAML = ROOT / "LLaMA-Factory" / "examples" / "train_lora" / "rema_plantuml_mappo_v1_tiny64.yaml"
V1_YAML = ROOT / "LLaMA-Factory" / "examples" / "train_lora" / "rema_plantuml_mappo_v1_tiny32.yaml"
V0_YAML = ROOT / "LLaMA-Factory" / "examples" / "train_lora" / "rema_plantuml_mappo_v0_tiny32.yaml"


V0_REWARD_BLOCK = """                if reward_mode == "relation_focus":
                    score = (
                        0.65 * float(normalized["total"])
                        + 0.20 * float(normalized["relation_pair_f1"])
                        + 0.10 * float(normalized["multiplicity_f1"])
                        + 0.05 * float(normalized["relation_label_f1"])
                    )
                else:
                    score = float(normalized["total"])
"""

V1_REWARD_BLOCK = """                if reward_mode == "relation_focus":
                    score = (
                        0.65 * float(normalized["total"])
                        + 0.20 * float(normalized["relation_pair_f1"])
                        + 0.10 * float(normalized["multiplicity_f1"])
                        + 0.05 * float(normalized["relation_label_f1"])
                    )
                elif reward_mode == "conservative_v1":
                    stripped_prediction = prediction.strip().lower()
                    is_plantuml_block = stripped_prediction.startswith("@startuml") and stripped_prediction.endswith("@enduml")
                    score = (
                        0.82 * float(normalized["total"])
                        + 0.06 * float(normalized["attribute_f1"])
                        + 0.05 * float(normalized["relation_pair_f1"])
                        + 0.03 * float(normalized["method_f1"])
                        + 0.02 * float(normalized["multiplicity_f1"])
                        + 0.02 * float(normalized["relation_label_f1"])
                    )
                    if not is_plantuml_block:
                        score = min(score, 0.05)
                    if float(normalized["class_f1"]) < 0.5:
                        score *= 0.4
                    if float(normalized["attribute_f1"]) < 0.2:
                        score -= 0.05
                else:
                    score = float(normalized["total"])
"""

V2_REWARD_BLOCK = """                if reward_mode == "relation_focus":
                    score = (
                        0.65 * float(normalized["total"])
                        + 0.20 * float(normalized["relation_pair_f1"])
                        + 0.10 * float(normalized["multiplicity_f1"])
                        + 0.05 * float(normalized["relation_label_f1"])
                    )
                elif reward_mode == "conservative_v1":
                    stripped_prediction = prediction.strip().lower()
                    is_plantuml_block = stripped_prediction.startswith("@startuml") and stripped_prediction.endswith("@enduml")
                    score = (
                        0.82 * float(normalized["total"])
                        + 0.06 * float(normalized["attribute_f1"])
                        + 0.05 * float(normalized["relation_pair_f1"])
                        + 0.03 * float(normalized["method_f1"])
                        + 0.02 * float(normalized["multiplicity_f1"])
                        + 0.02 * float(normalized["relation_label_f1"])
                    )
                    if not is_plantuml_block:
                        score = min(score, 0.05)
                    if float(normalized["class_f1"]) < 0.5:
                        score *= 0.4
                    if float(normalized["attribute_f1"]) < 0.2:
                        score -= 0.05
                elif reward_mode == "content_guarded_delta_v2":
                    stripped_prediction = prediction.strip().lower()
                    is_plantuml_block = stripped_prediction.startswith("@startuml") and stripped_prediction.endswith("@enduml")
                    current_content_score = (
                        0.03 * float(normalized["format_score"])
                        + 0.04 * float(normalized["syntax_score"])
                        + 0.30 * float(normalized["class_f1"])
                        + 0.24 * float(normalized["attribute_f1"])
                        + 0.10 * float(normalized["method_f1"])
                        + 0.16 * float(normalized["relation_pair_f1"])
                        + 0.05 * float(normalized["relation_label_f1"])
                        + 0.08 * float(normalized["multiplicity_f1"])
                    )
                    baseline_item = baseline_map.get(sample_id, {})
                    baseline = baseline_item.get("normalized_reward", {})
                    if baseline:
                        baseline_content_score = (
                            0.03 * float(baseline.get("format_score", 0.0))
                            + 0.04 * float(baseline.get("syntax_score", 0.0))
                            + 0.30 * float(baseline.get("class_f1", 0.0))
                            + 0.24 * float(baseline.get("attribute_f1", 0.0))
                            + 0.10 * float(baseline.get("method_f1", 0.0))
                            + 0.16 * float(baseline.get("relation_pair_f1", 0.0))
                            + 0.05 * float(baseline.get("relation_label_f1", 0.0))
                            + 0.08 * float(baseline.get("multiplicity_f1", 0.0))
                        )
                        delta_content = max(-0.20, min(0.20, current_content_score - baseline_content_score))
                        class_drop = max(0.0, float(baseline.get("class_f1", 0.0)) - float(normalized["class_f1"]) - 0.005)
                        attr_drop = max(0.0, float(baseline.get("attribute_f1", 0.0)) - float(normalized["attribute_f1"]) - 0.010)
                        relation_pair_gain = max(0.0, float(normalized["relation_pair_f1"]) - float(baseline.get("relation_pair_f1", 0.0)))
                        multiplicity_gain = max(0.0, float(normalized["multiplicity_f1"]) - float(baseline.get("multiplicity_f1", 0.0)))
                        relation_label_gain = max(0.0, float(normalized["relation_label_f1"]) - float(baseline.get("relation_label_f1", 0.0)))
                        relation_bonus = 0.0
                        if class_drop <= 0.0 and attr_drop <= 0.0:
                            relation_bonus = min(0.06, 0.05 * relation_pair_gain + 0.03 * multiplicity_gain + 0.01 * relation_label_gain)
                        score = current_content_score + 0.75 * delta_content + relation_bonus - 2.0 * class_drop - 1.2 * attr_drop
                    else:
                        score = current_content_score
                    if not is_plantuml_block:
                        score = min(score, 0.05)
                    if float(normalized["class_f1"]) < 0.50:
                        score -= 0.25
                    if float(normalized["attribute_f1"]) < 0.30:
                        score -= 0.10
                else:
                    score = float(normalized["total"])
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
    if 'reward_mode == "content_guarded_delta_v2"' in text:
        print("content_guarded_delta_v2 reward mode already exists.")
        return
    if V1_REWARD_BLOCK in text:
        text = text.replace(V1_REWARD_BLOCK, V2_REWARD_BLOCK, 1)
    elif V0_REWARD_BLOCK in text:
        text = text.replace(V0_REWARD_BLOCK, V2_REWARD_BLOCK, 1)
    else:
        raise SystemExit("Could not find known reward block to patch.")
    TRAINER.write_text(text, encoding="utf-8")
    print("Patched content_guarded_delta_v2 reward mode.")


def make_v2_yaml() -> None:
    source = TINY64_YAML if TINY64_YAML.exists() else V1_YAML if V1_YAML.exists() else V0_YAML
    if not source.exists():
        raise SystemExit("Could not find v0/v1 PPO yaml.")
    text = source.read_text(encoding="utf-8")
    text = text.replace("rema_plantuml_mappo_tiny32", "rema_plantuml_mappo_tiny64")
    text = text.replace("mappo_v0_tiny32", "mappo_v2_tiny64")
    text = text.replace("mappo_v1_tiny32", "mappo_v2_tiny64")
    text = text.replace("mappo_v1_tiny64", "mappo_v2_tiny64")
    text = text.replace("max_samples: 32", "max_samples: 64")
    text = text.replace("learning_rate: 5.0e-6", "learning_rate: 8.0e-7")
    text = text.replace("learning_rate: 1.0e-6", "learning_rate: 8.0e-7")
    text = text.replace("max_steps: 10", "max_steps: 20")
    text = text.replace("max_steps: 20", "max_steps: 20")
    if "save_steps: 1000" not in text:
        import re

        text = re.sub(r"^save_steps:.*$", "save_steps: 1000", text, flags=re.MULTILINE)
    V2_YAML.write_text(text, encoding="utf-8")
    print(f"Wrote {V2_YAML}")


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

    (scripts / "09_build_mappo_v2_tiny64_baseline.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
ROOT=/hy-tmp/rema_mappo_v0_tiny_20260615
BASE=/hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
ADAPTER="$ROOT/adapters/full_sft_v4"
OUT_DIR="$ROOT/outputs/mappo_v2_tiny64_baseline"
mkdir -p "$OUT_DIR"

python - <<'PY'
import json, re
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
    sample_id = item.get("id") or re.search(r"Sample ID:\\s*([^\\n]+)", item.get("instruction", "")).group(1).strip()
    reward = reward_rows[sample_id]
    rows.append({"id": sample_id, "HumanLang": reward["requirement"], "PlantUML": reward["gold_plantuml"]})

out = root / "outputs/mappo_v2_tiny64_baseline/baseline_data_tiny64.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote", out, "count", len(rows))
PY

python "$ROOT/evaluation/batch_infer_preserve_id.py" \
  --base-model "$BASE" \
  --adapter "$ADAPTER" \
  --data "$OUT_DIR/baseline_data_tiny64.json" \
  --output "$OUT_DIR/full_sft_baseline_predictions.jsonl"

python "$ROOT/evaluation/evaluate_predictions_normalized.py" \
  --input "$OUT_DIR/full_sft_baseline_predictions.jsonl" \
  --output "$OUT_DIR/full_sft_baseline_eval.jsonl" \
  --summary "$OUT_DIR/full_sft_baseline_summary.json" \
  --csv "$OUT_DIR/full_sft_baseline_review.csv"

python - <<'PY'
import json
from pathlib import Path

root = Path("/hy-tmp/rema_mappo_v0_tiny_20260615")
src = root / "outputs/mappo_v2_tiny64_baseline/full_sft_baseline_eval.jsonl"
dst = root / "LLaMA-Factory/data/rema_plantuml_mappo_baseline_tiny64.jsonl"
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

cat "$OUT_DIR/full_sft_baseline_summary.json"
""",
        encoding="utf-8",
    )

    (scripts / "10_run_mappo_v2_tiny64.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
cd /hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory
export REMA_PLANTUML_PPO=1
export REMA_FUSION_DIR=/hy-tmp/rema_mappo_v0_tiny_20260615/fusion
export REMA_PLANTUML_REWARD_MAP=/hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory/data/rema_plantuml_mappo_reward_map.jsonl
export REMA_PLANTUML_BASELINE_MAP=/hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory/data/rema_plantuml_mappo_baseline_tiny64.jsonl
export REMA_PLANTUML_REWARD_MODE=content_guarded_delta_v2
export REMA_PLANTUML_MAX_NEW_TOKENS=512
llamafactory-cli train examples/train_lora/rema_plantuml_mappo_v2_tiny64.yaml
""",
        encoding="utf-8",
    )

    (scripts / "11_eval_mappo_v2_tiny64.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
ROOT=/hy-tmp/rema_mappo_v0_tiny_20260615
BASE=/hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
ADAPTER="$ROOT/LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/mappo_v2_tiny64"
DATA="$ROOT/LLaMA-Factory/data/plantucd_test_142.json"
OUT_DIR="$ROOT/outputs/mappo_v2_tiny64"
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

    (scripts / "12_pairwise_mappo_v2_tiny64.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import json, csv
from pathlib import Path

base = Path("/hy-tmp/rema_full_sft_4090_20260614/outputs/full_sft_v4_test142_eval.jsonl")
new = Path("/hy-tmp/rema_mappo_v0_tiny_20260615/outputs/mappo_v2_tiny64/test142_eval.jsonl")
out = Path("/hy-tmp/rema_mappo_v2_tiny64_vs_full_sft_pairwise.csv")
summary = Path("/hy-tmp/rema_mappo_v2_tiny64_vs_full_sft_pairwise_summary.json")

def load(p):
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return rows

b, n = load(base), load(new)
metrics = ["total", "class_f1", "attribute_f1", "method_f1", "relation_pair_f1", "relation_label_f1", "multiplicity_f1"]
rows = []
for k in sorted(set(b) & set(n)):
    br, nr = b[k]["normalized_reward"], n[k]["normalized_reward"]
    row = {
        "id": k,
        "base_total": br["total"],
        "mappo_total": nr["total"],
        "delta_total": nr["total"] - br["total"],
        "requirement": n[k].get("requirement", ""),
        "base_prediction": b[k].get("prediction", ""),
        "mappo_prediction": n[k].get("prediction", ""),
        "gold": n[k].get("gold_plantuml", ""),
    }
    for m in metrics[1:]:
        row[f"delta_{m}"] = nr[m] - br[m]
    rows.append(row)

wins = sum(r["delta_total"] > 1e-6 for r in rows)
losses = sum(r["delta_total"] < -1e-6 for r in rows)
ties = len(rows) - wins - losses
avg_delta = {"total": sum(r["delta_total"] for r in rows) / len(rows)}
for m in metrics[1:]:
    avg_delta[m] = sum(r[f"delta_{m}"] for r in rows) / len(rows)

summary_obj = {
    "baseline": str(base),
    "mappo": str(new),
    "count": len(rows),
    "wins": wins,
    "losses": losses,
    "ties": ties,
    "avg_delta": avg_delta,
    "top_improved_ids": [r["id"] for r in sorted(rows, key=lambda x: x["delta_total"], reverse=True)[:10]],
    "top_degraded_ids": [r["id"] for r in sorted(rows, key=lambda x: x["delta_total"])[:10]],
}

with out.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda x: x["delta_total"]))

summary.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary_obj, ensure_ascii=False, indent=2))
PY
""",
        encoding="utf-8",
    )

    (scripts / "13_pack_mappo_v2_tiny64.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
tar -czf /hy-tmp/rema_mappo_v2_tiny64_results_$(date +%Y%m%d_%H%M%S).tar.gz \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/mappo_v2_tiny64 \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 outputs/mappo_v2_tiny64 \
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 outputs/mappo_v2_tiny64_baseline \
  -C /hy-tmp rema_mappo_v2_tiny64.log \
  -C /hy-tmp rema_mappo_v2_tiny64_eval.log \
  -C /hy-tmp rema_mappo_v2_tiny64_baseline.log \
  -C /hy-tmp rema_mappo_v2_tiny64_pairwise.log \
  -C /hy-tmp rema_mappo_v2_tiny64_vs_full_sft_pairwise.csv \
  -C /hy-tmp rema_mappo_v2_tiny64_vs_full_sft_pairwise_summary.json
ls -lh /hy-tmp/rema_mappo_v2_tiny64_results_*.tar.gz
""",
        encoding="utf-8",
    )

    for script in scripts.glob("*mappo_v2_tiny64*.sh"):
        script.chmod(0o755)
    print("Wrote v2 helper scripts.")


def main() -> None:
    if not TRAINER.exists():
        raise SystemExit(f"Trainer not found: {TRAINER}")
    ensure_v0_patch()
    patch_baseline_loader()
    patch_reward_mode()
    make_v2_yaml()
    write_helper_scripts()
    print("MAPPO-v2 content-guarded delta patch is ready.")


if __name__ == "__main__":
    main()
