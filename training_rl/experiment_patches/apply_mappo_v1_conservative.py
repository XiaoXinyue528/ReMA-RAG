from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ.get("REMA_ROOT", "/hy-tmp/rema_mappo_v0_tiny_20260615"))
TRAINER = ROOT / "LLaMA-Factory" / "src" / "llamafactory" / "train" / "ppo" / "trainer_qr_s_g.py"
V0_PATCH = ROOT / "patches" / "apply_plantuml_ppo_v0.py"
V0_YAML = ROOT / "LLaMA-Factory" / "examples" / "train_lora" / "rema_plantuml_mappo_v0_tiny32.yaml"
V1_YAML = ROOT / "LLaMA-Factory" / "examples" / "train_lora" / "rema_plantuml_mappo_v1_tiny32.yaml"


OLD_REWARD_BLOCK = """                if reward_mode == "relation_focus":
                    score = (
                        0.65 * float(normalized["total"])
                        + 0.20 * float(normalized["relation_pair_f1"])
                        + 0.10 * float(normalized["multiplicity_f1"])
                        + 0.05 * float(normalized["relation_label_f1"])
                    )
                else:
                    score = float(normalized["total"])
"""

NEW_REWARD_BLOCK = """                if reward_mode == "relation_focus":
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


def ensure_v0_patch() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    if "def plantuml_ppo_train" in text and "REMA_PLANTUML_PPO" in text:
        return
    if not V0_PATCH.exists():
        raise SystemExit(f"PlantUML PPO-v0 patch is missing: {V0_PATCH}")
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(V0_PATCH.read_text(encoding="utf-8"), namespace)


def patch_reward_mode() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    if 'reward_mode == "conservative_v1"' in text:
        print("conservative_v1 reward mode already exists.")
        return
    if OLD_REWARD_BLOCK not in text:
        raise SystemExit("Could not find relation_focus reward block to patch.")
    text = text.replace(OLD_REWARD_BLOCK, NEW_REWARD_BLOCK, 1)
    TRAINER.write_text(text, encoding="utf-8")
    print(f"Patched conservative_v1 reward mode into {TRAINER}")


def make_v1_yaml() -> None:
    if not V0_YAML.exists():
        raise SystemExit(f"Missing v0 yaml: {V0_YAML}")
    text = V0_YAML.read_text(encoding="utf-8")
    replacements = {
        "output_dir: saves/llama3-8b/lora/rema_plantuml/mappo_v0_tiny32": "output_dir: saves/llama3-8b/lora/rema_plantuml/mappo_v1_tiny32",
        "learning_rate: 5.0e-6": "learning_rate: 1.0e-6",
        "max_steps: 20": "max_steps: 10",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if "save_steps: 1000" not in text:
        import re

        text = re.sub(r"^save_steps:.*$", "save_steps: 1000", text, flags=re.MULTILINE)
    V1_YAML.write_text(text, encoding="utf-8")
    print(f"Wrote {V1_YAML}")


def write_scripts() -> None:
    scripts = ROOT / "cloud_scripts"
    scripts.mkdir(parents=True, exist_ok=True)

    (scripts / "06_run_mappo_v1_tiny32.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
cd /hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory
export REMA_PLANTUML_PPO=1
export REMA_FUSION_DIR=/hy-tmp/rema_mappo_v0_tiny_20260615/fusion
export REMA_PLANTUML_REWARD_MAP=/hy-tmp/rema_mappo_v0_tiny_20260615/LLaMA-Factory/data/rema_plantuml_mappo_reward_map.jsonl
export REMA_PLANTUML_REWARD_MODE=conservative_v1
export REMA_PLANTUML_MAX_NEW_TOKENS=512
llamafactory-cli train examples/train_lora/rema_plantuml_mappo_v1_tiny32.yaml
""",
        encoding="utf-8",
    )

    (scripts / "07_eval_mappo_v1_tiny32.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rema-mappo
ROOT=/hy-tmp/rema_mappo_v0_tiny_20260615
BASE=/hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
ADAPTER="$ROOT/LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/mappo_v1_tiny32"
DATA="$ROOT/LLaMA-Factory/data/plantucd_test_142.json"
OUT_DIR="$ROOT/outputs/mappo_v1_tiny32"
mkdir -p "$OUT_DIR"
python "$ROOT/evaluation/batch_infer.py" \\
  --base-model "$BASE" \\
  --adapter "$ADAPTER" \\
  --data "$DATA" \\
  --output "$OUT_DIR/test142_predictions.jsonl"
python "$ROOT/evaluation/evaluate_predictions_normalized.py" \\
  --input "$OUT_DIR/test142_predictions.jsonl" \\
  --output "$OUT_DIR/test142_eval.jsonl" \\
  --summary "$OUT_DIR/test142_summary.json" \\
  --csv "$OUT_DIR/test142_review.csv"
cat "$OUT_DIR/test142_summary.json"
""",
        encoding="utf-8",
    )

    (scripts / "08_pack_mappo_v1_tiny32.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
tar -czf /hy-tmp/rema_mappo_v1_tiny32_results_$(date +%Y%m%d_%H%M%S).tar.gz \\
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/mappo_v1_tiny32 \\
  -C /hy-tmp/rema_mappo_v0_tiny_20260615 outputs/mappo_v1_tiny32 \\
  -C /hy-tmp rema_mappo_v1_tiny32.log \\
  -C /hy-tmp rema_mappo_v1_tiny32_eval.log
ls -lh /hy-tmp/rema_mappo_v1_tiny32_results_*.tar.gz
""",
        encoding="utf-8",
    )

    for script in scripts.glob("0*_mappo_v1_tiny32.sh"):
        script.chmod(0o755)
    print(f"Wrote v1 cloud scripts in {scripts}")


def main() -> None:
    if not TRAINER.exists():
        raise SystemExit(f"Trainer not found: {TRAINER}")
    ensure_v0_patch()
    patch_reward_mode()
    make_v1_yaml()
    write_scripts()
    print("MAPPO-v1 conservative patch is ready.")


if __name__ == "__main__":
    main()
