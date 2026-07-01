from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rema.plantuml_canonicalizer import canonicalize_plantuml
from rema.reward_rule import compute_rule_reward


METRIC_KEYS = [
    "total",
    "format_score",
    "syntax_score",
    "class_f1",
    "attribute_f1",
    "method_f1",
    "relation_pair_f1",
    "relation_label_f1",
    "multiplicity_f1",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def average(rows: list[dict], block: str) -> dict[str, float]:
    return {
        key: (mean(float(row[block][key]) for row in rows) if rows else 0.0)
        for key in METRIC_KEYS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strict and canonicalized PlantUML rewards.")
    parser.add_argument("--input", required=True, help="Prediction JSONL with prediction and gold_plantuml fields.")
    parser.add_argument("--output", required=True, help="Output JSONL containing strict and normalized rewards.")
    parser.add_argument("--summary", required=True, help="Output summary JSON.")
    parser.add_argument("--csv", required=True, help="Output review CSV.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    csv_path = Path(args.csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for row in load_jsonl(input_path):
        prediction = row.get("prediction", "")
        gold = row.get("gold_plantuml", "")
        canonical_prediction = canonicalize_plantuml(prediction)
        canonical_gold = canonicalize_plantuml(gold)

        strict = compute_rule_reward(prediction=prediction, reference=gold).to_dict()
        normalized = compute_rule_reward(prediction=canonical_prediction, reference=canonical_gold).to_dict()
        delta = {
            key: round(float(normalized[key]) - float(strict[key]), 6)
            for key in METRIC_KEYS
        }

        out_rows.append(
            {
                "id": row.get("id"),
                "requirement": row.get("requirement", ""),
                "gold_plantuml": gold,
                "prediction": prediction,
                "canonical_gold": canonical_gold,
                "canonical_prediction": canonical_prediction,
                "strict_reward": strict,
                "normalized_reward": normalized,
                "delta": delta,
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "count": len(out_rows),
        "strict_mean": average(out_rows, "strict_reward"),
        "normalized_mean": average(out_rows, "normalized_reward"),
    }
    summary["delta_mean"] = {
        key: round(summary["normalized_mean"][key] - summary["strict_mean"][key], 6)
        for key in METRIC_KEYS
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "id",
            "strict_total",
            "normalized_total",
            "delta_total",
            "strict_attribute_f1",
            "normalized_attribute_f1",
            "delta_attribute_f1",
            "strict_method_f1",
            "normalized_method_f1",
            "delta_method_f1",
            "strict_relation_pair_f1",
            "normalized_relation_pair_f1",
            "strict_relation_label_f1",
            "normalized_relation_label_f1",
            "requirement",
            "gold_plantuml",
            "prediction",
            "canonical_prediction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(out_rows, key=lambda item: item["delta"]["total"], reverse=True):
            strict = row["strict_reward"]
            norm = row["normalized_reward"]
            delta = row["delta"]
            writer.writerow(
                {
                    "id": row["id"],
                    "strict_total": strict["total"],
                    "normalized_total": norm["total"],
                    "delta_total": delta["total"],
                    "strict_attribute_f1": strict["attribute_f1"],
                    "normalized_attribute_f1": norm["attribute_f1"],
                    "delta_attribute_f1": delta["attribute_f1"],
                    "strict_method_f1": strict["method_f1"],
                    "normalized_method_f1": norm["method_f1"],
                    "delta_method_f1": delta["method_f1"],
                    "strict_relation_pair_f1": strict["relation_pair_f1"],
                    "normalized_relation_pair_f1": norm["relation_pair_f1"],
                    "strict_relation_label_f1": strict["relation_label_f1"],
                    "normalized_relation_label_f1": norm["relation_label_f1"],
                    "requirement": row["requirement"],
                    "gold_plantuml": row["gold_plantuml"],
                    "prediction": row["prediction"],
                    "canonical_prediction": row["canonical_prediction"],
                }
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
