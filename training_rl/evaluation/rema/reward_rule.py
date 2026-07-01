from __future__ import annotations

from dataclasses import asdict, dataclass

from .metrics import exact_match_normalized, prf
from .plantuml_parser import parse_plantuml


DEFAULT_WEIGHTS = {
    "format": 0.05,
    "syntax": 0.05,
    "class_f1": 0.18,
    "attribute_f1": 0.18,
    "method_f1": 0.07,
    "relation_pair_f1": 0.24,
    "relation_label_f1": 0.12,
    "multiplicity_f1": 0.10,
}


@dataclass
class RuleReward:
    total: float
    exact_match: float
    format_score: float
    syntax_score: float
    class_f1: float
    attribute_f1: float
    method_f1: float
    relation_pair_f1: float
    relation_label_f1: float
    multiplicity_f1: float
    extra_class_count: int
    extra_relation_count: int
    missing_relation_count: int
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_rule_reward(prediction: str, reference: str, weights: dict[str, float] | None = None) -> RuleReward:
    weights = weights or DEFAULT_WEIGHTS
    pred_text = prediction or ""
    ref_text = reference or ""
    format_score = 1.0 if starts_and_ends_plantuml(pred_text) else 0.7 if pred_text.strip() else 0.0
    syntax_score = 1.0
    notes = []

    try:
        pred = parse_plantuml(pred_text)
    except Exception as exc:
        pred = parse_plantuml("")
        syntax_score = 0.0
        notes.append(f"prediction_parse_error: {exc}")

    gold = parse_plantuml(ref_text)
    class_score = prf({c.lower() for c in pred.classes}, {c.lower() for c in gold.classes})
    attr_score = prf(pred.attributes, gold.attributes)
    method_score = prf(pred.methods, gold.methods)
    relation_pair_score = prf([r.pair_key for r in pred.relations], [r.pair_key for r in gold.relations])
    relation_label_score = prf([r.label_key for r in pred.relations if r.label], [r.label_key for r in gold.relations if r.label])
    mult_score = prf([r.multiplicity_key for r in pred.relations if r.src_mult or r.dst_mult], [r.multiplicity_key for r in gold.relations if r.src_mult or r.dst_mult])

    pred_classes = {c.lower() for c in pred.classes}
    gold_classes = {c.lower() for c in gold.classes}
    extra_class_count = len(pred_classes - gold_classes)
    extra_relation_count = max(0, len(pred.relations) - len(gold.relations))
    missing_relation_count = max(0, len(gold.relations) - relation_pair_score.matched)

    total = (
        weights["format"] * format_score
        + weights["syntax"] * syntax_score
        + weights["class_f1"] * class_score.f1
        + weights["attribute_f1"] * attr_score.f1
        + weights["method_f1"] * method_score.f1
        + weights["relation_pair_f1"] * relation_pair_score.f1
        + weights["relation_label_f1"] * relation_label_score.f1
        + weights["multiplicity_f1"] * mult_score.f1
    )
    total -= min(0.25, 0.04 * extra_class_count)
    total -= min(0.25, 0.05 * extra_relation_count)

    if len(gold_classes) <= 2 and len(pred_classes) >= len(gold_classes) + 4:
        notes.append("severe_over_generation")
        total -= 0.20

    if class_score.f1 < 0.3 and gold_classes:
        notes.append("missing_core_classes")
        total *= 0.5

    if missing_relation_count and class_score.f1 >= 0.5:
        notes.append("missing_relations_after_classes_found")
        total -= min(0.15, 0.05 * missing_relation_count)
    elif missing_relation_count and class_score.f1 >= 0.8:
        notes.append("high_class_coverage_but_missing_relations")
        total -= min(0.10, 0.03 * missing_relation_count)

    if relation_pair_score.f1 >= 0.5 and relation_label_score.f1 < 0.4:
        notes.append("relation_labels_weak")

    if relation_pair_score.f1 >= 0.5 and mult_score.f1 < 0.4:
        notes.append("multiplicity_weak")

    total = max(0.0, min(1.0, total))
    return RuleReward(
        total=round(total, 6),
        exact_match=exact_match_normalized(pred_text, ref_text),
        format_score=format_score,
        syntax_score=syntax_score,
        class_f1=class_score.f1,
        attribute_f1=attr_score.f1,
        method_f1=method_score.f1,
        relation_pair_f1=relation_pair_score.f1,
        relation_label_f1=relation_label_score.f1,
        multiplicity_f1=mult_score.f1,
        extra_class_count=extra_class_count,
        extra_relation_count=extra_relation_count,
        missing_relation_count=missing_relation_count,
        notes=notes,
    )


def starts_and_ends_plantuml(text: str) -> bool:
    stripped = (text or "").strip().lower()
    return stripped.startswith("@startuml") and stripped.endswith("@enduml")


