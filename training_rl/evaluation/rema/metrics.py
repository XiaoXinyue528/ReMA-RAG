from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Hashable


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    matched: int
    predicted: int
    gold: int


def prf(predicted: Collection[Hashable], gold: Collection[Hashable]) -> PRF:
    pred_set = set(predicted)
    gold_set = set(gold)
    matched = len(pred_set & gold_set)
    precision = matched / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
    recall = matched / len(gold_set) if gold_set else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1, matched, len(pred_set), len(gold_set))


def exact_match_normalized(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def normalize_text(text: str) -> str:
    return "".join((text or "").lower().split())

