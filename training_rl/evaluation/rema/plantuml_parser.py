from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


CLASS_RE = re.compile(r"\b(?:abstract\s+)?(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\{(?P<body>.*?)\})?", re.S)
REL_RE = re.compile(
    r"(?P<src>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:(?P<src_mult>\"[^\"]+\")\s*)?"
    r"(?P<arrow><\|--|\*--|o--|-->|--|<--|\.\.>|<\.\.)\s*"
    r"(?:(?P<dst_mult>\"[^\"]+\")\s*)?"
    r"(?P<dst>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s*:\s*(?P<label>.+?))?\s*$"
)


@dataclass(frozen=True)
class UmlRelation:
    src: str
    dst: str
    arrow: str
    src_mult: str = ""
    dst_mult: str = ""
    label: str = ""

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.src.lower(), self.dst.lower())

    @property
    def label_key(self) -> tuple[str, str, str]:
        return (self.src.lower(), self.dst.lower(), normalize_name(self.label))

    @property
    def multiplicity_key(self) -> tuple[str, str, str, str]:
        return (self.src.lower(), self.dst.lower(), self.src_mult, self.dst_mult)


@dataclass
class UmlUnit:
    classes: set[str] = field(default_factory=set)
    attributes: set[tuple[str, str]] = field(default_factory=set)
    methods: set[tuple[str, str]] = field(default_factory=set)
    relations: list[UmlRelation] = field(default_factory=list)


def normalize_name(value: str) -> str:
    value = value.strip().strip('"').strip("'")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def strip_wrappers(text: str) -> str:
    text = text or ""
    text = re.sub(r"```(?:plantuml)?", "", text, flags=re.I)
    text = text.replace("```", "")
    text = re.sub(r"@startuml", "", text, flags=re.I)
    text = re.sub(r"@enduml", "", text, flags=re.I)
    return text.strip()


def parse_member(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("'") or line.startswith("//"):
        return None
    line = re.sub(r"^[+\-#~]\s*", "", line)
    name = line.split(":", 1)[0].strip()
    if not name:
        return None
    kind = "method" if "(" in name or ")" in name else "attribute"
    name = name.split("(", 1)[0].strip()
    return kind, normalize_name(name)


def parse_plantuml(text: str) -> UmlUnit:
    body = strip_wrappers(text)
    unit = UmlUnit()
    occupied_spans = []

    for match in CLASS_RE.finditer(body):
        class_name = match.group(1)
        unit.classes.add(class_name)
        occupied_spans.append(match.span())
        for raw_line in (match.group("body") or "").splitlines():
            parsed = parse_member(raw_line)
            if not parsed:
                continue
            kind, member_name = parsed
            if kind == "method":
                unit.methods.add((normalize_name(class_name), member_name))
            else:
                unit.attributes.add((normalize_name(class_name), member_name))

    relation_text = remove_spans(body, occupied_spans)
    for raw_line in relation_text.splitlines():
        line = raw_line.strip().rstrip(";")
        if not line or line.startswith("@"):
            continue
        match = REL_RE.match(line)
        if not match:
            continue
        relation = UmlRelation(
            src=match.group("src"),
            dst=match.group("dst"),
            arrow=match.group("arrow"),
            src_mult=(match.group("src_mult") or "").strip('"'),
            dst_mult=(match.group("dst_mult") or "").strip('"'),
            label=(match.group("label") or "").strip().strip('"'),
        )
        unit.relations.append(relation)
        unit.classes.add(relation.src)
        unit.classes.add(relation.dst)
    return unit


def remove_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for idx in range(start, end):
            chars[idx] = "\n"
    return "".join(chars)

