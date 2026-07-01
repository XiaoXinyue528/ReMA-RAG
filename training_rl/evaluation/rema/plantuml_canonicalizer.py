from __future__ import annotations

import re


TYPE_FIRST_RE = re.compile(
    r"^(?P<vis>[+\-#~])?\s*(?P<type>[A-Za-z_][A-Za-z0-9_<>\[\],.? ]*)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$"
)
METHOD_TYPE_FIRST_RE = re.compile(
    r"^(?P<vis>[+\-#~])?\s*(?P<ret>[A-Za-z_][A-Za-z0-9_<>\[\],.? ]*)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)\s*$"
)
METHOD_NO_RETURN_RE = re.compile(
    r"^(?P<vis>[+\-#~])?\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)\s*$"
)
NAME_COLON_RE = re.compile(r"^(?P<vis>[+\-#~])?\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<type>.+?)\s*$")
ARG_TYPE_FIRST_RE = re.compile(r"^(?P<type>[A-Za-z_][A-Za-z0-9_<>\[\],.?]*)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")

KNOWN_TYPE_PREFIXES = {
    "str",
    "string",
    "int",
    "integer",
    "float",
    "double",
    "bool",
    "boolean",
    "date",
    "time",
    "datetime",
    "void",
    "list",
    "set",
    "map",
    "dict",
    "array",
    "long",
    "short",
    "char",
    "user",
}


def canonicalize_plantuml(text: str) -> str:
    """Normalize common PlantUML class-member styles to `name: Type`.

    This is intentionally conservative: it only rewrites member lines inside
    class bodies and leaves relations, class names, and diagram wrappers alone.
    """
    lines = (text or "").splitlines()
    out: list[str] = []
    in_class = False

    for raw in lines:
        stripped = raw.strip()
        if re.match(r"^(?:abstract\s+)?(?:class|interface|enum)\s+\w+", stripped):
            in_class = True
            out.append(raw)
            continue
        if in_class and stripped == "}":
            in_class = False
            out.append(raw)
            continue
        if in_class:
            out.append(_canonicalize_member_line(raw))
        else:
            out.append(raw)
    return "\n".join(out)


def _canonicalize_member_line(raw: str) -> str:
    indent = raw[: len(raw) - len(raw.lstrip())]
    line = raw.strip()
    if not line or line.startswith("'") or line.startswith("//"):
        return raw

    converted = _canonicalize_method(line)
    if converted is not None:
        return indent + converted

    converted = _canonicalize_attribute(line)
    if converted is not None:
        return indent + converted

    return raw


def _canonicalize_attribute(line: str) -> str | None:
    if NAME_COLON_RE.match(line):
        return _normalize_spacing(line)

    match = TYPE_FIRST_RE.match(line)
    if not match:
        return None
    type_name = match.group("type").strip()
    attr_name = match.group("name").strip()
    if not _looks_like_type(type_name):
        return None
    return f"{match.group('vis') or ''}{attr_name}: {type_name}"


def _canonicalize_method(line: str) -> str | None:
    match = METHOD_TYPE_FIRST_RE.match(line)
    if match and _looks_like_type(match.group("ret").strip()):
        args = _canonicalize_args(match.group("args"))
        return f"{match.group('vis') or ''}{match.group('name')}({args}): {match.group('ret').strip()}"

    match = METHOD_NO_RETURN_RE.match(line)
    if match:
        args = _canonicalize_args(match.group("args"))
        return f"{match.group('vis') or ''}{match.group('name')}({args})"
    return None


def _canonicalize_args(args: str) -> str:
    if not args.strip():
        return ""
    converted = []
    for arg in args.split(","):
        item = arg.strip()
        if not item:
            continue
        if ":" in item:
            name, type_name = item.split(":", 1)
            converted.append(f"{name.strip()}: {type_name.strip()}")
            continue
        match = ARG_TYPE_FIRST_RE.match(item)
        if match and _looks_like_type(match.group("type")):
            converted.append(f"{match.group('name')}: {match.group('type')}")
        else:
            converted.append(item)
    return ", ".join(converted)


def _looks_like_type(value: str) -> bool:
    compact = value.strip()
    if not compact:
        return False
    head = re.split(r"[^A-Za-z0-9_]", compact, 1)[0].lower()
    if head in KNOWN_TYPE_PREFIXES:
        return True
    return bool(re.match(r"^[A-Z][A-Za-z0-9_]*(?:<.*>)?$", compact))


def _normalize_spacing(line: str) -> str:
    match = NAME_COLON_RE.match(line)
    if not match:
        return line
    return f"{match.group('vis') or ''}{match.group('name')}: {match.group('type').strip()}"
