"""Deterministic, value-free cross-check reminders for the self-review pass.

Two knowledge-free signals computed from the case's own artifact and the
model's own first patch (gold is never consulted):

  - mirrored blocks (clone detection): containers with at least two leaves
    whose canonical serialization is identical to another container's — the
    materialized-copy pattern, where one block is a copy of a variant selected
    from an in-document rule table or a duplicated record;
  - repeated-edit residue (systematic-edit location, after LASE): abstract
    each patch op path to a template by replacing numeric segments and "-"
    with "*"; a template instantiated by two or more ops is a systematic
    edit, and artifact locations that match it but are left unmodified by the
    patch are listed for re-examination.

Both signals emit PATHS ONLY — never values and never suggested content — so
the reminder reorganizes information the reviser already has. All transforms
are within-case and knowledge-free (index abstraction, value equality,
repetition counts); nothing is mined across cases.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

_MAX_LINES_PER_SECTION = 8
_MAX_PATHS_PER_PATTERN = 6

_REMINDER_INTRO = """\
# Mechanical cross-check signals (auxiliary)
The locations below were computed mechanically from the artifact and your patch.
They are re-examination hints, not change requests: for each location, decide
from the user feedback and the artifact whether it is forced to change; if it
is not, leave it unchanged."""

_MIRROR_HEADER = """\
## Mirrored blocks
Blocks whose current content is identical to each other; if the edit changes
which variant applies to one of them, the copy must follow:"""

_PATTERN_HEADER = """\
## Unmodified locations matching a repeated edit pattern
Your patch applies operations of the same shape at two or more locations; the
locations below match the same shape but are not modified by your patch:"""


def _esc(k: Any) -> str:
    return str(k).replace("~", "~0").replace("/", "~1")


def _walk(doc: Any, path: str = ""):
    if isinstance(doc, dict):
        yield path, doc
        for k, v in doc.items():
            yield from _walk(v, f"{path}/{_esc(k)}")
    elif isinstance(doc, list):
        yield path, doc
        for i, v in enumerate(doc):
            yield from _walk(v, f"{path}/{i}")
    else:
        yield path, doc


def _leaves(doc: Any) -> dict[str, str]:
    return {p: json.dumps(v, sort_keys=True)
            for p, v in _walk(doc) if not isinstance(v, (dict, list))}


def _mirror_groups(doc: Any) -> list[list[str]]:
    idx: dict[str, list[str]] = defaultdict(list)
    for p, v in _walk(doc):
        if not p or not isinstance(v, (dict, list)):
            continue
        if len(_leaves(v)) < 2:
            continue
        idx[json.dumps(v, sort_keys=True)].append(p)
    return [sorted(ps) for ps in idx.values() if len(ps) >= 2]


def _template(path: str) -> str:
    return "/".join("*" if seg.isdigit() or seg == "-" else seg
                    for seg in path.split("/"))


def _matches(tpl: str, path: str) -> bool:
    a, b = tpl.split("/"), path.split("/")
    return len(a) == len(b) and all(x == "*" or x == y for x, y in zip(a, b))


def build_review_reminders(current: Any, model_patch: Any, applied: Any) -> str:
    """Return the reminder section text, or "" when no signal fires.

    `applied` is the artifact after the model's own patch (None when the patch
    did not apply); the touched set is the net-effect leaf diff between
    `current` and `applied`, so sequential-index drift in op paths is moot.
    """
    try:
        touched: set[str] | None = None
        if applied is not None:
            cur_leaves, ap_leaves = _leaves(current), _leaves(applied)
            touched = {p for p in set(cur_leaves) | set(ap_leaves)
                       if cur_leaves.get(p) != ap_leaves.get(p)}

        def untouched(path: str) -> bool:
            if touched is None:
                return True
            if path in touched:
                return False
            pref = path + "/"
            return not any(t.startswith(pref) for t in touched)

        mirror_lines = ["- " + " == ".join(ps) for ps in _mirror_groups(current)
                        if any(untouched(p) for p in ps)]

        pattern_lines: list[str] = []
        if touched is not None and isinstance(model_patch, list):
            counts: dict[str, int] = defaultdict(int)
            for op in model_patch:
                if isinstance(op, dict) and isinstance(op.get("path"), str):
                    tpl = _template(op["path"])
                    if "*" in tpl:
                        counts[tpl] += 1
            doc_paths = [p for p, _ in _walk(current) if p]
            for tpl, n in sorted(counts.items()):
                if n < 2:
                    continue
                mates = [p for p in doc_paths if _matches(tpl, p) and untouched(p)]
                if not mates:
                    continue
                shown = mates[:_MAX_PATHS_PER_PATTERN]
                extra = len(mates) - len(shown)
                suffix = f" (+{extra} more)" if extra else ""
                pattern_lines.append(
                    f"- pattern {tpl} (applied {n} times in your patch): "
                    + ", ".join(shown) + suffix)

        parts = []
        if mirror_lines:
            parts.append(_MIRROR_HEADER + "\n" + "\n".join(mirror_lines[:_MAX_LINES_PER_SECTION]))
        if pattern_lines:
            parts.append(_PATTERN_HEADER + "\n" + "\n".join(pattern_lines[:_MAX_LINES_PER_SECTION]))
        if not parts:
            return ""
        return _REMINDER_INTRO + "\n\n" + "\n\n".join(parts)
    except Exception:  # noqa: BLE001 — reminders are best-effort; never kill the run
        return ""
