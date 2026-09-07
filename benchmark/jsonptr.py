"""Pure-stdlib JSON Pointer (RFC 6901) + JSON Patch (RFC 6902) helpers.

Shared by the CLI annotator and the GUI server. Deliberately dependency-free
(no pydantic) so the GUI backend can run on a bare python3 with no install.

`apply_patch` accepts patch ops as plain dicts ({"op","path","value"}) OR any
object exposing `.op` / `.path` / `.value` attributes (e.g. a pydantic
PatchOp), so both callers can reuse it.
"""

from __future__ import annotations

import copy
from typing import Any


def deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge ``incoming`` into ``base`` in place and return ``base``.

    Nested objects merge recursively; arrays and scalars replace wholesale.
    This lets a generation turn either introduce new keys, re-emit a collection
    (array) with its full extended contents, or re-emit an object with only the
    fields it adds that turn — without an object's earlier fields being lost.
    For keys that never repeat across turns this is identical to ``dict.update``.
    """
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def split_pointer(path: str) -> list[str]:
    if path == "":
        return []
    if not path.startswith("/"):
        raise ValueError(f"JSON Pointer must start with '/': {path!r}")
    return [seg.replace("~1", "/").replace("~0", "~") for seg in path[1:].split("/")]


def pointer_get(obj: Any, path: str, default: Any = ...) -> Any:
    cur = obj
    for seg in split_pointer(path):
        if isinstance(cur, list):
            try:
                idx = int(seg)
            except ValueError:
                if default is ...:
                    raise KeyError(f"list index expected, got {seg!r}")
                return default
            if idx < 0 or idx >= len(cur):
                if default is ...:
                    raise KeyError(f"list index out of range: {idx}")
                return default
            cur = cur[idx]
        elif isinstance(cur, dict):
            if seg not in cur:
                if default is ...:
                    raise KeyError(seg)
                return default
            cur = cur[seg]
        else:
            if default is ...:
                raise KeyError(f"cannot descend into {type(cur).__name__} at {seg!r}")
            return default
    return cur


def pointer_set(obj: Any, path: str, value: Any) -> None:
    segs = split_pointer(path)
    if not segs:
        raise ValueError("cannot set root via pointer_set")
    cur = obj
    for seg in segs[:-1]:
        if isinstance(cur, list):
            cur = cur[int(seg)]
        else:
            if seg not in cur:
                cur[seg] = {}
            cur = cur[seg]
    last = segs[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def pointer_add(obj: Any, path: str, value: Any) -> None:
    """RFC 6902 'add' semantics. For arrays, '/-' appends and a numeric index
    INSERTS at that position (does not overwrite). For objects, sets the key
    (creating intermediate objects as in pointer_set)."""
    segs = split_pointer(path)
    if not segs:
        raise ValueError("cannot add to root via pointer_add")
    cur = obj
    for seg in segs[:-1]:
        if isinstance(cur, list):
            cur = cur[int(seg)]
        else:
            if seg not in cur:
                cur[seg] = {}
            cur = cur[seg]
    last = segs[-1]
    if isinstance(cur, list):
        if last == "-":
            cur.append(value)
        else:
            cur.insert(int(last), value)
    else:
        cur[last] = value


def pointer_remove(obj: Any, path: str) -> None:
    segs = split_pointer(path)
    if not segs:
        raise ValueError("cannot remove root")
    cur = obj
    for seg in segs[:-1]:
        if isinstance(cur, list):
            cur = cur[int(seg)]
        else:
            cur = cur[seg]
    last = segs[-1]
    if isinstance(cur, list):
        del cur[int(last)]
    else:
        del cur[last]


def _op_field(op: Any, name: str) -> Any:
    if isinstance(op, dict):
        return op.get(name)
    return getattr(op, name, None)


def apply_patch(doc: Any, patch: list[Any]) -> Any:
    """Apply a list of patch ops (dicts or PatchOp-like objects) to a deep
    copy of `doc`, returning the result."""
    out = copy.deepcopy(doc)
    for op in patch:
        kind = _op_field(op, "op")
        path = _op_field(op, "path")
        if kind == "replace":
            pointer_set(out, path, _op_field(op, "value"))
        elif kind == "add":
            pointer_add(out, path, _op_field(op, "value"))
        elif kind == "remove":
            pointer_remove(out, path)
        else:
            raise ValueError(f"unsupported op: {kind!r}")
    return out


def list_leaf_paths(obj: Any, prefix: str = "") -> list[str]:
    """Return JSON Pointer paths to every leaf (primitive) in obj."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            esc = str(k).replace("~", "~0").replace("/", "~1")
            out.extend(list_leaf_paths(v, f"{prefix}/{esc}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(list_leaf_paths(v, f"{prefix}/{i}"))
    else:
        out.append(prefix or "")
    return out
