"""Stdlib-only HTTP server backing the annotation GUI.

Endpoints:
  GET  /                  -> the single-page UI (templates/index.html)
  GET  /api/samples       -> [{id, domain, n_keys, annotated, excluded}] (sorted)
  GET  /api/sample/<id>   -> {id, domain, conversation, current_json,
                             edit_instruction, annotation?}      (one sample)
  POST /api/annotate      -> body {id, expected_patch, annotator,
                             annotation_notes}; writes annotated/<id>.json,
                             returns {ok, expected_json}
  POST /api/exclude       -> body {id, excluded: bool}; toggles the `excluded`
                             flag on annotated/<id>.json (sample must already
                             be annotated). Returns {ok, excluded}.

No third-party dependencies: samples are read/written as plain dicts. The
written file mirrors AnnotatedSample's fields so it can later be loaded with
pydantic if desired.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from benchmark.jsonptr import apply_patch, list_leaf_paths

_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class Config:
    samples_dir: Path = Path("benchmark/data/samples")
    annotated_dir: Path = Path("benchmark/data/annotated")
    template: Path = Path(__file__).resolve().parent / "templates" / "index.html"


def _merge_assistant_turns(conversation: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the full artifact from single-key assistant turns."""
    merged: dict[str, Any] = {}
    for m in conversation:
        if m.get("role") != "assistant":
            continue
        try:
            obj = json.loads(m.get("content", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            merged.update(obj)
    return merged


_SKELETONS_CACHE: dict[str, Any] | None = None


def _skeletons() -> dict[str, Any]:
    """Lazy-load the skeleton catalog (domain -> theme -> patterns) from
    benchmark/data/skeletons.json (next to the samples dir). Returns {} if
    absent — the GUI then just omits the patterns."""
    global _SKELETONS_CACHE
    if _SKELETONS_CACHE is None:
        try:
            _SKELETONS_CACHE = json.loads(
                (Config.samples_dir.parent / "skeletons.json").read_text()
            )
        except Exception:  # noqa: BLE001
            _SKELETONS_CACHE = {}
    return _SKELETONS_CACHE


def _load_sample(sample_id: str) -> dict[str, Any] | None:
    path = Config.samples_dir / f"{sample_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # current_json is a serialized computed field; recompute defensively in
    # case an older sample lacks it.
    if "current_json" not in data:
        data["current_json"] = _merge_assistant_turns(data.get("conversation", []))
    # attach this sample's skeleton patterns for the GUI. Prefer the patterns
    # stored directly on the sample JSON (self-contained, no catalog needed);
    # fall back to the legacy skeletons.json catalog for older samples.
    if data.get("conversation_pattern") or data.get("cascade_pattern"):
        data["skeleton"] = {
            "conversation_pattern": data.get("conversation_pattern"),
            "edit_pattern": data.get("edit_pattern"),
            "cascade_pattern": data.get("cascade_pattern"),
            "requires_master_data": data.get("requires_master_data"),
        }
    else:
        sk = _skeletons().get(data.get("domain"), {}).get(data.get("theme"))
        if sk:
            data["skeleton"] = sk
    return data


def _list_samples() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in sorted(Config.samples_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        sid = data.get("id", f.stem)
        cj = data.get("current_json") or _merge_assistant_turns(data.get("conversation", []))
        ann_path = Config.annotated_dir / f"{sid}.json"
        annotated = ann_path.exists()
        excluded = False
        review_edited = False
        if annotated:
            try:
                ann = json.loads(ann_path.read_text())
                excluded = bool(ann.get("excluded", False))
                # "review_edited": the gold was changed in consultation during
                # human review. Explicit flag, or inferred from a "REVIEW EDIT"
                # marker left in annotation_notes.
                review_edited = bool(ann.get("review_edited", False)) or (
                    "REVIEW EDIT" in (ann.get("annotation_notes") or "")
                )
            except Exception:  # noqa: BLE001
                excluded = False
        out.append({
            "id": sid,
            "domain": data.get("domain", ""),
            "target_leaves": data.get("target_leaves"),
            "n_keys": len(cj) if isinstance(cj, dict) else 0,
            "n_leaves": len(list_leaf_paths(cj)) if isinstance(cj, dict) else 0,
            "annotated": annotated,
            "review_edited": review_edited,
            "excluded": excluded,
        })
    return out


def _spec_placeholder(op: dict[str, Any]) -> str:
    """Human-readable description of a non-strict matcher, used as the
    placeholder leaf value when building expected_json for preview/storage.
    The evaluator never reads expected_json; it uses the patch ops directly."""
    m = op.get("matcher") or "strict"
    opt = "(optional) " if op.get("optional") else ""
    if m == "date_range":
        return f"<{opt}date in ({op.get('after') or '(none)'}, {op.get('before') or '(none)'}]>"
    if m == "regex_match":
        return f"<{opt}matches /{op.get('pattern') or ''}/>"
    if m == "any":
        return f"<{opt}any value>"
    if m == "sum_with":
        paths = op.get("paths") or []
        return f"<{opt}sums to {op.get('expected_sum')} across [{', '.join(paths)}]>"
    if m == "ref_equal":
        return f"<{opt}same value as {op.get('ref_path') or '(ref_path?)'}>"
    return ""


def _expand_op_for_display(op: dict[str, Any]) -> dict[str, Any]:
    """Convert a patch op into one safe to apply for the expected_json preview:
    strict ops are passed through; non-strict ops have their value replaced
    with a descriptive placeholder string so the leaf reads as a spec."""
    matcher = op.get("matcher") or "strict"
    # strict add/replace passes through (a "<any>" sub-field shows literally);
    # non-strict matchers render as a descriptive placeholder.
    if op.get("op") == "remove" or matcher == "strict":
        return op
    return {**op, "value": _spec_placeholder(op)}


def _save_annotation(body: dict[str, Any]) -> dict[str, Any]:
    sid = body.get("id", "")
    if not _ID_RE.match(sid):
        raise ValueError(f"invalid id {sid!r}")
    sample = _load_sample(sid)
    if sample is None:
        raise ValueError(f"sample {sid!r} not found")
    patch = body.get("expected_patch", [])
    # When multiple ops share the same path, they are OR alternatives — apply
    # only the FIRST occurrence per path to compute the canonical
    # expected_json preview. For non-strict matchers, the canonical leaf value
    # is "any value satisfying the spec"; we render a descriptive placeholder
    # string into expected_json so the saved artifact stays human-readable.
    # (The evaluator computes its own gold_after from patch ops; expected_json
    # is purely the saved record for review.)
    seen_paths: set[str] = set()
    display_patch: list[dict[str, Any]] = []
    for op in patch:
        # Multi-path constraint ops (e.g. sum_with with `paths`) have no
        # singular `path`. Expand them into one placeholder display op per
        # constituent path so the expected_json preview shows the constraint.
        if op.get("paths"):
            placeholder = _spec_placeholder(op)
            for p in op["paths"]:
                if p in seen_paths:
                    continue
                seen_paths.add(p)
                display_patch.append({"op": "replace", "path": p, "value": placeholder})
            continue
        p = op.get("path")
        if p in seen_paths:
            continue
        seen_paths.add(p)
        display_patch.append(_expand_op_for_display(op))
    expected_json = apply_patch(sample["current_json"], display_patch)
    # The annotator may override the original edit_instruction (e.g. tightening
    # an ambiguous phrasing). If the body field is missing or empty, fall back
    # to the sample's value so existing UIs without the field still work.
    edit_instruction = body.get("edit_instruction") or sample.get("edit_instruction", "")
    # ValueConstraints are passed through as-is; the evaluator dispatches on
    # `matcher` (date_range / regex_match).
    value_constraints = body.get("value_constraints", []) or []
    # Preserve any existing `excluded` flag on re-save so toggling exclude is
    # orthogonal to re-annotating.
    prior_excluded = False
    prior_path = Config.annotated_dir / f"{sid}.json"
    if prior_path.exists():
        try:
            prior_excluded = bool(json.loads(prior_path.read_text()).get("excluded", False))
        except Exception:  # noqa: BLE001
            prior_excluded = False
    annotated = {
        "id": sid,
        "domain": sample.get("domain", ""),
        "target_leaves": sample.get("target_leaves"),
        "generator": sample.get("generator"),
        "conversation": sample.get("conversation", []),
        "edit_instruction": edit_instruction,
        "current_json": sample["current_json"],
        "theme": sample.get("theme"),
        "conversation_pattern": sample.get("conversation_pattern"),
        "edit_pattern": sample.get("edit_pattern"),
        "cascade_pattern": sample.get("cascade_pattern"),
        "requires_master_data": sample.get("requires_master_data"),
        # carried from the sample so the evaluator can render it with the feedback
        # (the dependent values for a new entity come from here); dropping it made
        # master_data cascades impossible to satisfy.
        "master_data": sample.get("master_data"),
        "expected_patch": patch,
        "expected_json": expected_json,
        "value_constraints": value_constraints,
        "annotator": body.get("annotator", ""),
        "annotation_notes": body.get("annotation_notes", ""),
        "annotated": True,
        "excluded": prior_excluded,
    }
    Config.annotated_dir.mkdir(parents=True, exist_ok=True)
    (Config.annotated_dir / f"{sid}.json").write_text(
        json.dumps(annotated, ensure_ascii=False, indent=2)
    )
    return {"ok": True, "expected_json": expected_json}


def _toggle_excluded(body: dict[str, Any]) -> dict[str, Any]:
    sid = body.get("id", "")
    if not _ID_RE.match(sid):
        raise ValueError(f"invalid id {sid!r}")
    path = Config.annotated_dir / f"{sid}.json"
    if not path.exists():
        raise ValueError(f"sample {sid!r} is not annotated yet")
    data = json.loads(path.read_text())
    data["excluded"] = bool(body.get("excluded", False))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return {"ok": True, "excluded": data["excluded"]}


class Handler(BaseHTTPRequestHandler):
    # quieter logging
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, obj: Any, status: int = 200) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_html(Config.template.read_text())
            return
        if path == "/api/samples":
            self._send_json(_list_samples())
            return
        if path.startswith("/api/sample/"):
            sid = path[len("/api/sample/"):]
            if not _ID_RE.match(sid):
                self._send_json({"error": "invalid id"}, 400)
                return
            sample = _load_sample(sid)
            if sample is None:
                self._send_json({"error": "not found"}, 404)
                return
            ann_path = Config.annotated_dir / f"{sid}.json"
            if ann_path.exists():
                try:
                    sample["annotation"] = json.loads(ann_path.read_text())
                except Exception:  # noqa: BLE001
                    pass
            self._send_json(sample)
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in ("/api/annotate", "/api/exclude"):
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/annotate":
                result = _save_annotation(body)
            else:
                result = _toggle_excluded(body)
            self._send_json(result)
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"{type(e).__name__}: {e}"}, 400)


def serve(*, host: str, port: int, samples_dir: Path, annotated_dir: Path) -> None:
    Config.samples_dir = samples_dir
    Config.annotated_dir = annotated_dir
    annotated_dir.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), Handler)
    n = len(list(samples_dir.glob("*.json")))
    print(f"Annotator GUI serving {n} samples from {samples_dir}")
    print(f"  → open http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        httpd.server_close()
