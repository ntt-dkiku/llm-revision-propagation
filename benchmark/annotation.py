"""Interactive CLI annotator for benchmark cases.

Workflow:
  1. Load every Sample from --in that doesn't yet exist in --out.
  2. For each, pretty-print the conversation, current_json, and edit
     instruction; pre-fill the patch with the direct edit (target → new value).
  3. Prompt the annotator to:
       a. enter the direct edit patch op + cascade patch ops
       b. add optional annotation notes
  4. Apply the patch to obtain expected_json (auto-verified consistency).
  5. Save the AnnotatedSample to --out as JSON.

Usage:
    python -m benchmark.annotation \
        --in benchmark/data/samples/ \
        --out benchmark/data/annotated/ \
        --annotator daisuke

Keyboard:
  - Enter on an empty prompt accepts the shown default.
  - Type `skip` at the patch op prompt to finish patch entry.
  - Type `q` at any prompt to save what you have and move on (case stays
    marked annotated only if expected_patch is non-empty).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmark.jsonptr import (
    apply_patch,
    list_leaf_paths,
    pointer_get,
    pointer_remove,
    pointer_set,
)
from benchmark.schema import (
    AnnotatedSample,
    PatchOp,
    Sample,
)


# --------------------------------- IO -------------------------------------


def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{label}{suffix}: ").strip()
    return raw or (default or "")


def _prompt_value(label: str, default: Any = None) -> Any:
    """Prompt for a JSON value. Accepts JSON literal or raw string."""
    suffix = f" [{json.dumps(default)}]" if default is not None else ""
    raw = input(f"{label}{suffix}: ").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def show_case(case: Sample) -> None:
    _print_header(f"{case.id}    [{case.domain}]")
    print("--- conversation ---")
    for i, m in enumerate(case.conversation, 1):
        body = m.content
        if len(body) > 600:
            body = body[:600] + " ...[truncated]"
        print(f"[{i}] {m.role}:")
        print(body)
    print()
    print("--- current_json ---")
    print(json.dumps(case.current_json, indent=2, ensure_ascii=False))
    print()
    print("--- edit instruction ---")
    print(f"  {case.edit_instruction}")


def collect_patch(case: Sample) -> list[PatchOp]:
    print()
    print("--- patch operations ---")
    print("  First op = the direct edit the user named in edit_instruction.")
    print("  Subsequent ops = cascade ops required to keep the artifact consistent.")
    print("  Per op: op (replace/add/remove), path (JSON Pointer), value (JSON literal).")
    print("  Type 'skip' at op to finish; 'q' to abort and save what you have.")
    patch: list[PatchOp] = []
    idx = 1
    while True:
        label = "direct edit" if idx == 1 else "cascade"
        op = _prompt(
            f"  [{idx}] op ({label}: replace/add/remove or 'skip')",
            "replace" if idx == 1 else "skip",
        ).lower()
        if op == "skip" or op == "":
            break
        if op == "q":
            return patch
        if op not in ("replace", "add", "remove"):
            print(f"    invalid op {op!r}, expected replace/add/remove")
            continue
        path = _prompt(f"  [{idx}] path")
        if not path:
            print("    (empty path, skipped)")
            continue
        kind = "direct" if idx == 1 else "cascade"
        if op == "remove":
            patch.append(PatchOp(op="remove", path=path, kind=kind))
        else:
            current = pointer_get(case.current_json, path, default=None)
            value = _prompt_value(f"  [{idx}] value", current)
            patch.append(PatchOp(op=op, path=path, value=value, kind=kind))  # type: ignore[arg-type]
        idx += 1
    return patch


def annotate_case(case: Sample, annotator: str) -> AnnotatedSample | None:
    show_case(case)
    skip = _prompt("\nAnnotate this case? (y/n/skip)", "y").lower()
    if skip in ("n", "skip"):
        print("  ... skipped")
        return None
    patch = collect_patch(case)
    if not patch:
        print("  (no patch ops — leaving unannotated)")
        return None
    notes = _prompt("Annotation notes (optional)")
    try:
        expected_json = apply_patch(case.current_json, patch)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: applying patch failed: {e}")
        expected_json = None
    # AnnotatedSample inherits computed_field current_json from Sample;
    # we strip it from the dumped dict so pydantic doesn't reject it as
    # an unknown init arg.
    base = {k: v for k, v in case.model_dump().items() if k != "current_json"}
    return AnnotatedSample(
        **base,
        expected_patch=patch,
        expected_json=expected_json,
        annotator=annotator,
        annotation_notes=notes,
        annotated=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_dir", type=Path,
                   default=Path("benchmark/data/samples"))
    p.add_argument("--out", dest="out_dir", type=Path,
                   default=Path("benchmark/data/annotated"))
    p.add_argument("--annotator", default="anon")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after this many annotated cases (0 = unlimited).")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(args.in_dir.glob("*.json"))
    done = 0
    for sf in sample_files:
        out_path = args.out_dir / sf.name
        if out_path.exists():
            continue
        try:
            case = Sample.model_validate_json(sf.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {sf.name}: parse error {e}")
            continue
        annotated = annotate_case(case, args.annotator)
        if annotated is None:
            continue
        out_path.write_text(annotated.model_dump_json(indent=2))
        print(f"  ✓ saved {out_path}")
        done += 1
        if args.limit and done >= args.limit:
            print(f"  limit reached ({done})")
            break
    print(f"\nAnnotated {done} new cases. Total in {args.out_dir}: "
          f"{len(list(args.out_dir.glob('*.json')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
