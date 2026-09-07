---
name: annotate-cascade-gold
description: >-
  How to annotate gold cascade-revision labels for the human-annotated
  JSON-Patch benchmark (benchmark/data). Use this when creating or reviewing
  `expected_patch` gold for a sample: a single feedback edit (direct) plus all
  dependent fields that must change (cascade). Covers the GUI tool, the gold
  schema, the matcher/optional mechanisms for free choices, the determinism +
  completeness checklist, and the project's annotation conventions.
---

# Annotating cascade-revision gold

This benchmark measures **cascade-aware revision**: given a JSON document and a
single feedback edit, does a model apply the direct edit **and** correctly
propagate every dependent (cascade) field, while leaving unrelated fields
untouched? The gold for each sample is a JSON-Patch (`expected_patch`) whose ops
are tagged `direct` or `cascade`.

Your job as annotator: produce a gold patch that is **deterministic** (the
unique correct revision derivable from the document + edit + master_data),
**complete** (no missed cascade), and **fair** (does not penalize choices the
edit leaves open).

## Tooling

- **GUI annotator**: `python -m benchmark.annotator_app --samples <dir-of-unannotated-cases> --annotated benchmark/data/annotated --port 5001`, then open `http://127.0.0.1:5001`. Un-annotated source cases live under `benchmark/data/raw/` (or `benchmark/data/samples/`); gold is written to `benchmark/data/annotated/`.
- **Scorer**: `benchmark/strict_score.py` (identity-anchored). Regression tests in `tests/test_strict_scoring.py` — run `python -m pytest tests/test_strict_scoring.py -q` after any scorer change.
- The GUI renders `ref_equal` / `<any>` / `optional` / `regex_match` ops as readable specs in the expected_json preview. The evaluator reads the **patch ops directly**, never the expected_json preview.

## Gold schema (`expected_patch`)

Each op:

```json
{"op": "replace|add|remove", "path": "/json/pointer", "kind": "direct|cascade", "value": <value>}
```

- **`kind` is mandatory on every op.** `direct` = the field the feedback edit names. `cascade` = a field that must change *as a consequence*. ~90% of gold ops are `cascade`; this tag is the benchmark's core signal — never omit it.
- Typically exactly **one `direct` op** (the trigger), then all `cascade` ops.
- `path` is an RFC-6901 JSON Pointer into `current_json`.
- Case-level fields you rely on: `current_json` (the document), `edit_instruction` (the feedback), `master_data` (ground-truth facts for recompute), `target_leaves` (tl = document size), `annotation_notes`, `value_constraints`, `annotated: true`.

### Matchers & flags — for choices the edit does NOT constrain

Do not hard-code a value the instruction leaves open; use a matcher so any
correct answer passes. Set `matcher` (and helper fields) on the op:

| matcher | use when | fields |
|---|---|---|
| `strict` (default) | the value is fully determined | `value` |
| `regex_match` | open-vocabulary status/label text (synonyms allowed) | `pattern` |
| `any` | value is an arbitrary free identifier (e.g. an inserted element's `id`) | — (or set the sub-field to the sentinel string `"<any>"` inside an add op's value) |
| `ref_equal` | a field must equal whatever value the model chose elsewhere (referential consistency) | `ref_path` |
| `date_range` | a date within a window | `after`, `before` |
| `sum_with` | a set of fields must sum to a total | `paths`, `expected_sum` |

- **`optional: true`** — the op is neither required (not a miss if absent) nor forbidden (not an over-edit if present). Use for legitimately-either-way cascades (e.g. a cosmetic note, or a field that shifts only under one valid strategy).
- **`"<any>"` sentinel**: inside an `add` op's object value, a sub-field valued `"<any>"` is scored as `matcher: any` (free identifier). Score any field that must *reference* that id with `ref_equal` + `ref_path`.
- **Positional renumber** (ids like T1,T2… that may stay stable+append OR renumber+rewire): give the new element id `"<any>"`; tail ids → `optional` + `any`; the first forced reference → required `ref_equal`; tail references → `optional` + `ref_equal` to the previous element's id.

The scorer also ignores cosmetic differences (trailing whitespace / a single
trailing period) and content-aligns wholesale `replace /array=[...]` ops when the
array length changed — so you don't need per-sample hacks for those.

## Workflow

1. Open the case in the GUI. Read `edit_instruction`, `current_json`, `master_data`.
2. Add the **direct** op for the field the edit explicitly names.
3. Trace **every** dependency from that field and add a `cascade` op for each field that must change. Use `master_data` and arithmetic to compute exact values (totals, taxes, dates, GPAs, balances, derived labels, summaries, parallel per-element lists, transitive chains).
4. For any field the edit leaves open, pick the right matcher/optional (table above) instead of a hard value.
5. Save. The GUI writes `annotated/<id>.json` with `annotated: true`.
6. Verify (next section).

## Determinism & completeness checklist

Before marking a case done, confirm:

- [ ] **Deterministic**: the gold is the *unique* correct revision from document + edit + master_data. If two answers are equally valid, encode that with a matcher/`optional`, don't pick one arbitrarily.
- [ ] **Complete (no missed cascade)**: walk the document for every field that *depends* on the edited value — recomputed numbers, derived/summary fields, status labels, parallel per-element lists (e.g. one note per line-item), transitive chains (A→B→C), cross-references. A missed dependent field is a silent gold error.
- [ ] **No over-inclusion**: do not add ops for fields the edit doesn't actually reach. If a field *generally* depends on the edited one but this *particular* edit doesn't change it (e.g. reordering items doesn't change a day's end-time), it is NOT a cascade.
- [ ] **Semantic, not lexical**: the cascade should require understanding the dependency, not string-matching the edited value. (Good benchmark content is arithmetic recompute / date propagation, with no lexical overlap to the trigger.)
- [ ] **Free choices not penalized**: arbitrary ids → `any`/`<any>`; references → `ref_equal`; open-vocab status → `regex_match`; either-way cascades → `optional`.

## Common pitfalls (caught in review — avoid these)

- **No no-op ops.** A gold op must represent an *actual change*. Do NOT add an op that sets a field to the value it already has (e.g. a status flag that was `non_compliant` and stays `non_compliant`). The scorer treats a non-optional gold op as a required change, so a no-op op is counted as a spurious `miss` when the model (correctly) leaves it alone. If a field genuinely shouldn't change, leave it out of the patch entirely.
- **Array elements at leaf granularity, not whole-array replace.** Write `/arr/0 = ...`, `/arr/1 = ...` (one op per element), or `add /arr/-` / `remove /arr/i` for length changes — NOT a single `replace /arr = [a, b]`. The scorer scores at leaf-path level and does NOT expand a gold whole-*array* replace into its elements (element identity needs provenance), so an array-level gold op will mismatch a model's element-level changes (spurious `over_edit` + `miss`), and even a perfect model is penalized.
- **Whole-*object* replace is OK (auto-expanded).** A gold `replace /obj = {k1: v1, k2: v2}` *is* expanded into per-leaf ops by the scorer (sub-leaves equal to the current value are skipped), so it matches a model that edits the leaves or replaces the object. Per-leaf ops (`/obj/k1 = v1`) are equivalent and also fine. (Note: a whole-object replace that *removes* a key is not fully captured — annotate key removals explicitly.)
- **Carry `master_data` into the annotated file.** For swap/master cases, the annotated JSON MUST keep the `master_data` field from the sample — the evaluator renders it with the feedback so the model can know the new entity's attributes. If it's dropped, the cascade values are unknowable and every baseline fails. (The GUI `_save_annotation` now carries it automatically; verify it's present for `requires_master_data` cases.)

## Conventions (project rules — follow exactly)

- **`annotation_notes` justify gold from the document / edit / master_data ONLY.** Never cite model or baseline behavior in notes (it makes the gold look retrofitted/biased). Generic scoring-intent phrasing ("optional, so neither miss nor over_edit") is fine.
- **Review edits**: when changing a gold during human review, **keep the original note and append** `| REVIEW EDIT (with author)` — never silently rewrite. Optionally set `"review_edited": true` (the GUI highlights these yellow and filters by it; a "REVIEW EDIT" marker in notes is also detected).
- **One sample at a time**: in 1-by-1 review, do not auto-advance; finish and confirm each case before the next.
- **Diversity within a tl bucket**: vary scenarios within the same `target_leaves` value; reusing a scenario across different tl is fine, duplicating within one tl is a diversity problem.
- **Scorer changes are always generalized** (never per-sample gold hacks): add a regression test in `tests/test_strict_scoring.py`, re-score, and confirm the full-run impact is only the intended records (zero regressions). Confirm the approach before editing the scorer.
- **English only** for code, data, prompts, and notes.

## Worked example

`course_plan_L10_01` — edit: "Change Biology 101 to 3 credit hours."

```json
"expected_patch": [
  {"op":"replace","path":"/courses/1/credits","kind":"direct","value":3},
  {"op":"replace","path":"/summary/total_credits","kind":"cascade","value":6},
  {"op":"replace","path":"/summary/credits_remaining","kind":"cascade","value":3},
  {"op":"replace","path":"/summary/gpa","kind":"cascade","value":3.5}
]
```
`annotation_notes`: "Biology 101 4->3 cr. total 7->6, remaining 9-6=3, gpa=(3*4+3*3)/6=3.5."

The single `direct` op sets credits; the three `cascade` ops recompute everything
that depends on it. None of the cascade fields share text with "3 credit hours" —
they require understanding the credit→summary dependency and recomputing.

## How the gold is scored (so you know what matters)

`benchmark/strict_score.py` recovers element identity through array shifts and reports:
`loc_precision/recall/f1`, `value_recall`, `direct_recall`, `cascade_recall`,
`over_edit` (# leaves changed that gold did not), `miss` (# required leaves not changed).
A case is an **exact match** iff `over_edit == 0` and `miss == 0` and `value_recall >= 1.0`.
A missed cascade raises `miss`; touching an unrelated field raises `over_edit`;
both break exact match — which is exactly why completeness and no-over-inclusion both matter.
