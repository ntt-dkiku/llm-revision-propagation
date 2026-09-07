"""Regression tests for the identity-anchored ("strict") cascade scorer that is
wired into benchmark.evaluator (`benchmark.strict_score.score_strict`).

The scorer recovers element identity across array remove/insert shifts so that,
for an id-less array, it can tell apart:
  * a FAILED CASCADE on a shifted survivor  -> value_recall / cascade_recall drop
  * a NOT-PROPAGATED survivor (left as-is)   -> miss (location not addressed)
  * a SPURIOUS edit / not-removed element    -> over_edit
without giving spurious "location" credit for shift-induced position changes.

Each entry in CASES is fully self-contained (its own `current` document, gold
patch and model patch) and asserts the headline metrics. Expected values are
hand-derived from the intended semantics.

HOW TO ADD A CASE
-----------------
Append a dict to CASES:
    {
        "id": "short-unique-name",
        "current": {...},                  # original document
        "gold":  [ {op..., "kind": "direct"|"cascade"}, ... ],
        "model": [ {op...}, ... ],          # model patch under test
        "expect": {"value_recall": .., "miss": .., "over_edit": ..,
                    "loc_f1": .., "direct_recall": .., "cascade_recall": ..},
    }
Only the keys present in "expect" are asserted. Distil any real failing case
from review into a minimal document + patches and lock it in here.
"""
import copy

import pytest

from benchmark.strict_score import score_strict

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

# Remove-with-shift. E1 (idx1) is removed; E2 (idx2) shifts into idx1 and its
# value genuinely changes 99->20, which coincidentally equals the value that
# used to sit at idx1 (E1.v == 20) — the position-vs-identity trap.
REMOVE_DOC = {
    "items": [
        {"k": "a", "v": 10},   # idx0  E0 (stays)
        {"k": "b", "v": 20},   # idx1  E1 (removed)
        {"k": "c", "v": 99},   # idx2  E2 (shifts to idx1; v 99->20)
        {"k": "d", "v": 30},   # idx3  E3 (shifts to idx2; v 30->40)
    ],
    "n": 4,
}
REMOVE_GOLD = [
    {"op": "remove", "path": "/items/1", "kind": "direct"},
    {"op": "replace", "path": "/items/1/v", "value": 20, "kind": "cascade"},
    {"op": "replace", "path": "/items/2/v", "value": 40, "kind": "cascade"},
    {"op": "replace", "path": "/n", "value": 3, "kind": "cascade"},
]

# Insert-with-shift.
INSERT_DOC = {
    "items": [
        {"k": "a", "v": 10},   # idx0  E0 (stays)
        {"k": "b", "v": 20},   # idx1  E1 (shifts to idx2; v 20->25)
        {"k": "c", "v": 30},   # idx2  E2 (shifts to idx3; v 30->35)
    ],
    "n": 3,
}
INSERT_GOLD = [
    {"op": "add", "path": "/items/1", "value": {"k": "NEW", "v": 99}, "kind": "direct"},
    {"op": "replace", "path": "/items/2/v", "value": 25, "kind": "cascade"},
    {"op": "replace", "path": "/items/3/v", "value": 35, "kind": "cascade"},
    {"op": "replace", "path": "/n", "value": 4, "kind": "cascade"},
]

# Flat document, no arrays / no structural change.
FLAT_DOC = {"title": "X", "count": 5, "note": "hello"}
FLAT_GOLD = [{"op": "replace", "path": "/count", "value": 6, "kind": "direct"}]


def _gold_with(patch, path, value):
    out = copy.deepcopy(patch)
    for o in out:
        if o.get("path") == path:
            o["value"] = value
    return out


CASES = [
    # ---- remove cascade ----
    {
        "id": "remove-correct",
        "current": REMOVE_DOC, "gold": REMOVE_GOLD, "model": REMOVE_GOLD,
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0, "loc_f1": 1.0,
                   "direct_recall": 1.0, "cascade_recall": 1.0},
    },
    {
        # removed correctly, but the shifted survivor's cascade value is wrong:
        # value_recall / cascade_recall drop, NOT over_edit; location still hit.
        "id": "remove-survivor-value-wrong",
        "current": REMOVE_DOC, "gold": REMOVE_GOLD,
        "model": _gold_with(REMOVE_GOLD, "/items/1/v", 21),
        "expect": {"value_recall": 0.75, "miss": 0, "over_edit": 0, "loc_f1": 1.0,
                   "direct_recall": 1.0, "cascade_recall": pytest.approx(2 / 3, abs=1e-3)},
    },
    {
        # removed but downstream NOT propagated (survivors left at original
        # values): the un-addressed cascades are MISSES, not value-near-hits.
        "id": "remove-without-propagation",
        "current": REMOVE_DOC, "gold": REMOVE_GOLD,
        "model": [{"op": "remove", "path": "/items/1"}],
        "expect": {"value_recall": 0.25, "miss": 3, "over_edit": 0, "loc_f1": 0.4,
                   "direct_recall": 1.0, "cascade_recall": 0.0},
    },
    {
        # NOT removed; edited in place instead -> removal miss, the kept
        # element's edits are over_edit, survivors fail their cascades.
        "id": "remove-not-done-inplace-edit",
        "current": REMOVE_DOC, "gold": REMOVE_GOLD,
        "model": [
            {"op": "replace", "path": "/items/1/k", "value": "c"},
            {"op": "replace", "path": "/items/1/v", "value": 21},
        ],
        "expect": {"value_recall": 0.0, "miss": 4, "over_edit": 2, "loc_f1": 0.0,
                   "direct_recall": 0.0, "cascade_recall": 0.0},
    },
    {
        # cascade correct, plus a spurious edit to a stable element -> over_edit.
        "id": "remove-correct-plus-spurious-edit",
        "current": REMOVE_DOC, "gold": REMOVE_GOLD,
        "model": REMOVE_GOLD + [{"op": "replace", "path": "/items/0/v", "value": 11}],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 1,
                   "loc_f1": pytest.approx(0.8889, abs=1e-3),
                   "direct_recall": 1.0, "cascade_recall": 1.0},
    },
    # ---- insert cascade ----
    {
        "id": "insert-correct",
        "current": INSERT_DOC, "gold": INSERT_GOLD, "model": INSERT_GOLD,
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0, "loc_f1": 1.0,
                   "direct_recall": 1.0, "cascade_recall": 1.0},
    },
    {
        # the SAME correct result, but the model rewrote the whole array in one
        # `replace /items = [...]` op (no granular add/remove). Provenance is
        # recovered by content alignment, so it scores identically to the
        # granular insert above. (Distilled from travel_itinerary_L100_05 qwen b3.)
        "id": "insert-via-wholesale-array-replace",
        "current": INSERT_DOC, "gold": INSERT_GOLD,
        "model": [
            {"op": "replace", "path": "/items", "value": [
                {"k": "a", "v": 10}, {"k": "NEW", "v": 99},
                {"k": "b", "v": 25}, {"k": "c", "v": 35}]},
            {"op": "replace", "path": "/n", "value": 4},
        ],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0, "loc_f1": 1.0,
                   "direct_recall": 1.0, "cascade_recall": 1.0},
    },
    {
        "id": "insert-downstream-value-wrong",
        "current": INSERT_DOC, "gold": INSERT_GOLD,
        "model": _gold_with(INSERT_GOLD, "/items/2/v", 26),
        "expect": {"value_recall": 0.8, "miss": 0, "over_edit": 0, "loc_f1": 1.0,
                   "direct_recall": 1.0, "cascade_recall": pytest.approx(2 / 3, abs=1e-3)},
    },
    # ---- flat (no structural change) ----
    {
        "id": "flat-correct",
        "current": FLAT_DOC, "gold": FLAT_GOLD, "model": FLAT_GOLD,
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0, "loc_f1": 1.0,
                   "direct_recall": 1.0},
    },
    {
        "id": "flat-required-value-wrong",
        "current": FLAT_DOC, "gold": FLAT_GOLD,
        "model": [{"op": "replace", "path": "/count", "value": 7}],
        "expect": {"value_recall": 0.0, "miss": 0, "over_edit": 0, "loc_f1": 1.0,
                   "direct_recall": 0.0},
    },
    {
        "id": "flat-spurious-edit",
        "current": FLAT_DOC, "gold": FLAT_GOLD,
        "model": FLAT_GOLD + [{"op": "replace", "path": "/note", "value": "bye"}],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 1,
                   "loc_f1": pytest.approx(0.6667, abs=1e-3), "direct_recall": 1.0},
    },
    # ---- trailing-period / whitespace leniency (cosmetic, not penalized) ----
    {
        # gold prescribes "Invoice Y"; model appends a trailing period. Cosmetic
        # only -> full value credit, not a mismatch.
        "id": "trailing-period-on-required-value",
        "current": {"label": "Invoice X", "ref": "stable"},
        "gold": [{"op": "replace", "path": "/label", "value": "Invoice Y",
                  "kind": "direct"}],
        "model": [{"op": "replace", "path": "/label", "value": "Invoice Y."}],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0, "loc_f1": 1.0,
                   "direct_recall": 1.0},
    },
    {
        # reverse direction: gold value ends with a period, model drops it.
        "id": "missing-trailing-period-on-required-value",
        "current": {"label": "Invoice X", "ref": "stable"},
        "gold": [{"op": "replace", "path": "/label", "value": "Invoice Y.",
                  "kind": "direct"}],
        "model": [{"op": "replace", "path": "/label", "value": "Invoice Y"}],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0, "loc_f1": 1.0,
                   "direct_recall": 1.0},
    },
    {
        # a trailing period added to a PROTECTED (non-gold) leaf is cosmetic and
        # must NOT count as over_edit.
        "id": "trailing-period-on-protected-leaf",
        "current": {"label": "Invoice X", "note": "hello"},
        "gold": [{"op": "replace", "path": "/label", "value": "Invoice Y",
                  "kind": "direct"}],
        "model": [{"op": "replace", "path": "/label", "value": "Invoice Y"},
                  {"op": "replace", "path": "/note", "value": "hello."}],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0, "loc_f1": 1.0,
                   "direct_recall": 1.0},
    },
    # ---- free identifier on insert ("<any>" sentinel + ref_equal) ----
    # Insert a task at index 1; its `id` is an arbitrary free identifier (gold
    # value "<any>" => accept any model value), and the shifted successor's predId
    # must point at *whatever id the model chose*, scored via ref_equal.
    {
        "id": "insert-free-id-canonical",
        "current": {"tasks": [{"id": "A", "predId": None},
                              {"id": "B", "predId": "A"}]},
        "gold": [
            {"op": "add", "path": "/tasks/1", "kind": "direct",
             "value": {"id": "<any>", "predId": "A"}},
            {"op": "replace", "path": "/tasks/2/predId", "kind": "cascade",
             "value": "NEW", "matcher": "ref_equal", "ref_path": "/tasks/1/id"},
        ],
        "model": [{"op": "add", "path": "/tasks/1",
                   "value": {"id": "NEW", "predId": "A"}},
                  {"op": "replace", "path": "/tasks/2/predId", "value": "NEW"}],
        "expect": {"value_recall": 1.0, "over_edit": 0, "miss": 0},
    },
    {
        # model chose a DIFFERENT id ("X3") but wired the successor to it
        # consistently -> fully correct, no penalty for the free identifier.
        "id": "insert-free-id-different-but-consistent",
        "current": {"tasks": [{"id": "A", "predId": None},
                              {"id": "B", "predId": "A"}]},
        "gold": [
            {"op": "add", "path": "/tasks/1", "kind": "direct",
             "value": {"id": "<any>", "predId": "A"}},
            {"op": "replace", "path": "/tasks/2/predId", "kind": "cascade",
             "value": "NEW", "matcher": "ref_equal", "ref_path": "/tasks/1/id"},
        ],
        "model": [{"op": "add", "path": "/tasks/1",
                   "value": {"id": "X3", "predId": "A"}},
                  {"op": "replace", "path": "/tasks/2/predId", "value": "X3"}],
        "expect": {"value_recall": 1.0, "over_edit": 0, "miss": 0},
    },
    {
        # model chose a different id but FAILED to relink the successor (left it
        # pointing at the old predecessor) -> the ref_equal leaf fails.
        "id": "insert-free-id-inconsistent-ref",
        "current": {"tasks": [{"id": "A", "predId": None},
                              {"id": "B", "predId": "A"}]},
        "gold": [
            {"op": "add", "path": "/tasks/1", "kind": "direct",
             "value": {"id": "<any>", "predId": "A"}},
            {"op": "replace", "path": "/tasks/2/predId", "kind": "cascade",
             "value": "NEW", "matcher": "ref_equal", "ref_path": "/tasks/1/id"},
        ],
        "model": [{"op": "add", "path": "/tasks/1",
                   "value": {"id": "X3", "predId": "A"}},
                  {"op": "replace", "path": "/tasks/2/predId", "value": "A"}],
        "expect": {"value_recall": pytest.approx(2 / 3, abs=1e-3), "over_edit": 0,
                   "cascade_recall": 0.0},
    },
    # ---- positional RENUMBER accepted (optional+any ids, optional+ref_equal refs) ----
    # Insert at idx1. The gold marks the tail ids as free (optional+any) and the
    # tail references as optional+ref_equal, so a model that renumbers ids
    # positionally (new task takes the slot id, later tasks shift up) and rewires
    # references consistently scores fully — same as keeping ids stable.
    {
        "id": "insert-renumber-consistent-full-credit",
        "current": {"tasks": [{"id": "T1", "predId": None},
                              {"id": "T2", "predId": "T1"},
                              {"id": "T3", "predId": "T2"}]},
        "gold": [
            {"op": "add", "path": "/tasks/1", "kind": "direct",
             "value": {"id": "<any>", "predId": "T1"}},
            {"op": "replace", "path": "/tasks/2/predId", "kind": "cascade",
             "value": "T2", "matcher": "ref_equal", "ref_path": "/tasks/1/id"},
            {"op": "replace", "path": "/tasks/2/id", "kind": "cascade",
             "value": "T2", "matcher": "any", "optional": True},
            {"op": "replace", "path": "/tasks/3/id", "kind": "cascade",
             "value": "T3", "matcher": "any", "optional": True},
            {"op": "replace", "path": "/tasks/3/predId", "kind": "cascade",
             "value": "T2", "matcher": "ref_equal", "ref_path": "/tasks/2/id",
             "optional": True},
        ],
        "model": [  # positional renumber: new=T2, old T2->T3, old T3->T4
            {"op": "add", "path": "/tasks/1", "value": {"id": "T2", "predId": "T1"}},
            {"op": "replace", "path": "/tasks/2/id", "value": "T3"},
            {"op": "replace", "path": "/tasks/2/predId", "value": "T2"},
            {"op": "replace", "path": "/tasks/3/id", "value": "T4"},
            {"op": "replace", "path": "/tasks/3/predId", "value": "T3"},
        ],
        "expect": {"value_recall": 1.0, "over_edit": 0, "miss": 0, "loc_f1": 1.0},
    },
    {
        # same gold, but the model forgot to relink T2 to the inserted task
        # (left predId pointing at the old predecessor): the REQUIRED ref_equal
        # fails -> value/cascade drop, but no over_edit.
        "id": "insert-renumber-broken-link-penalized",
        "current": {"tasks": [{"id": "T1", "predId": None},
                              {"id": "T2", "predId": "T1"},
                              {"id": "T3", "predId": "T2"}]},
        "gold": [
            {"op": "add", "path": "/tasks/1", "kind": "direct",
             "value": {"id": "<any>", "predId": "T1"}},
            {"op": "replace", "path": "/tasks/2/predId", "kind": "cascade",
             "value": "T2", "matcher": "ref_equal", "ref_path": "/tasks/1/id"},
            {"op": "replace", "path": "/tasks/2/id", "kind": "cascade",
             "value": "T2", "matcher": "any", "optional": True},
            {"op": "replace", "path": "/tasks/3/id", "kind": "cascade",
             "value": "T3", "matcher": "any", "optional": True},
            {"op": "replace", "path": "/tasks/3/predId", "kind": "cascade",
             "value": "T2", "matcher": "ref_equal", "ref_path": "/tasks/2/id",
             "optional": True},
        ],
        "model": [  # inserted, but never relinked T2 -> still points at T1
            {"op": "add", "path": "/tasks/1", "value": {"id": "NEW", "predId": "T1"}},
        ],
        "expect": {"value_recall": pytest.approx(2 / 3, abs=1e-3), "miss": 1,
                   "cascade_recall": 0.0, "over_edit": 0},
    },
    # ---- numeric tolerance for computed (non-integral) floats --------------
    # A storage estimate `rows * bytes / 1024^3` carries full float64 precision
    # in gold; a model that rounds to display precision (2.0116568 -> 2.01164,
    # rel err ~8e-6) must still count as the correct value, not a miss.
    {
        "id": "float-rounding-accepted",
        "current": {"rows": 1800000, "gib": 1.5087425708770752},
        "gold": [
            {"op": "replace", "path": "/rows", "kind": "direct", "value": 2400000},
            {"op": "replace", "path": "/gib", "kind": "cascade",
             "value": 2.0116567611694336},
        ],
        "model": [
            {"op": "replace", "path": "/rows", "value": 2400000},
            {"op": "replace", "path": "/gib", "value": 2.01164},  # rounded
        ],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0,
                   "cascade_recall": 1.0},
    },
    # A genuine arithmetic discrepancy (~$5 of compounded per-line rounding on a
    # $30k total, rel err ~1.8e-4) is OUTSIDE tolerance and stays a value miss.
    {
        "id": "float-drift-rejected",
        "current": {"total": 0.0},
        "gold": [{"op": "replace", "path": "/total", "kind": "cascade",
                  "value": 30000.1}],
        "model": [{"op": "replace", "path": "/total", "value": 30005.45}],
        "expect": {"value_recall": 0.0, "cascade_recall": 0.0},
    },
    # Integers are compared exactly: a near-miss count (5000 vs 5004, rel 8e-4)
    # is NOT forgiven even though it is within the float tolerance band.
    {
        "id": "integer-nearmiss-rejected",
        "current": {"batch": 0},
        "gold": [{"op": "replace", "path": "/batch", "kind": "cascade",
                  "value": 5000}],
        "model": [{"op": "replace", "path": "/batch", "value": 5004}],
        "expect": {"value_recall": 0.0, "cascade_recall": 0.0},
    },
    # ---- whole-object gold replace is expanded to per-leaf ops ------------
    # A gold that sets a whole object value must match a model that edits the
    # object's leaves (or replaces the whole object), with no phantom over/miss.
    {
        "id": "object-replace-expanded",
        "current": {"contact": {"name": "Maya Chen", "email": "maya@old.com"}},
        "gold": [{"op": "replace", "path": "/contact", "kind": "cascade",
                  "value": {"name": "Nora Ito", "email": "nora@new.com"}}],
        "model": [  # model edits the leaves individually
            {"op": "replace", "path": "/contact/name", "value": "Nora Ito"},
            {"op": "replace", "path": "/contact/email", "value": "nora@new.com"},
        ],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0,
                   "cascade_recall": 1.0},
    },
    {
        "id": "object-replace-whole-model",
        "current": {"contact": {"name": "Maya Chen", "email": "maya@old.com"}},
        "gold": [{"op": "replace", "path": "/contact", "kind": "cascade",
                  "value": {"name": "Nora Ito", "email": "nora@new.com"}}],
        "model": [  # model replaces the whole object
            {"op": "replace", "path": "/contact",
             "value": {"name": "Nora Ito", "email": "nora@new.com"}},
        ],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0},
    },
    # A sub-leaf whose gold value already equals current is a no-op and must be
    # skipped, not counted as a miss: only the genuinely-changed leaf is scored.
    {
        "id": "object-replace-noop-subleaf-skipped",
        "current": {"contact": {"name": "Maya Chen", "email": "keep@same.com"}},
        "gold": [{"op": "replace", "path": "/contact", "kind": "cascade",
                  "value": {"name": "Nora Ito", "email": "keep@same.com"}}],
        "model": [{"op": "replace", "path": "/contact/name", "value": "Nora Ito"}],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0},
    },
    # ---- wholesale array replacement (all-new content) ------------------
    # A role swap replaces a list of permissions with an entirely different list
    # (no textual overlap, shorter). Gold annotates it granularly (replace the
    # surviving prefix + remove the tail); a model that emits a single whole-array
    # replace producing the SAME final list must score perfectly — element
    # identity is moot when the final arrays are identical.
    {
        "id": "wholesale-array-replace-whole-model",
        "current": {"perms": ["old_a", "old_b", "old_c", "old_d"]},
        "gold": [
            {"op": "replace", "path": "/perms/0", "kind": "cascade", "value": "new_x"},
            {"op": "replace", "path": "/perms/1", "kind": "cascade", "value": "new_y"},
            {"op": "remove", "path": "/perms/3", "kind": "cascade"},
            {"op": "remove", "path": "/perms/2", "kind": "cascade"},
        ],
        "model": [{"op": "replace", "path": "/perms", "value": ["new_x", "new_y"]}],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0,
                   "cascade_recall": 1.0},
    },
    # A genuinely-wrong wholesale replace (one element off) is still penalized:
    # the final arrays are NOT equal, so reconciliation does not fire.
    {
        "id": "wholesale-array-replace-wrong",
        "current": {"perms": ["old_a", "old_b", "old_c", "old_d"]},
        "gold": [
            {"op": "replace", "path": "/perms/0", "kind": "cascade", "value": "new_x"},
            {"op": "replace", "path": "/perms/1", "kind": "cascade", "value": "new_y"},
            {"op": "remove", "path": "/perms/3", "kind": "cascade"},
            {"op": "remove", "path": "/perms/2", "kind": "cascade"},
        ],
        "model": [{"op": "replace", "path": "/perms", "value": ["new_x", "WRONG"]}],
        "expect": {"value_recall": 0.5},
    },
    # ---- object-valued ADD into a dict (new key). The gold op carries the whole
    # object; the scorer must expand it to leaves, or a byte-identical model
    # patch scores a phantom miss at the parent + over_edits at the children
    # (this was a real gold-vs-gold failure on the annotated data).
    {
        "id": "dict-add-object-exact",
        "current": {"tasks": {"a": {"dur": 3, "start": 0}}, "summary": {"count": 1}},
        "gold": [
            {"op": "add", "path": "/tasks/b", "kind": "direct",
             "value": {"dur": 2, "start": 3}},
            {"op": "replace", "path": "/summary/count", "kind": "cascade", "value": 2},
        ],
        "model": [
            {"op": "add", "path": "/tasks/b", "value": {"dur": 2, "start": 3}},
            {"op": "replace", "path": "/summary/count", "value": 2},
        ],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0, "loc_f1": 1.0},
    },
    {
        "id": "dict-add-object-wrong-subvalue",
        "current": {"tasks": {"a": {"dur": 3, "start": 0}}, "summary": {"count": 1}},
        "gold": [
            {"op": "add", "path": "/tasks/b", "kind": "direct",
             "value": {"dur": 2, "start": 3}},
            {"op": "replace", "path": "/summary/count", "kind": "cascade", "value": 2},
        ],
        "model": [
            {"op": "add", "path": "/tasks/b", "value": {"dur": 2, "start": 9}},
            {"op": "replace", "path": "/summary/count", "value": 2},
        ],
        # /tasks/b/start touched-but-wrong: value_recall 2/3, no phantom over_edit
        "expect": {"value_recall": 2 / 3, "over_edit": 0},
    },
    {
        "id": "dict-add-object-omitted",
        "current": {"tasks": {"a": {"dur": 3, "start": 0}}, "summary": {"count": 1}},
        "gold": [
            {"op": "add", "path": "/tasks/b", "kind": "direct",
             "value": {"dur": 2, "start": 3}},
            {"op": "replace", "path": "/summary/count", "kind": "cascade", "value": 2},
        ],
        "model": [{"op": "replace", "path": "/summary/count", "value": 2}],
        "expect": {"miss": 2, "over_edit": 0, "direct_recall": 0.0},
    },
    # ---- object-valued REMOVE of a dict key. Neither O nor M holds the unit
    # path as a leaf, so without leaf expansion `touched` stays False and a
    # perfect removal lands in `miss` (also a real gold-vs-gold failure).
    {
        "id": "dict-remove-object-exact",
        "current": {"tasks": {"a": {"dur": 3, "start": 0}, "b": {"dur": 2, "start": 3}},
                    "summary": {"count": 2}},
        "gold": [
            {"op": "remove", "path": "/tasks/b", "kind": "direct"},
            {"op": "replace", "path": "/summary/count", "kind": "cascade", "value": 1},
        ],
        "model": [
            {"op": "remove", "path": "/tasks/b"},
            {"op": "replace", "path": "/summary/count", "value": 1},
        ],
        "expect": {"value_recall": 1.0, "miss": 0, "over_edit": 0, "loc_f1": 1.0},
    },
    {
        "id": "dict-remove-object-not-removed",
        "current": {"tasks": {"a": {"dur": 3, "start": 0}, "b": {"dur": 2, "start": 3}},
                    "summary": {"count": 2}},
        "gold": [
            {"op": "remove", "path": "/tasks/b", "kind": "direct"},
            {"op": "replace", "path": "/summary/count", "kind": "cascade", "value": 1},
        ],
        "model": [{"op": "replace", "path": "/summary/count", "value": 1}],
        "expect": {"miss": 2, "over_edit": 0, "direct_recall": 0.0},
    },
]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_score_strict(case):
    s = score_strict(case["current"], case["gold"], case["model"])
    for key, want in case["expect"].items():
        got = s[key]
        assert got == want or got == pytest.approx(want, abs=1e-3), (
            f"{case['id']}: {key} = {got}, expected {want}\n"
            f"  full metrics: {{k: round(float(v),3) for k,v in s.items()}}"
        )


# --------------------------------------------------------------------------- #
# Gold idempotence over the annotated benchmark data: applying the gold patch
# itself must never register a location error. (value_recall may stay < 1 for
# flex-matcher ops — regex/range golds carry a pattern, not a literal value, so
# replaying them verbatim is not a satisfying model output; over/miss must
# still be zero.)
# --------------------------------------------------------------------------- #

_DATA_DIRS = [d for d in ("benchmark/data/annotated",)
              if __import__("os").path.isdir(d)]


@pytest.mark.skipif(not _DATA_DIRS, reason="annotated benchmark data not present")
def test_gold_idempotence_on_annotated_data():
    import glob
    import json

    failures = []
    for d in _DATA_DIRS:
        for f in sorted(glob.glob(f"{d}/*.json")):
            a = json.load(open(f))
            gold = [{k: v for k, v in op.items() if k != "kind"}
                    for op in a["expected_patch"]]
            r = score_strict(a["current_json"], a["expected_patch"], gold,
                             a.get("value_constraints"))
            if r["over_edit"] != 0 or r["miss"] != 0:
                failures.append((a["id"], r["over_edit"], r["miss"]))
    assert not failures, f"gold-vs-gold location errors: {failures}"
