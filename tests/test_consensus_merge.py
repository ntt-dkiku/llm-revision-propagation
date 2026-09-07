"""Unit tests for the deterministic AND/OR consensus merge and the
root-replace patch path through the scorer."""

from benchmark.jsonptr import apply_patch
from benchmark.strict_score import score_strict
from methods.subagent_consensus import merge_consensus


CUR = {"a": 1, "b": {"x": 10, "y": 20}, "items": [{"p": 1}, {"p": 2}]}


def s(**over):
    import copy
    d = copy.deepcopy(CUR)
    for k, v in over.items():
        parts = k.split(".")
        node = d
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    return d


def test_unanimous_change_kept_in_both_modes():
    samples = [s(a=5), s(a=5), s(a=5)]
    for mode in ("and", "or"):
        out = merge_consensus(CUR, samples, mode)
        assert out["a"] == 5 and out["b"] == CUR["b"]


def test_single_sample_change_distinguishes_modes():
    samples = [s(a=5), s(), s()]                              # 1 of 3 changes
    assert merge_consensus(CUR, samples, "and")["a"] == 1     # no unanimity
    assert merge_consensus(CUR, samples, "or")["a"] == 5      # union adopts any change
    assert merge_consensus(CUR, samples, "maj")["a"] == 1     # no strict majority


def test_two_of_three_change():
    samples = [s(a=5), s(a=5), s()]
    assert merge_consensus(CUR, samples, "and")["a"] == 1     # not unanimous
    assert merge_consensus(CUR, samples, "or")["a"] == 5      # union
    assert merge_consensus(CUR, samples, "maj")["a"] == 5     # 2/3 majority


def test_value_disagreement():
    samples = [s(a=5), s(a=6), s(a=5)]
    assert merge_consensus(CUR, samples, "and")["a"] == 1     # values differ -> keep current
    assert merge_consensus(CUR, samples, "or")["a"] == 5      # earliest changing sample
    assert merge_consensus(CUR, samples, "maj")["a"] == 5     # 2/3 on value 5


def test_all_distinct_changes():
    samples = [s(a=5), s(a=6), s(a=7)]
    assert merge_consensus(CUR, samples, "and")["a"] == 1
    assert merge_consensus(CUR, samples, "or")["a"] == 5      # earliest wins the conflict
    assert merge_consensus(CUR, samples, "maj")["a"] == 1     # no strict majority


def test_nested_dict_merge_is_per_key():
    samples = [s(**{"b.x": 99}), s(**{"b.x": 99, "b.y": 77}), s(**{"b.x": 99})]
    out_and = merge_consensus(CUR, samples, "and")
    assert out_and["b"]["x"] == 99 and out_and["b"]["y"] == 20
    out_or = merge_consensus(CUR, samples, "or")
    assert out_or["b"]["x"] == 99 and out_or["b"]["y"] == 77  # union keeps the lone change
    out_maj = merge_consensus(CUR, samples, "maj")
    assert out_maj["b"]["x"] == 99 and out_maj["b"]["y"] == 20  # 1/3 is no majority


def test_key_addition_and_removal():
    add2 = s(); add2["c"] = 7
    add3 = s(); add3["c"] = 7
    samples = [s(), add2, add3]
    assert "c" not in merge_consensus(CUR, samples, "and")    # not unanimous
    assert merge_consensus(CUR, samples, "or")["c"] == 7      # union adds
    assert merge_consensus(CUR, samples, "maj")["c"] == 7     # 2/3 adds
    rm = s(); del rm["a"]
    samples = [rm, rm, rm]
    assert "a" not in merge_consensus(CUR, samples, "and")    # unanimous removal
    samples = [rm, s(), s()]
    assert "a" not in merge_consensus(CUR, samples, "or")     # union removes too
    assert merge_consensus(CUR, samples, "maj")["a"] == 1     # 1/3 removal is no majority


def test_array_same_length_elementwise():
    samples = [s(**{"items": [{"p": 9}, {"p": 2}]}),
               s(**{"items": [{"p": 9}, {"p": 2}]}),
               s(**{"items": [{"p": 1}, {"p": 2}]})]
    assert merge_consensus(CUR, samples, "and")["items"][0]["p"] == 1
    assert merge_consensus(CUR, samples, "or")["items"][0]["p"] == 9
    assert merge_consensus(CUR, samples, "maj")["items"][0]["p"] == 9


def test_array_length_disagreement_is_one_unit():
    grow = s(**{"items": [{"p": 1}, {"p": 2}, {"p": 3}]})
    samples = [grow, grow, s()]
    assert merge_consensus(CUR, samples, "and")["items"] == CUR["items"]      # not unanimous
    assert len(merge_consensus(CUR, samples, "or")["items"]) == 3             # union takes the change
    assert len(merge_consensus(CUR, samples, "maj")["items"]) == 3            # 2/3 majority


def test_k2_derivation_uses_same_function():
    samples = [s(a=5), s(a=5)]
    assert merge_consensus(CUR, samples[:2], "and")["a"] == 5


def test_synthesized_patch_applies_and_scores():
    from methods.subagent_consensus import consensus_to_patch
    consensus = s(a=5)
    patch = consensus_to_patch(CUR, consensus)
    assert apply_patch(CUR, patch) == consensus
    gold = [{"op": "replace", "path": "/a", "value": 5, "kind": "direct"}]
    r = score_strict(CUR, gold, patch, None)
    assert r["over_edit"] == 0 and r["miss"] == 0 and r["value_recall"] >= 1.0


def test_synthesized_patch_handles_add_and_remove_keys():
    from methods.subagent_consensus import consensus_to_patch
    consensus = s(a=5)
    del consensus["b"]
    consensus["new_top"] = {"k": 1}
    patch = consensus_to_patch(CUR, consensus)
    assert apply_patch(CUR, patch) == consensus


def test_array_insertion_synthesized_as_add_op():
    from methods.subagent_consensus import consensus_to_patch
    cur = {"tasks": [{"n": "a"}, {"n": "b"}, {"n": "c"}]}
    new = {"tasks": [{"n": "a"}, {"n": "x"}, {"n": "b2"}, {"n": "c"}]}
    patch = consensus_to_patch(cur, new)
    assert apply_patch(cur, patch) == new
    assert any(o["op"] == "add" and o["path"].startswith("/tasks/") for o in patch)


def test_exact_sample_match_reuses_raw_patch():
    from methods.subagent_consensus import build_consensus_patch
    cur = {"a": 1, "tasks": [{"n": "a"}]}
    raw = [{"op": "add", "path": "/tasks/1", "value": {"n": "z"}}]
    applied = apply_patch(cur, raw)
    samples = [{"patch": raw, "_applied": applied}]
    assert build_consensus_patch(cur, applied, samples) == raw


def test_inserted_arrays_vote_as_one_unit():
    grow_a = s(**{"items": [{"p": 1}, {"p": 9}, {"p": 2}]})
    grow_b = s(**{"items": [{"p": 1}, {"p": 9}, {"p": 2}]})
    grow_c = s(**{"items": [{"p": 1}, {"p": 8}, {"p": 2}]})  # one element differs
    assert merge_consensus(CUR, [grow_a, grow_b, grow_b], "and")["items"][1]["p"] == 9  # unanimous unit
    assert merge_consensus(CUR, [grow_a, grow_b, grow_c], "and")["items"] == CUR["items"]  # no unanimity -> keep current whole
    assert merge_consensus(CUR, [grow_a, grow_b, grow_c], "maj")["items"][1]["p"] == 9  # 2/3 identical arrays
    assert merge_consensus(CUR, [grow_c, grow_a, grow_b], "or")["items"][1]["p"] == 8   # earliest differing array
