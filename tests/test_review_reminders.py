"""Unit tests for the deterministic self-review reminders."""

import copy

from methods.common import SELF_REVIEW_CHECK, SELF_REVIEW_OUTPUT, SELF_REVIEW_TEMPLATE
from methods.review_reminders import build_review_reminders


def test_template_is_exact_concatenation():
    assert SELF_REVIEW_TEMPLATE == SELF_REVIEW_CHECK + "\n\n" + SELF_REVIEW_OUTPUT


def test_mirrored_blocks_detected():
    doc = {
        "rule": {"low": {"x": 1, "y": "a"}, "high": {"x": 2, "y": "b"}},
        "active": {"x": 1, "y": "a"},
        "other": 5,
    }
    patch = [{"op": "replace", "path": "/other", "value": 6}]
    applied = copy.deepcopy(doc)
    applied["other"] = 6
    text = build_review_reminders(doc, patch, applied)
    assert "Mirrored blocks" in text
    assert "/active == /rule/low" in text


def test_repeated_pattern_residue_detected():
    doc = {"items": [{"price": 1}, {"price": 2}, {"price": 3}, {"price": 4}]}
    patch = [
        {"op": "replace", "path": "/items/0/price", "value": 10},
        {"op": "replace", "path": "/items/1/price", "value": 20},
    ]
    applied = copy.deepcopy(doc)
    applied["items"][0]["price"] = 10
    applied["items"][1]["price"] = 20
    text = build_review_reminders(doc, patch, applied)
    assert "repeated edit pattern" in text
    assert "pattern /items/*/price (applied 2 times in your patch)" in text
    assert "/items/2/price" in text and "/items/3/price" in text
    assert "/items/0/price" not in text.split("pattern", 1)[1]


def test_single_instance_pattern_not_reported():
    doc = {"items": [{"price": 1}, {"price": 2}]}
    patch = [{"op": "replace", "path": "/items/0/price", "value": 10}]
    applied = copy.deepcopy(doc)
    applied["items"][0]["price"] = 10
    assert build_review_reminders(doc, patch, applied) == ""


def test_no_signal_returns_empty():
    doc = {"a": 1, "b": {"c": 2}}
    patch = [{"op": "replace", "path": "/a", "value": 3}]
    applied = {"a": 3, "b": {"c": 2}}
    assert build_review_reminders(doc, patch, applied) == ""


def test_values_never_leak_into_reminder():
    doc = {
        "rule": {"low": {"x": 91234, "y": "secretvalue"}},
        "active": {"x": 91234, "y": "secretvalue"},
        "items": [{"qty": 7771}, {"qty": 7772}, {"qty": 7773}],
    }
    patch = [
        {"op": "replace", "path": "/items/0/qty", "value": 1},
        {"op": "replace", "path": "/items/1/qty", "value": 2},
    ]
    applied = copy.deepcopy(doc)
    applied["items"][0]["qty"] = 1
    applied["items"][1]["qty"] = 2
    text = build_review_reminders(doc, patch, applied)
    assert text
    for leak in ("91234", "secretvalue", "7771", "7772", "7773"):
        assert leak not in text


def test_unappliable_patch_still_reports_mirrors_only():
    doc = {
        "rule": {"low": {"x": 1, "y": "a"}},
        "active": {"x": 1, "y": "a"},
        "items": [{"q": 1}, {"q": 2}, {"q": 3}],
    }
    patch = [
        {"op": "replace", "path": "/items/0/q", "value": 9},
        {"op": "replace", "path": "/items/1/q", "value": 9},
    ]
    text = build_review_reminders(doc, patch, None)
    assert "/active == /rule/low" in text
    assert "repeated edit pattern" not in text
