"""Method registry for benchmark evaluation.

The paper's methods: single-pass baselines (J / H / J+H via the sub-agent
prompt), serial reflection (REFLECT), and parallel-sampling methods
(OR / AND / MAJ via rule-based merging, MED via medoid, SELECT via an LLM
judge), each with call-budget variants for the cost curves.
"""

from __future__ import annotations

from methods.common import RevisionMethod
from methods.subagent_revision import (
    SubHistMethod, SubHistJsonMethod, SubJsonMethod,
    SubHistJsonReviewMethod, SubHistJsonReviewRemindMethod,
    SubHistJsonReviewDeltaMethod, SubHistJsonReviewRemindDeltaMethod,
    SubHistJsonReview3Method, SubHistJsonReview4Method,
    SubHistJsonReview6Method, SubHistJsonReview7Method, SubHistJsonReview8Method,
)
from methods.subagent_consensus import (
    SubHistJsonCons3AndMethod, SubHistJsonCons3OrMethod, SubHistJsonCons3MajMethod,
    SubHistJsonCons5AndMethod, SubHistJsonCons7AndMethod, SubHistJsonCons8AndMethod,
    SubHistJsonCons9AndMethod,
    SubHistJsonJudge2Method, SubHistJsonJudge3Method, SubHistJsonJudge4Method, SubHistJsonJudge5Method,
    SubHistJsonMerge2Method, SubHistJsonMerge3Method, SubHistJsonMerge4Method, SubHistJsonMerge5Method,
)

_CLASSES: list[type[RevisionMethod]] = [
    # baselines
    SubJsonMethod, SubHistMethod, SubHistJsonMethod,
    # serial reflection (REFLECT) + its call-budget variants
    SubHistJsonReviewMethod, SubHistJsonReviewRemindMethod,
    SubHistJsonReviewDeltaMethod, SubHistJsonReviewRemindDeltaMethod,
    SubHistJsonReview3Method, SubHistJsonReview4Method,
    SubHistJsonReview6Method, SubHistJsonReview7Method, SubHistJsonReview8Method,
    # parallel-sampling rule-based consensus (OR / AND / MAJ; MED via medoid)
    SubHistJsonCons3AndMethod, SubHistJsonCons3OrMethod, SubHistJsonCons3MajMethod,
    SubHistJsonCons5AndMethod, SubHistJsonCons7AndMethod, SubHistJsonCons8AndMethod,
    SubHistJsonCons9AndMethod,
    # LLM-based selection (SELECT) at varying sample counts
    SubHistJsonJudge2Method, SubHistJsonJudge3Method, SubHistJsonJudge4Method, SubHistJsonJudge5Method,
    # element-wise majority-merge variants
    SubHistJsonMerge2Method, SubHistJsonMerge3Method, SubHistJsonMerge4Method, SubHistJsonMerge5Method,
]
METHOD_CLASSES: dict[str, type[RevisionMethod]] = {c.name: c for c in _CLASSES}
METHOD_NAMES = tuple(METHOD_CLASSES)


def create_method(method: str) -> RevisionMethod:
    try:
        return METHOD_CLASSES[method]()
    except KeyError as e:
        raise ValueError(f"unknown method: {method!r}") from e
