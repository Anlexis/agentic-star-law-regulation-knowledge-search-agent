# GOV-C2-005 — Unit Tests: RerankFilterNode (inner domain node 3)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# RULES.md C2 is RETIRED (2026-07-27, blocking): execute(self, state) takes no
# `config` parameter. Threshold / top_k overrides are exercised via the
# state-seeded `retrieval_config` field, never a direct execute(state,
# config=...) 2-arg call.
#
# Mirrors docs/03_test_spec.md Sec.2.4 (RRF-01..RRF-08).
# Deterministic — synthetic candidates, no seeded-KB dependency, no LLM, no
# network. framework.* / src.* imports only.

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.rerank_filter_node import RerankFilterNode
from src.schemas.state import from_json, to_json


def _doc(doc_id, score, category="act"):
    return {
        "id": doc_id,
        "title": f"entry {doc_id}",
        "category": category,
        "source": "seeded kb",
        "score": score,
        "excerpt": "excerpt text",
    }


def _docs(pairs):
    """Shorthand: [(id, score), ...] -> candidate entries."""
    return [_doc(doc_id, score) for doc_id, score in pairs]


def _make_state(candidates, **extra) -> dict:
    state = {
        "retrieved_documents": to_json(candidates),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestThresholdAndCap:
    def test_rrf_01_default_threshold_drops_weak_candidates(self):
        result = RerankFilterNode()(_make_state([_doc("gov-a", 0.9), _doc("gov-b", 0.1)]))
        kept = from_json(result["ranked_documents"])
        assert [d["id"] for d in kept] == ["gov-a"]  # 0.1 < default 0.25 floor

    def test_rrf_02_state_score_threshold_override(self):
        state = _make_state(
            [_doc("gov-a", 0.9), _doc("gov-b", 0.3)],
            retrieval_config=to_json({"score_threshold": 0.5}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["gov-a"]

    def test_rrf_03_state_top_k_override(self):
        state = _make_state(
            [_doc("gov-a", 0.9), _doc("gov-b", 0.8), _doc("gov-c", 0.7)],
            retrieval_config=to_json({"top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["gov-a"]

    def test_ranked_documents_is_json_string(self):
        # List-shaped state fields travel as JSON strings.
        result = RerankFilterNode()(_make_state([_doc("gov-a", 0.9)]))
        assert isinstance(result["ranked_documents"], str)


class TestCategoryBoost:
    def test_rrf_04_matching_category_is_boosted_and_reranked(self):
        state = _make_state(
            [_doc("gov-a", 0.30, category="cabinet_order"), _doc("gov-b", 0.25, category="act")],
            query_filters=to_json({"category": "act", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["gov-b", "gov-a"]
        assert kept[0]["score"] == 0.35  # 0.25 + 0.1 category boost

    def test_rrf_05_boost_is_capped_at_one(self):
        state = _make_state(
            [_doc("gov-a", 0.95, category="act")],
            query_filters=to_json({"category": "act", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert kept[0]["score"] == 1.0


class TestCallerTopK:
    def test_rrf_06_stricter_caller_top_k_wins(self):
        state = _make_state(
            [_doc("gov-a", 0.9), _doc("gov-b", 0.8), _doc("gov-c", 0.7)],
            query_filters=to_json({"category": None, "top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["gov-a"]

    def test_rrf_06_looser_caller_top_k_does_not_widen(self):
        state = _make_state(
            [_doc("gov-a", 0.9), _doc("gov-b", 0.8), _doc("gov-c", 0.7)],
            query_filters=to_json({"category": None, "top_k": 10}),
            retrieval_config=to_json({"top_k": 2, "score_threshold": 0.25}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["gov-a", "gov-b"]


class TestRobustness:
    def test_rrf_07_garbage_candidates_are_skipped_or_dropped(self):
        candidates = [
            "not-a-dict",
            {"id": "gov-bad", "title": "b", "category": "x", "source": "s", "score": "NaN?", "excerpt": "e"},
            _doc("gov-a", 0.9),
        ]
        kept = from_json(RerankFilterNode()(_make_state(candidates))["ranked_documents"])
        # The string entry is skipped; the uncoercible score becomes 0.0 and
        # falls below the relevance floor.
        assert [d["id"] for d in kept] == ["gov-a"]

    def test_rrf_08_deterministic_tie_break_by_id(self):
        kept = from_json(RerankFilterNode()(_make_state([_doc("gov-b", 0.5), _doc("gov-a", 0.5)]))["ranked_documents"])
        assert [d["id"] for d in kept] == ["gov-a", "gov-b"]


class TestOverridesCannotFailOpen:
    """A non-finite override must not silently look applied.

    This node can be driven directly, so it re-checks what reached it. NaN is
    the case worth naming: `max(0.25, nan)` returns 0.25 and `min(4, nan)`
    returns 4, so an unchecked non-finite override leaves the configured value
    in place while appearing to have been honoured. Falling back to the
    configured value is the SAFE outcome here — the test pins that it is
    reached deliberately rather than by accident of comparison semantics.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), True, "2", None, [2]])
    def test_unusable_top_k_override_leaves_the_configured_cap(self, bad):
        state = _make_state(
            _docs([("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5)]),
            query_filters=to_json({"top_k": bad}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert len(kept) == 4  # the configured cap, untouched

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.9", None])
    def test_unusable_score_threshold_override_leaves_the_configured_floor(self, bad):
        state = _make_state(
            _docs([("a", 0.9), ("b", 0.1)]),
            query_filters=to_json({"score_threshold": bad}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        # The configured 0.25 floor still applies: 0.9 survives, 0.1 does not.
        assert [c["id"] for c in kept] == ["a"]

    def test_a_finite_stricter_override_is_honoured(self):
        state = _make_state(
            _docs([("a", 0.9), ("b", 0.5)]),
            query_filters=to_json({"score_threshold": 0.7}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [c["id"] for c in kept] == ["a"]
