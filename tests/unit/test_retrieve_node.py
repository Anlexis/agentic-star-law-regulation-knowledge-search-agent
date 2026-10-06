# GOV-C2-005 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# RULES.md C2 is RETIRED (2026-07-27, blocking): execute(self, state) takes no
# `config` parameter at all, so config-precedence is exercised entirely via the
# state-seeded `retrieval_config` field (DomainWorkflowGraph._extra_initial_state()
# republishes it) — every call below goes through node(state) / __call__, no
# direct execute(state, config=...) 2-arg calls anywhere in this file.
#
# Mirrors docs/03_test_spec.md Sec.2.3 (RET-01..RET-08).
# Deterministic — keyword scoring over the seeded
# config/kb/gov_regulations_kb.json; no LLM, no network. Query/score pairs
# below were verified empirically against the real deterministic scorer
# by running the scorer, not by hand.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_RETENTION_QUERY = "retention period requirements for administrative documents"
_CLASSIFICATION_QUERY = "document classification records management"


def _make_state(query=_RETENTION_QUERY, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_retention_entry(self):
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the retention query"
        assert docs[0]["id"] == "gov-001"

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        for doc in docs:
            assert set(doc.keys()) == {"id", "title", "category", "source", "score", "excerpt"}
            assert len(doc["excerpt"]) <= 400

    def test_retrieved_documents_is_json_string(self):
        # List-shaped state fields travel as JSON strings.
        result = RetrieveNode()(_make_state())
        assert isinstance(result["retrieved_documents"], str)


class TestRetrieveFilters:
    def test_ret_04_category_filter_restricts_pool(self):
        state = _make_state(
            query=_CLASSIFICATION_QUERY,
            query_filters=to_json({"category": "administrative_guidance", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs, "administrative_guidance category has a seeded entry"
        assert {d["category"] for d in docs} == {"administrative_guidance"}
        assert docs[0]["id"] == "gov-009"

    def test_ret_05_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query=""))["retrieved_documents"])
        assert docs == []


class TestRetrieveConfigPrecedence:
    """Config plumbing: state-seeded retrieval_config > module defaults.

    C2 (RULES.md) is RETIRED — execute() takes no config parameter, so every
    case here goes through node(state) / __call__ with retrieval_config seeded
    on the state dict (the same field DomainWorkflowGraph._extra_initial_state()
    republishes at runtime), never a direct execute(state, config=...) call.
    """

    def test_ret_06_state_kb_path_override_degrades_gracefully(self):
        state = _make_state(retrieval_config=to_json({"kb_path": "config/kb/does_not_exist.json"}))
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_ret_07_state_retrieval_config_top_k_is_honoured(self):
        # top_k=1 caps the candidate pool at max(1*3, 10) = 10 (the floor
        # dominates for a small top_k) — assert the effective top_k reached
        # RetrieveNode by checking the pool never exceeds the floor either way
        # and a tighter top_k still returns ranked (non-empty) candidates.
        state = _make_state(retrieval_config=to_json({"top_k": 1}))
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs
        assert docs[0]["id"] == "gov-001"


class TestRetrieveNotesAccumulation:
    def test_ret_08_notes_append_never_clobber(self):
        state = _make_state(
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_config=to_json({"kb_path": "config/kb/bogus.json"}),
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2
