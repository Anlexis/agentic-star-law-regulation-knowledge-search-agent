# GOV-C2-005 — Unit Tests: retrieval quality over the seeded KB
#
# Golden-query suite: drives the REAL inner retrieval chain
# (InputValidateNode -> RetrieveNode -> RerankFilterNode) via node(state) /
# __call__ (ANONYMOUS inner nodes) against
# config/kb/gov_regulations_kb.json and pins the expected top hit per domain
# query — one query per seeded KB entry (gov-001..gov-012). The scorer is
# deterministic (keyword field-weights, stable tie-break), so exact top-1
# assertions are safe and catch KB / scorer / threshold regressions. Every
# query/expected-id pair was verified empirically against the real
# deterministic scorer, not hand-computed.
#
# Category-filtered searches route the category through the plain-text JSON
# envelope InputValidateNode parses (`{"query": ..., "category": ...}`), NOT
# by pre-seeding query_filters directly — InputValidateNode overwrites
# query_filters from its own parse of the raw payload, so a directly-seeded
# value would be clobbered before RetrieveNode ever sees it.
#
# Mirrors docs/03_test_spec.md Sec.2.9 (QUAL-01..QUAL-07).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json
import pathlib

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KB_IDS = {
    entry["id"]
    for entry in json.loads((_ROOT / "config" / "kb" / "gov_regulations_kb.json").read_text(encoding="utf-8"))
}

_DEFAULT_SCORE_THRESHOLD = 0.25  # mirrors config/agent.yaml retrieval block


def _search(payload: str) -> list[dict]:
    """Run the real inner retrieval chain and return the surviving passages."""
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "quality-session",
        "execution_time": {},
    }
    state.update(InputValidateNode()(state))
    state.update(RetrieveNode()(state))
    state.update(RerankFilterNode()(state))
    return from_json(state["ranked_documents"], [])


# (query, expected top-1 KB entry id) — one per seeded entry, verified against
# the deterministic scorer.
_GOLDEN_QUERIES = [
    ("retention period requirements for administrative documents", "gov-001"),
    ("rules for creating and classifying administrative document management records", "gov-002"),
    ("deadline for an administrative organ to decide a disclosure request", "gov-003"),
    ("grounds for withholding a document from an information disclosure request", "gov-004"),
    ("purpose limitation on use of personal information held by administrative organs", "gov-005"),
    ("standard processing period for an application to an administrative organ", "gov-006"),
    ("reasons an administrative organ must give when refusing an application", "gov-007"),
    ("retention period categories set by cabinet order for administrative documents", "gov-008"),
    ("classification of a document at the time of its creation", "gov-009"),
    ("retention obligations for electronic administrative records", "gov-010"),
    ("confidentiality duty of a public official after leaving service", "gov-011"),
    ("disclosure request rights against independent administrative agencies", "gov-012"),
]


class TestGoldenQueries:
    @pytest.mark.parametrize(("query", "expected_id"), _GOLDEN_QUERIES)
    def test_qual_01_top_hit_per_golden_query(self, query, expected_id):
        kept = _search(query)
        assert kept, f"no passage cleared the relevance floor for: {query!r}"
        assert kept[0]["id"] == expected_id

    def test_qual_02_all_survivors_clear_the_relevance_floor(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["score"] >= _DEFAULT_SCORE_THRESHOLD

    def test_qual_03_survivor_ids_exist_in_the_seeded_kb(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["id"] in _KB_IDS


class TestPrecision:
    def test_qual_04_national_security_query_keeps_only_the_exemption_entry(self):
        # Off-topic passages score below the floor and are cut — precision, not
        # just recall.
        kept = _search("why would a request for information be denied on national security grounds")
        assert [d["id"] for d in kept] == ["gov-004"]

    def test_qual_05_category_filter_restricts_to_that_category(self):
        payload = json.dumps(
            {"query": "retention period categories for administrative documents", "category": "cabinet_order"}
        )
        kept = _search(payload)
        assert kept, "cabinet_order category carries a seeded entry"
        assert {d["category"] for d in kept} == {"cabinet_order"}
        assert kept[0]["id"] == "gov-008"


class TestNoCoverage:
    def test_qual_06_out_of_domain_query_yields_no_survivors(self):
        assert _search("quantum telepathy sandwich recipes") == []

    def test_qual_07_no_coverage_produces_the_escalation_answer(self):
        state = {
            "ranked_documents": "[]",
            "search_query": "quantum telepathy sandwich recipes",
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "quality-session",
            "execution_time": {},
        }
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []
