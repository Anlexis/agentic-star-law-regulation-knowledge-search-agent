# GOV-C2-005 — Unit Tests: GenerateAnswerNode (inner domain node 4)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# The grounded answer and citations are domain fields, not inputs, so
# Title-Case statute titles inside them are safe to assert on — the redaction pass
# only ever touches user_input / validated_input / llm_response.
#
# Mirrors docs/03_test_spec.md Sec.2.5 (GEN-01..GEN-05).
# Deterministic — rule-assembled from ranked_documents only (grounded by
# construction; no LLM, no network). framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.schemas.state import from_json, to_json


def _ranked(*entries):
    return to_json(list(entries))


def _doc(doc_id, title, excerpt, source="seeded kb"):
    return {
        "id": doc_id,
        "title": title,
        "category": "act",
        "source": source,
        "score": 0.9,
        "excerpt": excerpt,
    }


def _make_state(ranked_documents, query="retention period for administrative documents", **extra) -> dict:
    state = {
        "ranked_documents": ranked_documents,
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestGroundedAnswer:
    def test_gen_01_answer_carries_numbered_citation_markers(self):
        ranked = _ranked(
            _doc(
                "gov-001",
                "Public Records and Archives Management Act — Retention of Administrative Documents",
                "documents must be retained per their classification.",
            ),
            _doc(
                "gov-008",
                "Cabinet Order for Enforcement of the Public Records and Archives Management Act",
                "sets standard retention-period categories.",
            ),
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        answer = result["grounded_answer"]
        assert "[1] Public Records and Archives Management Act — Retention of Administrative Documents:" in answer
        assert "[2] Cabinet Order for Enforcement of the Public Records and Archives Management Act:" in answer

    def test_gen_02_lead_sentence_quotes_the_query(self):
        ranked = _ranked(_doc("gov-001", "Retention of Administrative Documents", "excerpt."))
        result = GenerateAnswerNode()(_make_state(ranked, query="retention period for administrative documents"))
        assert 'the question: "retention period for administrative documents"' in result["grounded_answer"]

    def test_gen_03_citations_mirror_ranked_order(self):
        ranked = _ranked(
            _doc("gov-001", "Retention of Administrative Documents", "a.", source="Act No. 66 of 2009, Art. 5"),
            _doc("gov-008", "Cabinet Order Retention Categories", "b."),
        )
        citations = from_json(GenerateAnswerNode()(_make_state(ranked))["citations"])
        assert [c["ref"] for c in citations] == [1, 2]
        assert [c["id"] for c in citations] == ["gov-001", "gov-008"]
        assert citations[0]["source"] == "Act No. 66 of 2009, Art. 5"

    def test_citations_is_json_string(self):
        # List-shaped state fields travel as JSON strings.
        ranked = _ranked(_doc("gov-001", "Retention of Administrative Documents", "a."))
        result = GenerateAnswerNode()(_make_state(ranked))
        assert isinstance(result["citations"], str)

    def test_gen_04_answer_is_grounded_in_ranked_passages_only(self):
        ranked = _ranked(
            _doc(
                "gov-001",
                "Retention of Administrative Documents",
                "permanent retention documents transfer to the National Archives.",
            )
        )
        answer = GenerateAnswerNode()(_make_state(ranked))["grounded_answer"]
        # Every content line traces to the single ranked passage.
        assert "permanent retention documents transfer to the National Archives." in answer
        assert "[2]" not in answer


class TestNoCoverage:
    def test_gen_05_empty_ranked_set_yields_no_coverage_answer(self):
        result = GenerateAnswerNode()(_make_state(_ranked()))
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []

    def test_missing_ranked_field_is_treated_as_no_coverage(self):
        state = _make_state(None)
        del state["ranked_documents"]
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]
