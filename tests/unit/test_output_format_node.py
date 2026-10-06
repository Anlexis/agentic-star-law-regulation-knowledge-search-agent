# GOV-C2-005 — Unit Tests: OutputFormatNode (inner domain node 5, terminal)
#
# Invocation canon: node(state), so the framework's own gates run around
# execute(). formatted_answer is a domain field; the standing legal-advice
# disclaimer is part of THIS node's output contract.
#
# This node is also the first of the two boundaries that enforce the template's
# output invariant, and the only one that can see the retrieved passage set — so
# the grounding tests that need that set live here.
#
# Mirrors docs/03_test_spec.md Sec.2.6 (FMT-01..FMT-05, GND-01..GND-07).
# Deterministic — no model call, no network.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import OutputFormatNode
from src.output_invariants import WITHHELD_OUTPUT
from src.schemas.state import to_json

_DISCLAIMER_FRAGMENT = "does not constitute legal advice or a formal regulatory determination"

_CITATION = {
    "ref": 1,
    "id": "gov-001",
    "title": "Retention of Administrative Documents",
    "source": "Act No. 66 of 2009, Art. 5",
}
_RANKED = [{"id": "gov-001", "title": _CITATION["title"], "score": 0.9}]


def _make_state(grounded_answer, citations, ranked=None, **extra) -> dict:
    state = {
        "grounded_answer": grounded_answer,
        "citations": citations,
        "ranked_documents": to_json(_RANKED if ranked is None else ranked),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestFormattedAnswer:
    def test_fmt_01_composes_header_body_sources_disclaimer(self):
        result = OutputFormatNode()(_make_state("[1] the grounded answer body.", to_json([_CITATION])))
        answer = result["formatted_answer"]
        assert answer.startswith("# Government Regulation Search Result")
        assert "[1] the grounded answer body." in answer
        assert "## Sources" in answer
        assert "- [1] Retention of Administrative Documents (Act No. 66 of 2009, Art. 5)" in answer
        assert _DISCLAIMER_FRAGMENT in answer
        assert result["status"] == AgentStatus.SUCCESS.value
        # State carries the plain string, never the bare enum.
        assert type(result["status"]) is str  # noqa: E721 — isinstance() would accept a str-subclassing enum

    def test_fmt_02_source_suffix_omitted_when_blank(self):
        # An unattributed citation is refused outright, so the "no suffix"
        # rendering is exercised through a citation whose source is present but
        # whose rendering has nothing to append.
        citation = dict(_CITATION, source="Act No. 66 of 2009")
        answer = OutputFormatNode()(_make_state("[1] body.", to_json([citation])))["formatted_answer"]
        assert "- [1] Retention of Administrative Documents (Act No. 66 of 2009)" in answer
        assert "()" not in answer

    def test_fmt_03_disclaimer_present_on_every_answer(self):
        # The disclaimer rides WITH the substance, never separately.
        for grounded in ("a body.", ""):
            answer = OutputFormatNode()(_make_state(grounded, to_json([]), ranked=[]))["formatted_answer"]
            assert _DISCLAIMER_FRAGMENT in answer


class TestDegradedInputs:
    def test_fmt_04_no_citations_renders_explicit_none_line(self):
        answer = OutputFormatNode()(_make_state("no coverage body.", to_json([]), ranked=[]))["formatted_answer"]
        assert "- none (no knowledge-base provision cleared the relevance threshold)" in answer

    def test_fmt_05_missing_grounded_answer_uses_fallback_text(self):
        state = _make_state("", to_json([]), ranked=[])
        del state["grounded_answer"]
        result = OutputFormatNode()(state)
        assert "No answer is available for this request." in result["formatted_answer"]
        assert result["status"] == AgentStatus.SUCCESS.value


class TestGroundingIsEnforcedNotAssumed:
    """The answer is refused whole when a cited provision cannot be traced.

    "Grounded by construction" holds only while the construction is unchanged.
    These cases are what the boundary catches if it ever is.
    """

    def _assert_withheld(self, result, violation):
        assert result["status"] == AgentStatus.ERROR.value
        assert any(violation in str(e) for e in result["error_log"])
        # Refusing is only half of it: the answer field has to end up holding
        # something NON-EMPTY. Leaving it absent or blank keeps it falsy, and
        # every consumer downstream reads a falsy answer as "look elsewhere"
        # rather than as "this was refused" — which is how a refused answer
        # gets shipped by the fallback further out.
        assert result["formatted_answer"] == WITHHELD_OUTPUT
        assert result["formatted_answer"]
        # The citation list is the other half of the answer and goes with it.
        assert result["citations"] is None

    def _assert_no_answer_text_survives(self, result, released):
        """Nothing the refused answer was built from may ride out in the delta."""
        for value in result.values():
            assert released not in str(value)

    def test_a_refused_answer_leaves_no_answer_text_in_the_delta(self):
        """The refusal names the rule, never the passage that broke it."""
        released = "administrative documents must be retained for ten years"
        result = OutputFormatNode()(_make_state(f"[7] {released}.", to_json([_CITATION])))
        self._assert_withheld(result, "ungrounded_citation_marker")
        self._assert_no_answer_text_survives(result, released)

    def test_citation_marker_with_no_backing_citation_is_refused(self):
        result = OutputFormatNode()(_make_state("[1] real point. [7] invented point.", to_json([_CITATION])))
        self._assert_withheld(result, "ungrounded_citation_marker")

    def test_citation_naming_a_passage_retrieval_never_returned_is_refused(self):
        fabricated = dict(_CITATION, id="gov-999", title="Invented Act")
        result = OutputFormatNode()(_make_state("[1] point.", to_json([fabricated])))
        self._assert_withheld(result, "citation_source_was_not_retrieved")

    def test_citation_without_a_source_is_refused(self):
        result = OutputFormatNode()(_make_state("[1] point.", to_json([dict(_CITATION, source="  ")])))
        self._assert_withheld(result, "citation_without_source")

    def test_citation_without_a_title_is_refused(self):
        result = OutputFormatNode()(_make_state("[1] point.", to_json([dict(_CITATION, title="")])))
        self._assert_withheld(result, "citation_without_title")

    def test_citations_when_nothing_was_retrieved_are_refused(self):
        result = OutputFormatNode()(_make_state("[1] point.", to_json([_CITATION]), ranked=[]))
        self._assert_withheld(result, "citation_source_was_not_retrieved")
