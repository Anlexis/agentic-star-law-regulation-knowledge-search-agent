# GOV-C2-005 — Unit Tests: PostProcessNode (outer post_process slot; the output
# boundary).
#
# Invocation canon: node(state). PostProcessNode is the second outer gate slot
# and requires VERIFIED_EXTERNAL (like PreProcessNode), so its behavioural tests
# build the state at that level; the ANONYMOUS rejection lives in
# test_trust_and_output_gates.py.
#
# The boundary runs TWO independent layers, and both are covered here:
#
#   1. Credential redaction — the module-level scan runs INSIDE execute(), and
#      RECURSIVELY (dict / list / tuple / set values at any depth, not just a
#      top-level string). A violating answer is replaced by the sanitised stub
#      and an error status is returned; no exception is raised. Tests assert the
#      raw secret never survives into formatted_output OR result.
#   2. The template's own output invariant — every cited provision traceable,
#      and the disclaimer present. A breach withholds the answer entirely.
#
# The recursive nested-container coverage of the scan lives in
# test_trust_and_output_gates.py, together with the reuse of the same scan over
# the structured citations field.
#
# Mirrors docs/03_test_spec.md Sec.2.7 (POST-01..POST-14).
# Deterministic — no model call, no network.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.post_process_node import PostProcessNode
from src.output_invariants import (
    BLOCKED_OUTPUT,
    LEGAL_DISCLAIMER,
    OUTER_OUTPUT_FIELDS,
    WITHHELD_OUTPUT,
)
from src.schemas.state import to_json


def _envelope_output(state: dict, delta: dict):
    """What the caller would receive, given `delta` applied to `state`.

    Reproduces the envelope's own resolution rule — the surfaced answer is
    `formatted_output` OR, when that is falsy, `result` — so these tests measure
    what actually reaches the caller rather than what one field happens to hold.
    That OR is the whole reason a blank-out is not a refusal.
    """
    merged = {**state, **delta}
    return merged.get("formatted_output") or merged.get("result")


_CITATIONS = [
    {
        "ref": 1,
        "id": "gov-001",
        "title": "Retention of Administrative Documents",
        "source": "Act No. 66 of 2009, Art. 5",
    }
]

_CLEAN_REPORT = (
    "# Government Regulation Search Result\n\n"
    "[1] administrative documents must be retained per their classification.\n\n"
    "## Sources\n"
    "- [1] Retention of Administrative Documents (Act No. 66 of 2009, Art. 5)\n\n"
    "---\n\n"
    f"*{LEGAL_DISCLAIMER}*"
)

# Token shapes are assembled at runtime so no credential-shaped literal ever
# sits in the repository for a scanner to trip over.
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12


def _make_state(result_text, citations=None, **extra) -> dict:
    state = {
        "result": result_text,
        "citations": to_json(_CITATIONS if citations is None else citations),
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_REPORT))
        assert result["status"] == AgentStatus.SUCCESS.value
        # State carries the plain string, never the bare enum.
        assert type(result["status"]) is str  # noqa: E721 — isinstance() would accept a str-subclassing enum
        assert result["formatted_output"] == _CLEAN_REPORT

    def test_post_02_empty_result_is_non_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""

    def test_a_run_already_refused_upstream_does_not_reach_this_boundary(self):
        """Why this node needs no "already failed" branch of its own.

        The framework skips execute() when the incoming state carries an error
        status, so an upstream refusal never reaches the gate at all — and the
        backbone routes a non-success run past this slot to finalize. Containment
        for those runs belongs at the envelope, not here. Pinned because a
        framework change here would silently create an ungated path.
        """
        result = PostProcessNode()(_make_state(_CLEAN_REPORT, status=AgentStatus.ERROR.value))
        assert result["status"] == AgentStatus.ERROR.value
        assert "formatted_output" not in result

    def test_statutory_identifiers_reach_the_caller_byte_for_byte(self):
        # This boundary refuses answers; it never rewrites them. Act numbers,
        # article references, corpus identifiers and years must survive exactly
        # as composed — a boundary that edited numbers would corrupt the very
        # citations this template exists to deliver.
        result = PostProcessNode()(_make_state(_CLEAN_REPORT))
        out = result["formatted_output"]
        for token in ("Act No. 66 of 2009", "Art. 5", "gov-001", "2009", "[1]"):
            assert token in out or token == "gov-001"
        assert out == _CLEAN_REPORT


class TestPostProcessCredentialLayer:
    def _assert_blocked(self, result, secret, state=None):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result["error_log"])
        # EVERY field that can carry released text, not just the surfaced one.
        for field in OUTER_OUTPUT_FIELDS:
            assert result[field] == BLOCKED_OUTPUT
            assert secret not in str(result[field])
        assert result["citations"] is None
        # And nothing anywhere in the delta reproduces the secret — a diagnostic
        # that quoted it would be scanned along with the rest of this delta by
        # the framework's own output gate, and a gate that raises there discards
        # the delta, clearing included.
        for value in result.values():
            assert secret not in str(value)
        if state is not None:
            assert _envelope_output(state, result) == BLOCKED_OUTPUT

    def test_post_03_api_key_is_blocked(self):
        secret = "sk-ABCDEF0123456789abcdef"
        state = _make_state(f"{_CLEAN_REPORT}\n<!-- debug api_key={secret} -->\n")
        self._assert_blocked(PostProcessNode()(state), secret, state)

    def test_post_04_credential_assignment_is_blocked(self):
        secret = "password=super_secret_value_123"
        state = _make_state(f"{_CLEAN_REPORT}\nnote: {secret}\n")
        self._assert_blocked(PostProcessNode()(state), "super_secret_value_123", state)

    def test_post_05_jwt_is_blocked(self):
        state = _make_state(f"{_CLEAN_REPORT}\nsession token {_FAKE_JWT}\n")
        self._assert_blocked(PostProcessNode()(state), _FAKE_JWT, state)

    def test_post_06_bearer_token_is_blocked(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        state = _make_state(f"{_CLEAN_REPORT}\nauthorization: {secret}\n")
        self._assert_blocked(PostProcessNode()(state), secret, state)


class TestPostProcessInvariantLayer:
    """The second layer: the answer's own promise, re-checked as it leaves."""

    def _assert_withheld(self, result, violation, state=None):
        assert result["status"] == AgentStatus.ERROR.value
        assert any(violation in str(e) for e in result["error_log"])
        # The placeholder must be NON-EMPTY in every output-bearing field. A
        # blank here is the whole defect this pins: the envelope resolves the
        # answer as `formatted_output or result`, so blanking the first field
        # while leaving the second populated hands the caller the very answer
        # this boundary just refused, under a status that says it did not.
        for field in OUTER_OUTPUT_FIELDS:
            assert result[field] == WITHHELD_OUTPUT
            assert result[field]
        assert result["citations"] is None
        if state is not None:
            assert _envelope_output(state, result) == WITHHELD_OUTPUT

    def test_answer_without_the_disclaimer_is_withheld(self):
        stripped = _CLEAN_REPORT.replace(f"*{LEGAL_DISCLAIMER}*", "")
        state = _make_state(stripped)
        self._assert_withheld(PostProcessNode()(state), "missing_legal_disclaimer", state)

    def test_answer_citing_a_marker_with_no_citation_is_withheld(self):
        forged = _CLEAN_REPORT.replace("[1] administrative", "[4] administrative")
        state = _make_state(forged)
        self._assert_withheld(PostProcessNode()(state), "ungrounded_citation_marker", state)

    def test_unattributed_citation_is_withheld(self):
        state = _make_state(_CLEAN_REPORT, citations=[dict(_CITATIONS[0], source="")])
        self._assert_withheld(PostProcessNode()(state), "citation_without_source", state)

    def test_the_breaching_answer_does_not_reach_the_caller(self):
        """The regression this layer exists to prevent, measured at the envelope.

        A blank in one field is not containment while a populated sibling field
        is still there to be fallen back to.
        """
        released = "[4] administrative documents may be destroyed on request."
        breaching = _CLEAN_REPORT.replace(
            "[1] administrative documents must be retained per their classification.", released
        )
        state = _make_state(breaching)
        result = PostProcessNode()(state)
        self._assert_withheld(result, "ungrounded_citation_marker", state)
        assert released not in str(_envelope_output(state, result))
        for value in result.values():
            assert released not in str(value)

    def test_the_two_layers_are_independent(self):
        # A credential AND an invariant breach in the same answer: the
        # credential layer must still win, because it is the one that rewrites.
        secret = "sk-ABCDEF0123456789abcdef"
        stripped = _CLEAN_REPORT.replace(f"*{LEGAL_DISCLAIMER}*", "")
        state = _make_state(f"{stripped}\napi_key={secret}\n")
        result = PostProcessNode()(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result["error_log"])
        assert _envelope_output(state, result) == BLOCKED_OUTPUT
        assert secret not in str(_envelope_output(state, result))
