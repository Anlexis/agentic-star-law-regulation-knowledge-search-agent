# PB-6 — Invoke-Order Boundary: a full agent.invoke() must execute the fixed
# backbone in order.
#
# The outer backbone is fixed and is NEVER overridden by a template
# (add_edges() belongs to the framework):
#
#     START -> initialize -> pre_process -> main -> {route} -> post_process
#           -> finalize -> END
#
# The framework records every executed node in `node_history`, a state field
# whose reducer accumulates entries in execution order. Each entry is the node's
# CLASS NAME, appended as the node runs.
#
# This template is nested: the `main` slot is a GraphNode subclass
# (RegulationKnowledgeGraphNode) that delegates to the inner
# DomainWorkflowGraph. The inner graph runs with its own state, and its inner
# node_history is NOT merged back (merge_output() maps only regulation_answer /
# result / citations / status), so the OUTER node_history contains exactly the
# five backbone slots and never the inner domain nodes.
#
# This test drives a real end-to-end invoke() over the deployment sign-off
# payload and asserts the surfaced node_history matches the backbone order. A
# successful terminal status is required: on any other status route()
# short-circuits main -> finalize and the post_process slot is skipped, which is
# itself an invoke-order violation this test would catch.
#
# Trust context: InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
# the manifest's declared caller level. for_internal() is NEVER used here — it
# would over-privilege the run and hide regressions on the outer gate.
#
# ---------------------------------------------------------------------------
# REDACTION INTERACTION WITH THIS PAYLOAD (verified against the installed
# framework, not assumed):
#
# _VALID_PAYLOAD contains "Public Records Management Act" — four Title-Case
# words each carrying at least one lowercase letter. The framework's input gate
# treats that shape as a personal name and replaces it with "[MASKED]" on
# user_input and validated_input BEFORE PreProcessNode.execute() runs. A statute
# name and a person's name are the same shape, and the gate cannot tell them
# apart.
#
# This propagates: PreProcessNode reads the already-masked input, so
# validated_input carries "[MASKED]" where the act name was, and
# extract_input() hands that text to the inner graph as the search text.
# GenerateAnswerNode echoes the question into its lead sentence, so the final
# output carries "[MASKED]" inside the echoed question.
#
# It does NOT break the success path. The surviving tokens — "retention",
# "obligations", "administrative", "records" — still clear the relevance floor
# against the seeded corpus, so the status stays successful and node_history
# still traverses the full backbone in order. The Sources section and the inline
# citation markers are built from the corpus entries themselves, never from the
# query text, so the real act name ("Public Records and Archives Management
# Act") DOES appear in the output — just not inside the echoed question.
#
# These tests assert that observed shape rather than assuming an unmasked echo.
# The unit-level pin is in test_pre_process_node.py.
# ---------------------------------------------------------------------------
#
# docs/03_test_spec.md section 4.
# Deterministic — no model call, no network.
from typing import ClassVar
from framework.nodes.base_node import BaseNode

import json
import pathlib

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph

# --- TEMPLATE-SPECIFIC ------------------------------------------------------
# The `main`-slot GraphNode class name for THIS template. The other four
# backbone slot names are fixed by the framework and are identical in every
# template, so this is the only entry a sibling template would change.
_MAIN_SLOT_NODE = "RegulationKnowledgeGraphNode"

# The payload that drives the full workflow to a successful terminal status.
# It must stay byte-equal to the `input` field of the deployment sign-off
# payload — enforced by test_payload_matches_deploy_invoke_payload below. See
# the redaction note above: this payload IS partially masked before execute()
# runs, which is expected and asserted on explicitly.
_VALID_PAYLOAD = (
    "What are the retention obligations for administrative records under " "the Public Records Management Act?"
)

_DEPLOY_PAYLOAD_PATH = pathlib.Path(__file__).resolve().parents[2] / "deploy" / "invoke_payload.json"
# --- END TEMPLATE-SPECIFIC --------------------------------------------------

# The backbone execution order, by node class name as recorded in node_history.
# Four entries are fixed for every template; only _MAIN_SLOT_NODE varies.
_EXPECTED_ORDER = [
    "InitializeNode",  # framework default  (initialize slot)
    "PreProcessNode",  # the caller contract (pre_process slot)
    _MAIN_SLOT_NODE,  # TEMPLATE-SPECIFIC  (main slot GraphNode)
    "PostProcessNode",  # the output boundary (post_process slot)
    "FinalizeNode",  # framework default  (finalize slot)
]


class _PrivilegedTrustGateFixture(BaseNode):
    """Always-present privileged node used to prove the S-1 negative boundary."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _security_gate_input(self, state):
        return state

    def execute(self, state):
        return {"status": "success"}

    def _security_gate_output(self, result):
        return result


def _trust_predecessor(required: TrustLevel) -> TrustLevel:
    """Return a lower valid trust level; fail loudly if the framework adds one."""
    predecessors = {
        TrustLevel.VERIFIED_EXTERNAL: TrustLevel.ANONYMOUS,
        TrustLevel.INTERNAL: TrustLevel.VERIFIED_EXTERNAL,
    }
    try:
        return predecessors[required]
    except KeyError as exc:
        raise AssertionError(f"no lower trust level defined for {required!r}") from exc


def _run() -> dict:
    """Run a full end-to-end invocation at the manifest's declared trust level."""
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="pb6-suite")
    return Graph().invoke(_VALID_PAYLOAD, ctx=ctx)


class TestInvokeOrderBoundary:
    """PB-6: full agent.invoke() executes the backbone in the fixed order."""

    def test_payload_matches_deploy_invoke_payload(self):
        """The payload here and the deployment sign-off payload must be the
        same bytes, so this proves invoke order for the payload that is
        actually used to sign the deployment off."""
        deployed = json.loads(_DEPLOY_PAYLOAD_PATH.read_text(encoding="utf-8"))
        assert _VALID_PAYLOAD == deployed["input"]

    def test_invoke_reaches_success(self):
        """The full run must succeed — otherwise route() short-circuits
        main -> finalize and the post_process slot never runs. The redaction of
        the statute name (see module docstring) does not change that: the
        surviving query tokens still clear the relevance floor."""
        result = _run()
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected SUCCESS, got {result.get('status')!r}. result={result!r}"

    def test_output_is_non_empty(self):
        """A successful run must surface a non-empty gated output."""
        assert _run().get("output"), "invoke() surfaced an empty output"

    def test_node_history_is_populated(self):
        """node_history must be a non-empty list of node class-name strings."""
        history = _run().get("node_history")
        assert isinstance(history, list) and history, f"node_history must be a non-empty list, got {history!r}"
        assert all(isinstance(n, str) for n in history), f"node_history entries must be strings, got {history!r}"

    def test_backbone_slot_order(self):
        """Core invoke-order boundary: the pre_process slot runs before the
        domain main slot, which runs before the post_process slot — as a strict
        ordered subsequence of node_history."""
        history = _run().get("node_history", [])
        ordered_slots = ["PreProcessNode", _MAIN_SLOT_NODE, "PostProcessNode"]
        for name in ordered_slots:
            assert name in history, f"Expected backbone slot {name!r} in node_history, got {history!r}"
        positions = [history.index(name) for name in ordered_slots]
        assert positions == sorted(positions), (
            f"Backbone slots executed out of order: {ordered_slots} at {positions}. " f"node_history={history!r}"
        )

    def test_full_backbone_sequence(self):
        """The complete backbone order:
        initialize -> pre_process -> main -> post_process -> finalize."""
        history = _run().get("node_history", [])
        assert history == _EXPECTED_ORDER, (
            "node_history does not match the canonical backbone order.\n"
            f"  expected: {_EXPECTED_ORDER}\n"
            f"  actual:   {history}"
        )

    def test_s1_denial_refuses_execution_before_execute(self, monkeypatch):
        """TC-08: an always-present privileged node proves the negative S-1 path."""
        import framework.nodes.base_node as base_node_module

        events: list[str] = []
        execute_calls: list[object] = []
        monkeypatch.setattr(
            base_node_module,
            "emit_trace_event",
            lambda event_type, _payload, _state: events.append(event_type),
        )
        original_execute = _PrivilegedTrustGateFixture.execute

        def spy_execute(self, state):
            execute_calls.append(state)
            return original_execute(self, state)

        monkeypatch.setattr(_PrivilegedTrustGateFixture, "execute", spy_execute)
        result = _PrivilegedTrustGateFixture()(
            {
                "caller_trust_level": _trust_predecessor(_PrivilegedTrustGateFixture.required_trust_level).value,
                "correlation_id": "tc08-s1-denial",
            }
        )

        assert result["status"] == "error"
        assert "S-1 trust gate denied" in result["error_log"][0]
        assert events == ["s1_denied"]
        assert not execute_calls


class TestRedactionObservedBehaviour:
    """Pins the observed end-to-end shape of the redaction interaction described
    in the module docstring. It is a real property of this template with this
    payload, established by running it — not an assumption."""

    def test_masked_statute_name_appears_in_echoed_question(self):
        """The framework masks the Title-Case act name before
        PreProcessNode.execute() runs, and the answer echoes the question — so
        the raw phrase never survives into the output and [MASKED] stands in
        its place."""
        output = _run().get("output", "")
        assert "Public Records Management Act" not in output
        assert "[MASKED]" in output

    def test_real_statute_name_still_appears_via_kb_sourced_citations(self):
        """Despite the masked echo the answer is still correctly grounded: the
        real act name reaches the output through the corpus entries' own title
        and source fields, never through the query text, so the redaction does
        not compromise citation integrity."""
        output = _run().get("output", "")
        assert "Public Records and Archives Management Act" in output

    def test_citations_are_populated_despite_masking(self):
        """The structured citation list is still populated with real corpus
        entries — redaction reduces the query's token overlap but does not empty
        retrieval out for this payload."""
        result = _run()
        citations = result.get("citations")
        assert isinstance(citations, list) and citations
        for citation in citations:
            assert citation.get("id", "").startswith("gov-")
