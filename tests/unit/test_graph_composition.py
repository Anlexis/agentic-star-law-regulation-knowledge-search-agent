# GOV-C2-005 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (GovernmentRegulationKnowledgeAgent / Graph)
# end-to-end via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide regressions on the outer trust gate).
#
# The happy-path query here is deliberately free of anything the platform's
# redaction pass recognises, so these assertions are not entangled with that
# interaction — the canonical payload (which DOES trip the Title-Case name
# heuristic) is exercised end to end in
# tests/proof_of_boundary/test_pb_invoke_order.py, which asserts the observed
# behaviour: still successful, still correctly cited, with the redaction marker
# in the echoed question.
#
# Mirrors docs/03_test_spec.md Sec.3 (INT-05..INT-12, CNT-01..CNT-06).
# Deterministic — no model call, no network.

import pathlib

import yaml
from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    GovernmentRegulationKnowledgeAgent,
    Graph,
    RegulationKnowledgeGraphNode,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.output_invariants import OUTER_OUTPUT_FIELDS, WITHHELD_OUTPUT
from src.schemas.state import State, from_json, to_json

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RETENTION_QUERY = "retention period requirements for administrative documents"


def _run(user_input: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph().invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_int_05_inherits_agent_base_graph_directly(self):
        assert issubclass(GovernmentRegulationKnowledgeAgent, AgentBaseGraph)

    def test_int_05_graph_alias(self):
        assert Graph is GovernmentRegulationKnowledgeAgent

    def test_state_schema_is_state(self):
        assert GovernmentRegulationKnowledgeAgent().state_schema is State

    def test_int_06_compile_fills_all_backbone_slots(self):
        agent = GovernmentRegulationKnowledgeAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], RegulationKnowledgeGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in GovernmentRegulationKnowledgeAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_07_get_subgraph_returns_the_inner_graph(self):
        subgraph = RegulationKnowledgeGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_int_08_extract_input_prefers_validated_input(self):
        node = RegulationKnowledgeGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_09_merge_output_maps_the_inner_contract(self):
        node = RegulationKnowledgeGraphNode()
        citations = to_json([{"ref": 1, "id": "gov-001", "title": "t", "source": "s"}])
        delta = node.merge_output(
            {},
            {"formatted_answer": "ANSWER", "citations": citations, "status": AgentStatus.SUCCESS.value},
        )
        # The inner formatted_answer surfaces as BOTH regulation_answer and
        # result (PostProcessNode reads state["result"]).
        assert delta == {
            "regulation_answer": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "status": AgentStatus.SUCCESS.value,
            # Always present so a reason can never be dropped at the boundary.
            "error_code": "",
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert RegulationKnowledgeGraphNode.error_strategy == "propagate"
        assert RegulationKnowledgeGraphNode.propagate_hitl is False

    def test_int_10_parent_config_reads_the_runtime_config_file(self):
        """The forwarded tuning comes from config/config.yaml, live.

        Reading the manifest instead would return nothing after the manifest
        became a flat registry entry, and the pipeline would quietly run on its
        module defaults while the deployment believed it had configured tuning.
        """
        runtime = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
        cfg = RegulationKnowledgeGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"] == runtime["retrieval"]

    def test_int_10b_parent_config_never_forwards_an_empty_block(self):
        # The entry point resolves the runtime config and threads it in at
        # register_nodes() time. When nothing reaches the node, the forwarded
        # config degrades to the shipped tuning, never to {}.
        cfg = RegulationKnowledgeGraphNode(runtime_config={})._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_path"] == "config/kb/gov_regulations_kb.json"
        assert cfg["configurable"]["retrieval"]["top_k"] == 4


class TestNonSuccessRunsCarryNoAnswer:
    """Containment at the two graph-level boundaries.

    An error status is a label. What decides whether an answer was actually
    withheld is what the output-bearing fields hold when the envelope resolves
    them, so these assert on that rather than on the status.
    """

    _RELEASED = "# Government Regulation Search Result\n\nreleased answer text."

    def _envelope(self, state):
        agent = GovernmentRegulationKnowledgeAgent()
        return agent.get_output(state)

    def test_merge_output_withholds_when_the_inner_run_did_not_succeed(self):
        """No inner answer means every outer field gets the placeholder.

        Reachable for any non-success inner outcome the framework hands back
        rather than raising on (a cancelled or timed-out inner run, a resumed
        one), and the guard that keeps a later pass from inheriting an earlier
        pass's answer.
        """
        delta = RegulationKnowledgeGraphNode().merge_output(
            {},
            {"formatted_answer": None, "citations": to_json([{"ref": 1}]), "status": AgentStatus.TIMEOUT.value},
        )
        assert delta["status"] == AgentStatus.ERROR.value
        for field in OUTER_OUTPUT_FIELDS:
            assert delta[field] == WITHHELD_OUTPUT
            assert delta[field]
        assert delta["citations"] is None

    def test_a_non_success_envelope_never_carries_answer_text(self):
        """The last line of defence, where the answer is finally resolved.

        Answer text still sitting in `result` under an error status was written
        by something that was afterwards overruled. Shipping it because the
        status says error releases it exactly as surely as shipping it because
        the status says success.
        """
        envelope = self._envelope({"status": AgentStatus.ERROR.value, "result": self._RELEASED})
        assert envelope["output"] == WITHHELD_OUTPUT
        assert self._RELEASED not in str(envelope["output"])

    def test_a_boundary_placeholder_keeps_its_own_wording(self):
        """A refusal that already said something specific is not overwritten."""
        blocked = "[OUTPUT BLOCKED — disallowed content detected.]"
        envelope = self._envelope({"status": AgentStatus.ERROR.value, "formatted_output": blocked})
        assert envelope["output"] == blocked

    def test_a_run_that_produced_nothing_still_reports_nothing(self):
        """Absent output stays absent — there is nothing to withhold."""
        envelope = self._envelope({"status": AgentStatus.ERROR.value})
        assert not envelope["output"]

    def test_a_credential_shaped_citation_field_withholds_the_answer_too(self):
        """The citation gate withholds the rendered answer, not just the field.

        The citation list is not a footnote on the answer, it is half of it —
        an envelope reporting an error while still carrying the other half has
        released the product.
        """
        state = {
            "status": AgentStatus.SUCCESS.value,
            "result": self._RELEASED,
            # Assembled at runtime so no credential-shaped literal is stored here.
            "citations": to_json([{"ref": 1, "id": "ak-" + "A" * 20, "title": "t", "source": "s"}]),
        }
        envelope = self._envelope(state)
        assert envelope["status"] == AgentStatus.ERROR.value
        assert "citations" not in envelope
        assert envelope["output"] == WITHHELD_OUTPUT
        assert self._RELEASED not in str(envelope["output"])

    def test_a_malformed_citation_withholds_the_answer_too(self):
        state = {
            "status": AgentStatus.SUCCESS.value,
            "result": self._RELEASED,
            "citations": to_json([{"ref": 1, "id": "gov-001", "title": "", "source": "s"}]),
        }
        envelope = self._envelope(state)
        assert envelope["status"] == AgentStatus.ERROR.value
        assert "citations" not in envelope
        assert envelope["output"] == WITHHELD_OUTPUT

    def test_a_clean_success_is_untouched(self):
        """Control: the guard costs a good answer nothing."""
        state = {
            "status": AgentStatus.SUCCESS.value,
            "result": self._RELEASED,
            "citations": to_json([{"ref": 1, "id": "gov-001", "title": "t", "source": "s"}]),
        }
        envelope = self._envelope(state)
        assert envelope["status"] == AgentStatus.SUCCESS.value
        assert envelope["output"] == self._RELEASED
        assert envelope["citations"][0]["id"] == "gov-001"


class TestGetOutputStructuredCitations:
    """The structured citation list is extended onto the base envelope — only
    on SUCCESS, and only after it is re-scanned at the output boundary."""

    def test_citations_surfaced_on_success(self):
        result = _run(_RETENTION_QUERY)
        assert result.get("status") == AgentStatus.SUCCESS.value
        citations = result.get("citations")
        assert isinstance(citations, list) and citations
        assert citations[0]["id"] == "gov-001"

    def test_citations_absent_on_non_success(self):
        result = _run(_RETENTION_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert "citations" not in result


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_int_11_invoke_returns_success(self):
        result = _run(_RETENTION_QUERY)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_11_output_is_the_gated_formatted_answer(self):
        output = _run(_RETENTION_QUERY).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Government Regulation Search Result")
        assert "[1]" in output
        assert "does not constitute legal advice or a formal regulatory determination" in output

    def test_int_11_e2e_traverses_the_post_process_gate(self):
        history = _run(_RETENTION_QUERY).get("node_history", [])
        for cls_name in ("PreProcessNode", "RegulationKnowledgeGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_no_coverage_query_still_terminates_success(self):
        result = _run("quantum telepathy sandwich recipes")
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result.get("output", "")

    def test_int_12_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """Trust at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. The error state short-circuits the
        main slot (it sees the error status and skips the inner graph)
        and routes past post_process to finalize — no domain answer is ever
        produced."""
        result = _run(_RETENTION_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStateRoundTrip:
    """State helpers: producers to_json() on write, consumers from_json()."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "gov-001", "score": 0.7, "title": "administrative document retention"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"category": "act", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
