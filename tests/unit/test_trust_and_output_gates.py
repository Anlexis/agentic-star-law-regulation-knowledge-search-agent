# GOV-C2-005 — Unit Tests: the trust gate and the output gate.
#
# Trust contract: tests invoke nodes via node(state), which runs the framework's
# trust check, its redaction pass, execute(), and its output check in order —
# never via node.execute(state), which would step around the trust check
# entirely. A denial RETURNS an error dict (it never raises) with an error
# status and "trust gate denied" in error_log; execute() never runs, so
# execute-only output keys are ABSENT from the returned dict.
#
# The `main` backbone slot is RegulationKnowledgeGraphNode, a GraphNode wrapping
# the inner five-node retrieval workflow whose domain nodes all admit ANONYMOUS
# callers. This file exercises InputValidateNode as the representative
# admits-anonymous case and asserts the declared level of all five; the
# per-node behavioural suites live in their own files.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.output_invariants import LEGAL_DISCLAIMER
from src.schemas.state import to_json

_CITATIONS = [
    {
        "ref": 1,
        "id": "gov-001",
        "title": "Retention of Administrative Documents",
        "source": "Act No. 66 of 2009, Art. 5",
    }
]

# A complete rendered answer: body with a citation marker, a Sources list, and
# the standing disclaimer — everything the output boundary requires.
_RENDERED_ANSWER = (
    "# Government Regulation Search Result\n\n"
    "[1] administrative documents must be retained per their classification.\n\n"
    "## Sources\n"
    "- [1] Retention of Administrative Documents (Act No. 66 of 2009, Art. 5)\n\n"
    "---\n\n"
    f"*{LEGAL_DISCLAIMER}*"
)


def _make_state(
    trust_value: str,
    user_input: str = "what are the retention obligations for administrative records under the Public Records Management Act?",
    **extra,
) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": trust_value,
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestTrustGate:
    """Trust gate tests — every invocation goes through node(state)."""

    def test_anonymous_caller_allowed_on_inner_domain_node(self):
        """An ANONYMOUS caller passes an ANONYMOUS inner domain node
        (InputValidateNode — the first node of the inner workflow)."""
        node = InputValidateNode()  # required_trust_level = ANONYMOUS
        result = node(_make_state(TrustLevel.ANONYMOUS.value, validated_input="test query"))
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("search_query") is not None

    def test_anonymous_caller_allowed_on_every_inner_domain_node(self):
        """All five inner domain nodes actually ADMIT an ANONYMOUS caller
        through node(state) — not merely declare that they do, which is what
        TestTrustLevelMatrix below checks. Each tolerates a near-empty state
        gracefully, so this proves the gate itself passes ANONYMOUS cleanly for
        every inner node. The external boundary lives on the outer
        pre_process / post_process slots."""
        cases = [
            (RetrieveNode(), {"search_query": "retention period"}),
            (RerankFilterNode(), {"retrieved_documents": "[]"}),
            (GenerateAnswerNode(), {"ranked_documents": "[]", "search_query": "retention period"}),
            (OutputFormatNode(), {"grounded_answer": "a grounded answer.", "citations": "[]"}),
        ]
        for node, extra in cases:
            result = node(_make_state(TrustLevel.ANONYMOUS.value, **extra))
            assert "trust gate denied" not in str(
                result.get("error_log", [])
            ), f"{node.__class__.__name__} unexpectedly denied an ANONYMOUS caller"

    def test_anonymous_caller_denied_on_pre_process(self):
        """Rejection: an ANONYMOUS caller on the VERIFIED_EXTERNAL PreProcessNode.

        __call__ must RETURN an error dict (never raise) with status ERROR and
        'trust gate denied' in the error_log. execute() never ran, so the
        execute-only output key (validated_input) must be ABSENT.
        """
        node = PreProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(_make_state(TrustLevel.ANONYMOUS.value))
        assert result.get("status") == AgentStatus.ERROR.value
        error_log = result.get("error_log", [])
        assert any(
            "trust gate denied" in str(e) for e in error_log
        ), f"Expected 'trust gate denied' in error_log, got: {error_log}"
        assert "validated_input" not in result, "execute() must not run on a trust denial — validated_input leaked"

    def test_verified_external_caller_passes_pre_process(self):
        """A VERIFIED_EXTERNAL caller clears the pre_process gate and the node
        writes validated_input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("validated_input")

    def test_pre_process_empty_input_rejected_after_gate(self):
        """The gate passes, then the node's own validation rejects empty input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, user_input=""))
        assert result.get("status") == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("empty" in str(e) for e in result.get("error_log", []))

    def test_anonymous_caller_denied_on_post_process(self):
        """Rejection on the other VERIFIED_EXTERNAL outer slot (post_process).

        The denial dict carries no execute-only key (formatted_output ABSENT).
        """
        node = PostProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                result=_RENDERED_ANSWER,
            )
        )
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("trust gate denied" in str(e) for e in result.get("error_log", []))
        assert "formatted_output" not in result, "execute() must not run on a trust denial — formatted_output leaked"

    def test_verified_external_caller_passes_post_process(self):
        """A VERIFIED_EXTERNAL caller clears the post_process gate.

        The result carried here is a complete rendered answer: past the trust
        gate the node still applies the output boundary, and an answer missing
        its citations or its disclaimer would be withheld on those grounds
        rather than on trust.
        """
        node = PostProcessNode()
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                result=_RENDERED_ANSWER,
                citations=to_json(_CITATIONS),
            )
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output")


class TestOutputGateNestedScan:
    """The output credential gate scans RECURSIVELY, not just top-level strings.

    A scan that only looked at a top-level string would miss a violation nested
    inside a dict or a list, which is exactly the shape the structured citation
    payload has.

    These exercise the module-level scan function directly, isolated from the
    gate plumbing covered above, plus one direct execute() call proving the node
    wires it in. The direct call here targets scan depth, not trust routing —
    the trust gate itself is covered exhaustively by TestTrustGate.
    """

    def test_credential_nested_in_dict_is_blocked(self):
        from src.nodes.post_process_node import _security_gate_output

        violation = _security_gate_output({"citation": "Act X Art. 3", "debug": {"api_key": "sk-1234567890abcdef1234"}})
        assert violation == "api_key"

    def test_credential_nested_in_list_is_blocked(self):
        from src.nodes.post_process_node import _security_gate_output

        violation = _security_gate_output(["clean line", {"note": "token: abcd1234efgh5678"}])
        assert violation == "credential_assignment"

    def test_clean_nested_structure_passes(self):
        from src.nodes.post_process_node import _security_gate_output

        violation = _security_gate_output(
            {"citation": "Act X Art. 3", "excerpt": "administrative records must be retained for..."}
        )
        assert violation is None

    def test_post_process_node_blocks_nested_violation_end_to_end(self):
        """execute() itself (called directly) must block a credential nested
        inside a dict result, not just a bare top-level string."""
        node = PostProcessNode()
        result = node.execute({"result": {"citation": "Act X Art. 3", "debug": {"api_key": "sk-1234567890abcdef1234"}}})
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result.get("error_log", []))


class TestTrustLevelMatrix:
    """The backbone's declared trust matrix (config/agent.yaml).

    The outer gate slots (pre_process / post_process) require
    VERIFIED_EXTERNAL, matching the manifest's declared required_trust_level.
    The `main` slot is RegulationKnowledgeGraphNode, a GraphNode that carries no
    level of its own, wrapping the inner workflow whose domain nodes all declare
    ANONYMOUS — the external boundary lives on the outer backbone.
    """

    def test_outer_gate_nodes_require_verified_external(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
        assert PostProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL

    def test_inner_domain_nodes_admit_anonymous(self):
        for node_cls in (
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
        ):
            assert node_cls.required_trust_level is TrustLevel.ANONYMOUS, (
                f"{node_cls.__name__} must declare TrustLevel.ANONYMOUS " "(inner domain node convention)"
            )
