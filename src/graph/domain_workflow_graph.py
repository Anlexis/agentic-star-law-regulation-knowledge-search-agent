"""AgentCore Platform v1.0"""

# GOV-C2-005 - DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the full government-regulation KB search domain workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Called by RegulationKnowledgeGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology, no forced backbone)
#   - Implements every BaseGraph abstract method
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - get_output() designed together with RegulationKnowledgeGraphNode.merge_output()

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.graph.context_bridge import read_caller_options
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for GOV-C2-005.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by RegulationKnowledgeGraphNode.get_subgraph() in graph.py, which
    passes the manifest-derived config (`_parent_config()`) into the ctor.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  - parse + normalise the query
          -> retrieve        (RetrieveNode)       - keyword-score the seeded KB
          -> rerank_filter   (RerankFilterNode)   - boost / threshold / top_k cut
          -> generate_answer (GenerateAnswerNode) - grounded answer + citations
          -> output_format   (OutputFormatNode)   - final format + legal disclaimer
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns - not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "gov_c2_005_regulation_knowledge_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate inner graph config before compilation.

        The forwarded `retrieval` block (top_k / score_threshold / kb_path) is
        read per-call by the domain nodes with safe defaults, so absence is
        non-fatal. Validation is permissive here rather than raising
        ConfigError.
        """
        pass

    # -- Config forwarding into state (manifest -> inner nodes) -----------------

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the inner state with the tuning and the caller's options.

        Two things the domain nodes need cannot reach them by themselves:

        retrieval_config  The `retrieval` block that
                          RegulationKnowledgeGraphNode._parent_config() read from
                          config/config.yaml and forwarded under
                          config["configurable"]. Republished here as a state
                          field because execute() takes no config parameter.

        caller_options    The validated caller options. The framework hands the
                          inner graph only the question string, so options taken
                          from the invocation's context mapping arrive through
                          the context bridge (src/graph/context_bridge.py) —
                          without it a caller-selected category or passage cap
                          would be dropped in silence and the caller would get
                          the default answer back as though it had been applied.

        Both are stored as JSON strings rather than bare containers: inner state
        is checkpointed, and a bare dict or list in a checkpointed field is not
        safe to serialise.
        """
        retrieval = (self.config or {}).get("configurable", {}).get("retrieval") or {}
        return {
            "retrieval_config": to_json(retrieval),
            "caller_options": to_json(read_caller_options()),
        }

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments — FunctionNode
        subclasses take no __init__; config flows in via the state fields seeded
        by _extra_initial_state() above, never a per-call execute() parameter.
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear GOV regulation search domain topology.

        Each step passes its partial-dict output into the shared State.
        For this template the topology is intentionally linear — no conditional
        branching between domain nodes. route() is implemented as required by
        the abstract base but add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by the BaseGraph abstract base.

        For this linear topology add_conditional_edges() is not used, so this
        method is never called at runtime. It is implemented to satisfy the
        contract. Returns END on error so an unexpected call does not re-enter a
        processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by RegulationKnowledgeGraphNode.merge_output() in
        graph.py as the `sub_result` argument. Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output()  emits: "formatted_answer", "citations", "status", ...
            Outer merge_output() reads: sub_result.get("formatted_answer"),
                                        sub_result.get("citations"),
                                        sub_result.get("status")

        Additional fields (intake_notes, trace_id, correlation_id,
        node_history) are surfaced for observability / downstream extension.
        merge_output() currently maps formatted_answer + citations + status
        into the outer state delta; the remaining fields are available for
        future outer-merge extensions without an inner-graph change.
        """
        return {
            "formatted_answer": state.get("formatted_answer"),
            "citations": state.get("citations"),
            "status": state.get("status"),
            # Carried explicitly: the boundary only moves the keys named here,
            # so a run that completed without an answer would otherwise arrive
            # at the outer graph indistinguishable from one that answered.
            "error_code": state.get("error_code"),
            "intake_notes": state.get("intake_notes"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
