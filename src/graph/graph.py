"""AgentCore Platform v1.0"""

# GOV-C2-005 — outer graph (Category 2, two-layer nested architecture).
#
# Government Regulation Knowledge Agent: retrieval-grounded question answering
# over a seeded corpus of acts, cabinet orders, ministerial ordinances and
# administrative guidance.
#
# Architecture:
#
#   Outer backbone (fixed — do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (retry)
#                                             -> pre_process
#
#   The `main` slot is a GraphNode subclass (RegulationKnowledgeGraphNode) that
#   delegates the whole regulation-search workflow to DomainWorkflowGraph (the
#   inner graph: input_validate -> retrieve -> rerank_filter -> generate_answer
#   -> output_format).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- carries validated caller options inward
#
# Class-name contract:
#   graph.py class:           GovernmentRegulationKnowledgeAgent (this file)
#   config/agent.yaml class:  "src.graph.graph.GovernmentRegulationKnowledgeAgent"
#   src/api/server.py import: from src.graph.graph import GovernmentRegulationKnowledgeAgent
#
# Rules enforced:
#   - GovernmentRegulationKnowledgeAgent inherits AgentBaseGraph (direct
#     framework inheritance)
#   - super().register_nodes() called first (fills initialize + finalize)
#   - RegulationKnowledgeGraphNode assigned to self._nodes["main"]
#   - _parent_config() forwards live retrieval tuning read from
#     config/config.yaml (never an empty block)
#   - merge_output() returns only changed keys
#   - get_output() EXTENDS super().get_output() — structured `citations`
#     surfaced ONLY on SUCCESS, and only after the output boundary checks pass
#   - add_edges() NOT overridden on the outer graph

from typing import TYPE_CHECKING, Any, ClassVar, Dict

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import publish_caller_options
from src.nodes.post_process_node import PostProcessNode, _security_gate_output
from src.nodes.pre_process_node import PreProcessNode
from src.output_invariants import (
    OUTER_OUTPUT_FIELDS,
    WITHHELD_OUTPUT,
    check_citations_wellformed,
    is_withheld,
    withheld_delta,
)
from src.schemas.state import State, from_json

if TYPE_CHECKING:
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

# Used only when config/config.yaml cannot be read at all. The values mirror the
# tuning shipped in that file, so an unreadable config degrades to the shipped
# behaviour rather than to an empty block that would silently retune retrieval.
_FALLBACK_RETRIEVAL: Dict[str, Any] = {
    "top_k": 4,
    "score_threshold": 0.25,
    "kb_path": "config/kb/gov_regulations_kb.json",
}


class RegulationKnowledgeGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph, the inner retrieval pipeline. Called by the
    backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    instantiate DomainWorkflowGraph with the forwarded
                        runtime config (_parent_config())
      extract_input()   pull the redacted question out of outer state, and
                        publish the validated caller options for the inner graph
      merge_output()    map sub_result fields into the outer state delta
                        (changed keys only)
      error_strategy    "propagate": re-raise inner errors as SubgraphError
    """

    def __init__(self, runtime_config: dict[str, Any] | None = None) -> None:
        """Receive the runtime config from the outer graph.

        A BaseNode has no config back-reference of its own, so the outer
        AgentBaseGraph reads `self.config` and threads it in here at
        register_nodes() time. Static construction input - not mutable state.
        """
        # A non-mapping runtime config degrades to {} instead of raising: reading and
        # parsing config/config.yaml belongs to the entry point, and this node only has
        # to survive whatever it is handed.
        self._runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}

    # "propagate": re-raise inner graph exceptions as SubgraphError (fail fast).
    # "handle": call on_subgraph_error() instead — for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: human-in-the-loop interrupts stay inside the inner graph. This
    # template does not use them at all.
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the live retrieval tuning to the inner graph.

        Reads config/config.yaml — the runtime-parameter file — and returns its
        `retrieval` block under config["configurable"]. The inner graph
        republishes that block into inner state so RetrieveNode and
        RerankFilterNode read live values instead of dead declarations.

        The manifest (config/agent.yaml) is deliberately NOT read here: it is a
        registry entry and carries no tuning, so reading it would return nothing
        and the pipeline would run on its module defaults while the deployment
        believed it had configured something.
        """
        runtime = self._runtime_config
        retrieval = runtime.get("retrieval")
        if not isinstance(retrieval, dict) or not retrieval:
            retrieval = dict(_FALLBACK_RETRIEVAL)
        return {"configurable": {"retrieval": dict(retrieval)}}

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported inside the method to avoid a circular
        import at module load time.

        The inner graph receives the runtime config through its constructor
        (graph-level injection of immutable config). Its domain NODES still take
        no constructor arguments and read config exclusively from state:
        execute(self, state) -> dict, with no config parameter.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to work on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.

        This override is deliberate: GraphNode.execute() is not final, and the
        marker is the only signal that distinguishes "nothing to do" from "not
        run yet".
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: Dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the string handed to inner_graph.invoke(), and bridge options.

        PreProcessNode has already validated and redacted the question and has
        parsed the caller's options into an inert, bounded mapping. The string
        return value is the framework's only channel into the inner graph, so
        the options travel through the context bridge instead — without it a
        caller asking for one category, or for a single passage, would silently
        receive the default unfiltered answer.
        """
        publish_caller_options(from_json(state.get("caller_options"), {}) or {})
        return str(state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner graph's sub_result back into the outer state delta.

        Returns ONLY changed keys — never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "formatted_answer", "citations", "status", ...
          This merge_output() reads -> sub_result.get("formatted_answer"),
                                       sub_result.get("citations"),
                                       sub_result.get("status")

        regulation_answer: the final rendered answer, written by OutputFormatNode
          inside the inner graph.
        result: PostProcessNode reads state.get("result"), so the rendered answer
          is mapped there as well; otherwise the final output that leaves the
          agent — and the boundary checks over it — would see nothing.
        citations: re-surfaced at the outer layer (still a JSON string) so
          get_output() can read it without reaching into the inner graph.
        status: terminal status value from the inner graph run.

        A non-success inner run has no answer to forward, and this is the last
        point where that can be stated positively. The withheld placeholder is
        written over every outer output field rather than mapping the inner
        graph's absent answer through as None: None is falsy, and the envelope
        resolves the caller's answer as `formatted_output or result`, so a falsy
        answer means "keep looking" — at anything a retry pass or another node
        happens to have left behind. Writing something non-empty is what ends
        the search. It also covers the case where the inner terminal node's
        delta was discarded by the framework's own output gate: no answer
        arrives, and none should.
        """
        status = sub_result.get("status")
        answer = sub_result.get("formatted_answer")

        # A request that could not be accepted as written produces no answer on
        # purpose. Without this branch the "no answer" test below would convert
        # it into a terminal failure, which is exactly what the marker exists to
        # avoid: the caller would lose the conversation and see only an
        # exception type instead of what to correct. The outer reason wins - a
        # reason settled before the inner run is the real one, while the inner
        # graph only sees its downstream consequence.
        marker = state.get("error_code") or sub_result.get("error_code", "")
        if marker:
            return {
                **withheld_delta(WITHHELD_OUTPUT, OUTER_OUTPUT_FIELDS),
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
            }

        if status != AgentStatus.SUCCESS.value or not answer:
            return {
                **withheld_delta(WITHHELD_OUTPUT, OUTER_OUTPUT_FIELDS),
                "status": AgentStatus.ERROR.value,
            }
        return {
            "regulation_answer": answer,
            "result": answer,
            "citations": sub_result.get("citations"),
            "status": status,
            # Always present so a reason can never be dropped at the boundary.
            "error_code": "",
        }


class GovernmentRegulationKnowledgeAgent(AgentBaseGraph):
    """Outer graph for GOV-C2-005 (Category 2 retrieval pipeline, nested).

    Inherits AgentBaseGraph directly. Domain logic is fully encapsulated in
    RegulationKnowledgeGraphNode (main slot), which delegates to
    DomainWorkflowGraph.

    Backbone (fixed):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() and get_output() are the ONLY overrides:
      - super().register_nodes() fills initialize and finalize
      - pre_process:  PreProcessNode (owns the caller contract)
      - main:         RegulationKnowledgeGraphNode
      - post_process: PostProcessNode (output boundary)
      - get_output(): extends the base envelope with the structured `citations`
        field, only on SUCCESS and only when it passes the boundary checks

    add_edges() is NOT overridden — backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the platform registry."""
        return "GovernmentRegulationKnowledgeAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default initialize node (schema_version, session_id, trust
        level) and finalize node (response metadata, total time).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = RegulationKnowledgeGraphNode(runtime_config=self.config)
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Extend the base envelope with the structured citation list.

        The statutory citations — act, cabinet order, ministerial ordinance or
        guidance name, with its article reference and source string — ARE the
        product a programmatic caller needs, not a footnote, so they are
        surfaced as a real decoded list on top of the base output / status /
        trace_id / correlation_id / node_history envelope.

        Fail closed, three times over:
          1. Only a SUCCESS run attaches the structured field at all. Any other
             outcome (trust denial, refused payload, rejected option, blocked
             output, subgraph error) returns the base envelope alone.
          2. The citation list is re-scanned by the same credential gate the
             output boundary uses. That gate covered the rendered answer;
             this second pass covers the field get_output() adds on top of it.
          3. The citation list must still be well formed — every entry naming a
             titled, attributed source. A violation flips the status to ERROR
             and withholds the field entirely, never a partial payload.

        A violation found here withholds the rendered answer too, not just the
        field that failed. The answer and its citation list are one product: an
        envelope that reports an error while still carrying the answer text has
        released it, and the error label is decoration. So both violation paths
        replace the base envelope's output with the withheld placeholder.
        """
        base: Dict[str, Any] = super().get_output(state)

        if state.get("status") != AgentStatus.SUCCESS.value:
            # Last line of defence. The boundaries above write a placeholder
            # over the output fields when they refuse, so a non-success envelope
            # should already carry one. Answer text here instead means it was
            # written by something that was later overruled — an earlier pass
            # through the backbone, or a node whose own error path never reached
            # the output fields — and shipping it under an error status releases
            # it just as surely as shipping it under a success one. An absent
            # output stays absent: nothing was produced, so there is nothing to
            # withhold and nothing to describe.
            if base.get("output") and not is_withheld(base["output"]):
                base["output"] = WITHHELD_OUTPUT
            return base

        # A run that completed without processing the request has no citation
        # list to surface. SUCCESS reports that the run reached a defined end
        # safely, not that an answer was produced - so the structured field is
        # withheld here exactly as it is on a non-SUCCESS status. Only the
        # sentence saying what to correct is returned.
        if state.get("error_code"):
            return base

        citations = from_json(state.get("citations"), []) or []

        if _security_gate_output({"citations": citations}):
            base["status"] = AgentStatus.ERROR.value
            base["output"] = WITHHELD_OUTPUT
            return base

        if check_citations_wellformed(citations):
            base["status"] = AgentStatus.ERROR.value
            base["output"] = WITHHELD_OUTPUT
            return base

        base["citations"] = citations
        return base


# Back-compat alias — config/agent.yaml declares the dotted path to the class
# above, and src/api/server.py imports it directly. Keep both names pointing at
# the agent.
Graph = GovernmentRegulationKnowledgeAgent
