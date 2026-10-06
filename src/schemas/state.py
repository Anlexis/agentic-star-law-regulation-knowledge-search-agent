"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic model. Graph checkpoints are
# serialised with msgpack, and model objects corrupt silently there. Extend
# AgentState with agent-specific fields only. Do NOT add credentials or secrets.
#
# For the same reason, structured fields (dict / list[dict]) are stored as JSON
# STRINGS rather than bare Python containers. Producers serialise with to_json()
# on write; consumers deserialise with from_json() on read.
#
# GOV-C2-005 — Government Regulation Knowledge Agent. Two-layer nested graph:
# outer backbone (AgentBaseGraph) plus inner domain workflow (BaseGraph). The
# fields below cover both layers.
#
# Confidentiality note: direct identifiers a caller might paste in while
# describing a case (bank-account-shaped codes, long reference numbers, e-mail
# addresses) are stripped by PreProcessNode before any field here is written.
# Only the normalised legal question, knowledge-base passage summaries and the
# final grounded answer are persisted — never raw caller identifiers.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for GOV-C2-005.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are NotRequired so the TypedDict is valid at graph
    initialisation, before any node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / RegulationKnowledgeGraphNode.merge_output
    # ------------------------------------------------------------------

    # Redacted, validated query payload produced by PreProcessNode. Raw input is
    # NOT persisted beyond that node.
    validated_input: NotRequired[str]

    # JSON STRING (to_json) of the caller options PreProcessNode validated:
    # {"category": str, "top_k": int, "score_threshold": float, "channel": str},
    # each key present only when the caller supplied it. Carried into the inner
    # graph by src/graph/context_bridge.py, since the framework passes only the
    # question string across that boundary.
    caller_options: NotRequired[Optional[str]]

    # Final government-regulation answer, mapped from the inner graph's
    # formatted_answer output via merge_output.
    regulation_answer: NotRequired[str]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text legal/regulatory question (whitespace-collapsed,
    # length-capped).
    search_query: NotRequired[str]

    # JSON STRING (to_json) of the settled query options. Deserialised shape:
    # {"category": str | None, "top_k": int | None, "score_threshold": float |
    # None}. category is one of "act" | "cabinet_order" |
    # "ministerial_ordinance" | "administrative_guidance" when supplied.
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: NotRequired[Optional[str]]

    # Retrieval tuning read from config/config.yaml by
    # RegulationKnowledgeGraphNode._parent_config() and republished into inner
    # state by DomainWorkflowGraph._extra_initial_state(). JSON STRING (to_json)
    # of {"top_k": int, "score_threshold": float, "kb_path": str}. Consumers
    # (RetrieveNode, RerankFilterNode) read it back via from_json().
    retrieval_config: NotRequired[Optional[str]]

    # RetrieveNode output
    # JSON STRING (to_json) of scored KB candidates. Deserialised shape:
    # list[dict], each entry {"id": str, "title": str, "category": str,
    # "source": str, "score": float, "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: NotRequired[Optional[str]]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: NotRequired[Optional[str]]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers.
    grounded_answer: NotRequired[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict], each
    # entry {"ref": int, "id": str, "title": str, "source": str} — title carries
    # the act / order / ordinance name and its article reference, source carries
    # the statute citation string. Consumers (OutputFormatNode) read it back via
    # from_json(); also re-surfaced at the outer layer by get_output().
    citations: NotRequired[Optional[str]]

    # OutputFormatNode output
    # Final formatted answer (body + sources + legal-advice disclaimer). Written
    # by OutputFormatNode; surfaced to the outer graph via get_output() ->
    # merge_output().
    formatted_answer: NotRequired[str]

    # Validation / parse notes accumulated during intake (no PII).
    # JSON STRING (to_json) of list[str].
    intake_notes: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Degraded completion marker
    # ------------------------------------------------------------------

    # Set when the run completes WITHOUT producing an answer because the
    # caller's request could not be accepted as written - a rejection the
    # caller can correct and retry. The run still completes: no retrieval is
    # performed, no answer is assembled, and the domain audit event for the
    # rejection is still emitted. Carrying this as a completion marker rather
    # than a terminal error is what lets the caller see the reason and send a
    # corrected request on the same conversation.
    #
    # Content the agent refuses outright, and a breach of a contract the
    # caller cannot influence, are NOT reported here - those stay terminal so
    # they are not mistaken for something a reworded request would get past.
    #
    # Once set, every later domain node passes through without doing work, and
    # the value is carried across the inner/outer boundary by get_output() and
    # merge_output().
    error_code: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
