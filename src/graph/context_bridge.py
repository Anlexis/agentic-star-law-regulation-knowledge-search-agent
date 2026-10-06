"""AgentCore Platform v1.0"""

# GOV-C2-005 — caller-context bridge between the outer and the inner graph.
#
# The framework's GraphNode calls
#     subgraph.invoke(user_input, session_id=..., ctx=...)
# which carries the question text but NOT the caller's `input_context` mapping.
# The inner graph therefore starts with an empty input_context, and any caller
# option the inner nodes need would silently fall back to its default: a caller
# asking for cabinet-order provisions only, or for a single most-relevant
# passage, would receive the unfiltered four-passage answer instead and have no
# way to tell the difference.
#
# This module closes that gap with a ContextVar:
#
#   outer RegulationKnowledgeGraphNode.extract_input() -> publish_caller_options()
#   inner DomainWorkflowGraph._extra_initial_state()   -> read_caller_options()
#
# Both run on the same call stack inside one GraphNode invocation, so the value
# published by the outer node is the value the inner graph reads, and concurrent
# invocations (threads, async tasks) each see their own value.
#
# Only ALREADY-VALIDATED values travel here: PreProcessNode is the single owner
# of the caller contract and the bridge transports its output — it never carries
# raw caller data.

from contextvars import ContextVar
from typing import Any, Dict

_CALLER_OPTIONS: ContextVar[Dict[str, Any]] = ContextVar("gov_c2_005_caller_options", default={})


def publish_caller_options(values: Dict[str, Any]) -> None:
    """Publish the validated caller options for the inner graph to pick up.

    A defensive copy is stored so a later mutation of the caller's dict cannot
    change what the inner graph already read.
    """
    _CALLER_OPTIONS.set(dict(values))


def read_caller_options() -> Dict[str, Any]:
    """Return the validated caller options published for this invocation.

    Returns an empty dict when the inner graph is invoked directly (unit tests,
    or any caller that bypasses the outer graph) — every inner node then applies
    its own documented default.
    """
    return dict(_CALLER_OPTIONS.get())


def clear_caller_options() -> None:
    """Reset the bridge to empty. Used by tests to prove isolation."""
    _CALLER_OPTIONS.set({})
