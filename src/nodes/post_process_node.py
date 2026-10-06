"""AgentCore Platform v1.0"""

# GOV-C2-005 — PostProcessNode (outer post_process slot; the output boundary).
#
# Reads the final answer from state["result"] — written by
# RegulationKnowledgeGraphNode.merge_output() from the inner graph's rendered
# answer — and surfaces it only after it clears the boundary. This node GATES;
# it does not compose. The answer body, the Sources list and the disclaimer are
# composed upstream by the inner graph's OutputFormatNode.
#
# TWO INDEPENDENT LAYERS, each with its own audit event:
#
#   1. Credential redaction. The answer is scanned RECURSIVELY for
#      credential-shaped strings: a top-level-string-only scan misses a
#      violation nested inside a dict or list, so the scan accepts `content: Any`
#      and walks dict values and list/tuple/set elements, checking every string
#      leaf. A match replaces the output with a sanitised stub and returns an
#      error status.
#
#   2. The template's own output invariant — every cited provision traceable to
#      a retrieved passage, and the standing legal-advice disclaimer present.
#      See src/output_invariants.py for why this, and not a numeric rounding
#      grid, is what this template's output boundary enforces.
#
# The layers stay independent on purpose. Layer 2 never REWRITES the answer, it
# only accepts or refuses it whole, so it cannot damage a pattern layer 1 is
# looking for. A boundary layer that edits text has to run after every scanner
# that reads it, and that ordering is easy to lose.
#
# Both layers refuse the same way: they return an error status AND write a
# non-empty placeholder over every output-bearing field (withheld_delta() in
# src/output_invariants.py owns that field set). Returning an error alone is not
# containment — the envelope resolves the answer as `formatted_output or
# result`, so a layer that blanks only the first field hands the caller the
# answer it just refused. See the withholding note in src/output_invariants.py.
#
# No _extra_security_gate_input/_output instance methods are defined on this
# node — the framework auto-wraps such hooks, and this gate is called from
# execute() directly so the same function can be reused at the outer envelope.
#
# The manifest declares required_trust_level VERIFIED_EXTERNAL
# (config/agent.yaml) and this node carries the same level.

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, OUTPUT_BLOCKED, TOO_LONG
from src.output_invariants import (
    BLOCKED_OUTPUT,
    OUTER_OUTPUT_FIELDS,
    WITHHELD_OUTPUT,
    check_rendered_answer,
    withheld_delta,
)
from src.schemas.state import from_json

logger = logging.getLogger(__name__)

# Disallowed output content. Each tuple: (name, compiled regex) — order matters
# (most specific first).
_DISALLOWED_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    # API key shapes: sk-..., pk-..., ak-...
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    # JSON web token: three base64url segments separated by dots.
    (
        "jwt",
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ),
    # Bearer token in an authorization-like context.
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/]{20,}", re.IGNORECASE)),
    # Credential assignment patterns.
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
]

# Every diagnostic this node emits names the rule that was broken and nothing
# else. The names come from this module's own pattern table and from
# src/output_invariants.py — never from the text that was scanned. A message
# carrying the matched substring would be scanned by the framework's output gate
# along with the rest of this delta, and a gate that raises there discards the
# whole delta, the clearing included.
_INVARIANT_MESSAGE = (
    "PostProcessNode: output withheld — the answer did not satisfy the " "grounding and disclaimer contract"
)

# Recursion-depth guard. Output is always a shallow, framework-produced
# structure; this only stops a pathological or cyclic structure from recursing
# unbounded and is not expected to be reached in normal operation.
_MAX_SCAN_DEPTH = 12


def _security_gate_output(content: Any, _depth: int = 0) -> Optional[str]:
    """Scan output content for credential-shaped strings.

    Walks nested dict values and list/tuple/set elements so a violation buried
    inside a structured payload is caught, not just one in a top-level string.
    Returns the name of the first match, or None when clean.
    """
    if _depth > _MAX_SCAN_DEPTH or content is None:
        return None
    if isinstance(content, str):
        for name, pattern in _DISALLOWED_PATTERNS:
            if pattern.search(content):
                return name
        return None
    if isinstance(content, dict):
        for value in content.values():
            violation = _security_gate_output(value, _depth + 1)
            if violation:
                return violation
        return None
    if isinstance(content, (list, tuple, set)):
        for item in content:
            violation = _security_gate_output(item, _depth + 1)
            if violation:
                return violation
        return None
    # Non-string scalar (int / float / bool / ...) — nothing to scan.
    return None


# Caller-facing wording for a run that completed without an answer. The marker
# is an internal reason code; this maps it to the sentence the caller sees.
# Static sentences only - no request value is ever substituted, so nothing the
# caller sent can be reflected back through this path.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Finalize the output, behind the two output-boundary layers."""

    # Explicit by design, not inherited implicitly. Outer boundary gate slot —
    # matches the manifest's declared required_trust_level (config/agent.yaml).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        emit_progress("Finalising the response...")

        # The run completed without an answer because the request could not be
        # accepted as written. Report the reason as the response: the caller
        # needs to know what to change, and an empty body would leave them with
        # nothing. Status stays SUCCESS - the run did what it could with the
        # request it was given, and the caller can correct it and send again on
        # the same conversation.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "formatted_output": message,
                "result": message,
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
            }
        result = state.get("result", "")

        if not result or (isinstance(result, str) and not result.strip()):
            # Nothing to gate — forward as-is (non-fatal). A run that was
            # already refused upstream never arrives here: the framework skips
            # execute() when the incoming state carries an error status, and the
            # backbone routes a non-success run past this slot to finalize. The
            # envelope's own last-line check in src/graph/graph.py is what
            # covers those, since this node is not on their path.
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        # Layer 1 — credential redaction.
        violation = _security_gate_output(result)
        if violation:
            logger.error("PostProcessNode: output blocked — violation type: %s", violation)
            emit_progress(OUTPUT_BLOCKED)
            return {
                **withheld_delta(BLOCKED_OUTPUT, OUTER_OUTPUT_FIELDS),
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: output blocked — disallowed content " f"detected ({violation})"],
            }

        # Layer 2 — the template's own output invariant. Only meaningful over a
        # rendered answer; a non-string result has already cleared layer 1 and
        # carries no answer to check.
        if isinstance(result, str):
            citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []
            breach = check_rendered_answer(result, citations)
            if breach:
                emit_trace_event("post_process_invariant_violation", {"violation": breach}, state)
                logger.error("PostProcessNode: output withheld — invariant breach: %s", breach)
                emit_progress(OUTPUT_BLOCKED)
                return {
                    **withheld_delta(WITHHELD_OUTPUT, OUTER_OUTPUT_FIELDS),
                    "status": AgentStatus.ERROR.value,
                    "error_log": [f"{_INVARIANT_MESSAGE} ({breach})"],
                }

        # Clean. Audit: a finalized output was emitted.
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(str(result))},
            state,
        )

        return {
            "formatted_output": result,
            "status": AgentStatus.SUCCESS.value,
        }
