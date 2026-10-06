"""AgentCore Platform v1.0"""

# GOV-C2-005 — PreProcessNode (outer pre_process slot).
#
# This node owns the caller-data contract. Everything a caller can influence is
# checked here, once, before any of it reaches the retrieval pipeline:
#
#   trust      The manifest declares required_trust_level VERIFIED_EXTERNAL, and
#              this node carries the same level, so an unvouched caller is
#              stopped at the outer boundary rather than inside the workflow.
#   refusal    Instruction-override, prompt-disclosure, role-reassignment and
#              chat-role-marker payloads are refused HERE, in the template, in
#              both the question and the context mapping. Relying on a platform
#              gate alone would mean the refusal disappears wherever that gate is
#              absent or configured differently — the request would then reach
#              the answer path and return a normal, successful answer.
#   redaction  Direct identifiers a civil servant might paste in while describing
#              a case are stripped from the question before it is written to
#              state, so raw identifiers never reach the domain nodes or the
#              checkpoint store.
#   options    Caller options (category / top_k / score_threshold / channel) are
#              parsed against explicit bounds and become an inert, validated
#              mapping. A present-but-invalid option rejects the request naming
#              the field; the rejected value is never echoed back.
#
# Node contract: extend FunctionNode; implement execute(self, state) -> dict —
# no `config` parameter. Return ONLY the fields this node changes (never full
# state). Read input_context via state.get("input_context", {}) — read-only.

import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED
from src.caller_contract import (
    CallerContractError,
    detect_injection,
    parse_caller_options,
    scan_context_for_injection,
)
from src.schemas.state import to_json

# Surface-level identifier patterns redacted before validated_input is written.
# Downstream domain nodes only ever operate on the normalised question text and
# knowledge-base passage summaries, never raw case or reference numbers.
_PII_PATTERNS: List["re.Pattern[str]"] = [
    # Bank-account-shaped code: 2 letters + 2 digits + up to 30 alphanumerics.
    re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    # Long reference / case / account numbers: 10-19 consecutive digits
    # (optionally grouped).
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{2,11}\b"),
    # E-mail addresses.
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
]
_PII_REPLACEMENT = "[REDACTED]"

# Refusal text. Deliberately uniform: it names neither the matched pattern nor
# any part of the payload, so a caller cannot use the response to map the rules.
_REFUSAL_MESSAGE = "PreProcessNode: request refused — the payload is not a regulation question."


def _surface_strip_identifiers(text: str) -> str:
    """Redact obvious direct-identifier tokens from a free-text string."""
    for pattern in _PII_PATTERNS:
        text = pattern.sub(_PII_REPLACEMENT, text)
    return text


class PreProcessNode(FunctionNode):
    """Validate, refuse, redact and bound the request before the workflow runs."""

    # Explicit by design, not inherited implicitly. Outer boundary gate slot —
    # matches the manifest's declared required_trust_level (config/agent.yaml).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        emit_progress("Checking the request...")
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only

        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            # Nothing to answer, but the caller can send a question and try
            # again - so the run completes carrying the reason rather than
            # terminating and surfacing only an exception type.
            emit_progress(EMPTY_INPUT)
            emit_trace_event("pre_process_declined", {"reason": "EMPTY_INPUT"}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        # Refuse before anything else looks at the payload, and refuse the
        # context channel on the same terms as the question.
        refusal = detect_injection(user_input) or scan_context_for_injection(input_context)
        if refusal:
            emit_trace_event("pre_process_refused", {"reason": refusal}, state)
            emit_progress(INPUT_REJECTED)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [_REFUSAL_MESSAGE],
            }

        try:
            caller_options = parse_caller_options(input_context)
        except CallerContractError as exc:
            # A value the caller can correct. The field name travels; the value
            # does not. The run completes carrying the reason so the request can
            # be sent again with a usable value.
            emit_progress(INPUT_REJECTED)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "INVALID_REQUEST",
                "error_log": [f"PreProcessNode: {exc}"],
            }

        validated_input = _surface_strip_identifiers(user_input.strip())

        emit_trace_event(
            "pre_process_complete",
            {
                "input_chars": len(validated_input),
                "caller_options": sorted(caller_options),
            },
            state,
        )

        enriched_context: Dict[str, Any] = {
            "source": "GovernmentRegulationKnowledgeAgent",
            "channel": caller_options.get("channel", "unknown"),
        }

        return {
            "validated_input": validated_input,
            "caller_options": to_json(caller_options),
            "enriched_context": enriched_context,
            "status": AgentStatus.SUCCESS.value,
        }
