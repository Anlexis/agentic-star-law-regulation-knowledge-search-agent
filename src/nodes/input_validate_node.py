"""AgentCore Platform v1.0"""

# GOV-C2-005 — InputValidateNode
# Domain node 1: settle the search request the rest of the pipeline will run.
#
# Two channels can carry a caller's options, and this node reconciles them:
#
#   context channel   The invocation's context mapping, validated by
#                     PreProcessNode and carried inward by the context bridge.
#                     This is the first-class channel and it wins.
#   request envelope  A JSON object sent as the question itself, kept for
#                     callers that have only the one string field:
#
#                         plain text                -> the whole string is the query
#                         {"query": "...",
#                          "category": "...",
#                          "top_k": N,
#                          "score_threshold": F}    -> query plus options
#
# Envelope options are parsed with the SAME bounds as the context channel, from
# the same module, so neither channel is the lenient one. An option that is
# present but out of bounds REJECTS the request naming the field — the caller is
# told its request was not run, rather than silently receiving an answer computed
# with a value the template substituted.
#
# category, when supplied, is one of: "act" | "cabinet_order" |
# "ministerial_ordinance" | "administrative_guidance", matching the seeded
# corpus (config/kb/gov_regulations_kb.json).
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import INPUT_REJECTED
from src.caller_contract import (
    CallerContractError,
    parse_category,
    parse_score_threshold,
    parse_top_k,
)
from src.schemas.state import from_json, to_json

# Hard cap on the normalised query length (defence in depth on input size).
_MAX_QUERY_CHARS = 2000

_WHITESPACE_RE = re.compile(r"\s+")

# Envelope keys this node accepts, and the parser each is checked with.
_ENVELOPE_PARSERS = {
    "category": parse_category,
    "top_k": parse_top_k,
    "score_threshold": parse_score_threshold,
}


class InputValidateNode(FunctionNode):
    """Normalise the query and settle the caller's retrieval options.

    Input state keys:
        validated_input | user_input: redacted request payload
        caller_options:               validated options from the context channel

    Output state keys (partial dict):
        search_query:  normalised free-text search query
        query_filters: JSON dict {"category", "top_k", "score_threshold"}
        intake_notes:  (when anomalies were seen) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        emit_progress("Checking the request...")
        raw = state.get("validated_input") or state.get("user_input", "")
        notes: List[str] = []

        # Options already validated by PreProcessNode, carried in by the bridge.
        options: Dict[str, Any] = {
            key: value
            for key, value in (from_json(state.get("caller_options"), {}) or {}).items()
            if key in _ENVELOPE_PARSERS
        }

        query = ""
        if isinstance(raw, str) and raw.strip():
            text = raw.strip()
            payload: Any = None
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    notes.append(
                        "InputValidateNode: JSON-looking input did not parse — " "treated as a plain text query."
                    )
            if isinstance(payload, dict):
                query = str(payload.get("query") or payload.get("question") or "")
                try:
                    for key, parser in _ENVELOPE_PARSERS.items():
                        value = payload.get(key)
                        if value is None or key in options:
                            # Absent, or already set by the context channel,
                            # which is authoritative when both are supplied.
                            continue
                        options[key] = parser(value)
                except CallerContractError as exc:
                    # Fail closed on the retrieval path. The field name travels;
                    # the value does not. The run COMPLETES carrying the reason,
                    # so the caller can correct the value and send the request
                    # again on the same conversation.
                    emit_progress(INPUT_REJECTED)
                    return {
                        "status": AgentStatus.SUCCESS.value,
                        "error_code": "INVALID_REQUEST",
                        "error_log": [f"InputValidateNode: {exc}"],
                    }
            else:
                query = text
        else:
            notes.append("InputValidateNode: empty request — no query to search.")

        # Normalise whitespace and cap length.
        query = _WHITESPACE_RE.sub(" ", query).strip()
        if len(query) > _MAX_QUERY_CHARS:
            query = query[:_MAX_QUERY_CHARS]
            notes.append(f"InputValidateNode: query truncated to {_MAX_QUERY_CHARS} chars.")

        filters = {key: options.get(key) for key in _ENVELOPE_PARSERS}

        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "options_applied": sorted(k for k, v in filters.items() if v is not None),
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": query,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
