"""AgentCore Platform v1.0"""

# GOV-C2-005 — RerankFilterNode
# Domain node 3: rerank the retrieval candidates and enforce the relevance
# floor. Deterministic: a small category-match boost on top of the retrieval
# score, drop everything below `score_threshold`, cap the survivors at `top_k`.
#
# Config: reads `top_k` and `score_threshold` from the state-seeded
# `retrieval_config` field, normalised to finite, in-bounds values with the
# module defaults below as the fallback.
#
# A caller override (query_filters) applies only when it is STRICTER than the
# configured tuning — a smaller passage cap, or a higher relevance floor. A
# caller can narrow what it receives; it cannot talk the template into citing
# passages the deployment's own relevance floor rejected.
#
# execute() takes no `config` parameter.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import math
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.runtime_config import DEFAULT_RETRIEVAL, resolve_retrieval_tuning
from src.schemas.state import from_json, to_json
from framework.schemas.agent_status import AgentStatus

# Defaults mirror the `retrieval` block in config/config.yaml — one definition,
# shared with RetrieveNode.
_DEFAULT_RETRIEVAL: Dict[str, Any] = DEFAULT_RETRIEVAL

# Boost applied when a candidate's category matches the caller's filter.
_CATEGORY_BOOST = 0.1


def _resolve_retrieval_config(state: AgentState) -> Dict[str, Any]:
    """Effective retrieval tuning for this call, finite and in bounds."""
    return resolve_retrieval_tuning(from_json(state.get("retrieval_config"), None))


def _finite_override(value: Any) -> Optional[float]:
    """Return a caller override as a finite number, or None if it is unusable.

    Booleans are excluded explicitly: `True` is an `int` in Python and would
    otherwise be read as the override 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class RerankFilterNode(FunctionNode):
    """Rerank candidates, apply the score threshold, cap at top_k.

    Input state keys:
        retrieved_documents: JSON list of scored candidates (from RetrieveNode)
        query_filters:       JSON dict with the caller's validated options
        retrieval_config:    forwarded retrieval tuning (JSON)

    Output state keys (partial dict):
        ranked_documents: JSON list of surviving passages (score desc, <= top_k)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # The request was already found unacceptable upstream: this run
        # completes without an answer, so there is nothing for this step to
        # do. Returning the marker keeps it on the node's own result dict,
        # which is what the output gate inspects.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}

        emit_progress("Ranking the results...")
        candidates: List[Dict[str, Any]] = from_json(state.get("retrieved_documents"), []) or []
        filters = from_json(state.get("query_filters"), {}) or {}
        retrieval_cfg = _resolve_retrieval_config(state)

        top_k = int(retrieval_cfg["top_k"])
        score_threshold = float(retrieval_cfg["score_threshold"])

        # Caller overrides apply only in the stricter direction: a smaller cap
        # on how many passages are cited, and a higher relevance floor. Both
        # arrived already validated — bounded, finite, and rejected outright if
        # they were not — but they are re-checked here rather than trusted.
        # This node can also be driven directly, and a non-finite value that
        # slipped in would not raise: every comparison against NaN is False, so
        # min() and max() would silently return the other operand and the
        # override would look applied when it had not been.
        caller_top_k = _finite_override(filters.get("top_k"))
        if caller_top_k is not None:
            top_k = min(top_k, max(1, int(caller_top_k)))

        caller_threshold = _finite_override(filters.get("score_threshold"))
        if caller_threshold is not None:
            score_threshold = max(score_threshold, caller_threshold)

        category = filters.get("category")

        reranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            entry = dict(candidate)  # local copy - inputs stay immutable
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if category and str(entry.get("category", "")).lower() == str(category).lower():
                score = min(1.0, score + _CATEGORY_BOOST)
            entry["score"] = round(score, 4)
            reranked.append(entry)

        # Deterministic ordering: score desc, then id asc for stable ties.
        reranked.sort(key=lambda c: (-c.get("score", 0.0), str(c.get("id", ""))))

        kept = [c for c in reranked if c.get("score", 0.0) >= score_threshold][:top_k]
        dropped = len(reranked) - len(kept)

        # Audit: rerank and relevance floor applied.
        emit_trace_event(
            "rerank_filter_complete",
            {
                "kept": len(kept),
                "dropped": dropped,
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
            state,
        )

        return {"ranked_documents": to_json(kept)}
