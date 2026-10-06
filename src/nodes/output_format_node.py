"""AgentCore Platform v1.0"""

# GOV-C2-005 — OutputFormatNode
# Domain node 5 (terminal): compose the final answer — the grounded statutory
# body, the Sources list, and the standing legal-advice disclaimer.
#
# This node is also the FIRST of the two boundaries that enforce the template's
# output invariant. It is the only place that can see the retrieved passage set
# alongside the citations, so it is the only place that can catch a citation
# naming a source retrieval never returned. The outer boundary re-checks what it
# can see (the rendered answer and the citation list) as the answer leaves.
#
# On a violation the node fails CLOSED: it emits an error status and writes a
# non-empty withheld placeholder over the answer field, rather than a
# partly-trustworthy document. An answer whose provenance cannot be shown is
# worse than no answer in this domain.
#
# The placeholder, rather than simply omitting the field, is what makes the
# refusal stick. Omitting it leaves the answer field falsy, and every consumer
# downstream — the outer merge, and the envelope's `formatted_output or result`
# fallback — reads a falsy answer as "look somewhere else" rather than as "this
# was refused". See the withholding note in src/output_invariants.py.
#
# The disclaimer is part of THIS node's output contract, not of the outer
# post_process slot — post_process gates, it does not compose.
#
# Wired by the inner graph (DomainWorkflowGraph). The inner graph's get_output()
# surfaces formatted_answer + citations + status to the outer merge_output().
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import OUTPUT_BLOCKED
from src.output_invariants import (
    INNER_OUTPUT_FIELDS,
    LEGAL_DISCLAIMER,
    WITHHELD_OUTPUT,
    check_grounding,
    check_sources_were_retrieved,
    withheld_delta,
)
from src.schemas.state import from_json

# Names the rule that was broken, never the passage or citation text that broke
# it. Quoting the content would put it back into the delta this node returns,
# which is the one place a refusal has to be quietest: the framework scans every
# value of that delta, and a scan that raises there discards the whole delta.
_UNGROUNDED_MESSAGE = (
    "OutputFormatNode: answer withheld — a cited provision could not be traced "
    "to a retrieved knowledge-base passage."
)


class OutputFormatNode(FunctionNode):
    """Compose the final answer: body, sources, and legal-advice disclaimer.

    Input state keys:
        grounded_answer:  answer body with [n] citation markers
        citations:        JSON list [{ref, id, title, source}]
        ranked_documents: JSON list of the passages retrieval actually returned

    Output state keys (partial dict):
        formatted_answer: final rendered answer string, or the non-empty
                          withheld placeholder when the answer is refused
        citations:        cleared to None when the answer is refused — the
                          citation list is the other half of the answer, and
                          half an answer released under an error status is
                          still a release
        status:           terminal status value (a plain string — never the
                          bare enum, which does not survive checkpointing)
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

        emit_progress("Formatting the response...")
        grounded_answer = state.get("grounded_answer") or ("No answer is available for this request.")
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []
        ranked: List[Dict[str, Any]] = from_json(state.get("ranked_documents"), []) or []

        violation = check_grounding(grounded_answer, citations) or (check_sources_were_retrieved(citations, ranked))
        if violation:
            emit_trace_event("output_format_grounding_violation", {"violation": violation}, state)
            emit_progress(OUTPUT_BLOCKED)
            return {
                **withheld_delta(WITHHELD_OUTPUT, INNER_OUTPUT_FIELDS),
                "status": AgentStatus.ERROR.value,
                "error_log": [f"{_UNGROUNDED_MESSAGE} ({violation})"],
            }

        lines: List[str] = []
        lines.append("# Government Regulation Search Result")
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                title = str(citation.get("title", "")).strip()
                source = str(citation.get("source", "")).strip()
                suffix = f" ({source})" if source else ""
                lines.append(f"- [{ref}] {title}{suffix}")
        else:
            lines.append("- none (no knowledge-base provision cleared the relevance threshold)")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*{LEGAL_DISCLAIMER}*")

        formatted_answer = "\n".join(lines)

        # Audit: final answer composed, disclaimer attached, citations grounded.
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
