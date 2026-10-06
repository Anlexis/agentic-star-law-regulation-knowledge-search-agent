"""AgentCore Platform v1.0"""

# GOV-C2-005 — GenerateAnswerNode
# Domain node 4: assemble the grounded statutory answer from the ranked
# knowledge-base passages.
#
# The assembly is deterministic — no model call. The answer is a lead sentence
# plus one cited point per passage, each carrying a numbered citation marker
# [n]. Nothing outside the ranked_documents input reaches the answer body, so
# every provision stated is traceable to a passage that retrieval returned. The
# output boundary re-checks that property rather than trusting it, because
# "grounded by construction" stops being true the moment the construction
# changes (see src/output_invariants.py).
#
# The caller's question is echoed into the lead sentence, which makes caller
# text part of the rendered report. It is passed through sanitize_echo() first:
# a question containing bracketed digits or markdown structure would otherwise
# be able to forge a citation marker or a section heading that no retrieved
# passage backs.
#
# Swapping this assembly for model-based synthesis changes only the inside of
# execute(); the state contract and the grounding rule are unchanged. The
# synthesis contract is written out in config/prompts/answer_synthesis_prompt.md
# and in docs/02_design.md.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.output_invariants import sanitize_echo
from src.schemas.state import from_json, to_json
from framework.schemas.agent_status import AgentStatus

# Answer body used when no KB passage cleared the relevance threshold.
_NO_COVERAGE_ANSWER = (
    "The government regulation knowledge base does not contain sufficient "
    "coverage to answer this question. Rephrase the query with more specific "
    "statutory or regulatory terms, or escalate to the legal/compliance team "
    "for a manual review."
)

# Cited excerpt length per passage inside the answer body.
_POINT_EXCERPT_CHARS = 240


def _first_sentences(text: str, limit: int) -> str:
    """Trim an excerpt at a sentence boundary where possible, else hard-cap."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    period = cut.rfind(". ")
    if period > limit // 2:
        return cut[: period + 1]
    return cut.rstrip() + "..."


class GenerateAnswerNode(FunctionNode):
    """Rule-based grounded answer assembly with numbered citations.

    Input state keys:
        ranked_documents: JSON list of surviving passages (from RerankFilterNode)
        search_query:     normalised query (for the lead sentence)

    Output state keys (partial dict):
        grounded_answer: answer body with [n] citation markers
        citations:       JSON list [{ref, id, title, source}]
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

        emit_progress("Composing the answer...")
        ranked: List[Dict[str, Any]] = from_json(state.get("ranked_documents"), []) or []
        query = sanitize_echo(state.get("search_query") or "")

        citations: List[Dict[str, Any]] = []

        if not ranked:
            grounded_answer = _NO_COVERAGE_ANSWER
        else:
            lines: List[str] = []
            if query:
                lines.append(
                    f"Based on the seeded government regulation knowledge base, "
                    f'the following provisions answer the question: "{query}"'
                )
            else:
                lines.append(
                    "Based on the seeded government regulation knowledge base, " "the most relevant provisions are:"
                )
            lines.append("")
            for ref, doc in enumerate(ranked, start=1):
                if not isinstance(doc, dict):
                    continue
                title = str(doc.get("title", "")).strip()
                excerpt = _first_sentences(str(doc.get("excerpt", "")), _POINT_EXCERPT_CHARS)
                lines.append(f"[{ref}] {title}: {excerpt}")
                citations.append(
                    {
                        "ref": ref,
                        "id": str(doc.get("id", "")),
                        "title": title,
                        "source": str(doc.get("source", "")),
                    }
                )
            grounded_answer = "\n".join(lines)

        # Audit: grounded answer assembled.
        emit_trace_event(
            "generate_answer_complete",
            {
                "citation_count": len(citations),
                "answer_chars": len(grounded_answer),
                "no_coverage": not ranked,
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
        }
