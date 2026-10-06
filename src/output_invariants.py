"""AgentCore Platform v1.0"""

# GOV-C2-005 — the invariants every answer this template emits must satisfy.
#
# The promise this template makes to its callers is a GROUNDING promise, not a
# numeric one: every provision it states is traceable to a passage that was
# actually retrieved from the seeded knowledge base, and every answer carries the
# standing legal-advice disclaimer. There are no monetary aggregates anywhere in
# the rendered output — the answer is statutory text, act titles and article
# references — so the rounding grid some domains apply at their output boundary
# does not exist here and nothing rewrites numbers in the answer. Article
# numbers, act numbers ("Act No. 66 of 2009"), knowledge-base identifiers
# ("gov-001") and years therefore reach the caller byte-for-byte.
#
# The checks below are what replaces that grid. They are enforced twice, at two
# independent boundaries with their own audit events:
#
#   inner  OutputFormatNode, which can additionally see the retrieved passage set
#          and so can catch a citation naming a source that was never retrieved.
#   outer  PostProcessNode and the agent's get_output(), which re-check the
#          rendered answer and the structured citation list as they leave.
#
# None of these checks REWRITES the answer; they accept it or refuse it whole.
# That keeps them order-independent with respect to the credential redaction
# layer, which does rewrite — a check that mutated text could otherwise destroy
# the very pattern the credential scan is looking for.

import re
from typing import Any, Dict, List, Optional, Sequence, Set

# ---------------------------------------------------------------------------
# Withholding — what a refusal has to leave behind
# ---------------------------------------------------------------------------
#
# Refusing an answer and withholding it are not the same act, and only the
# second one is worth anything.
#
# The caller-visible answer is resolved as `formatted_output or result` — an OR
# across two state fields. A boundary that refuses by writing an EMPTY string
# into the first field has not withheld anything: the empty string is falsy, so
# the OR falls through to the second field, and the exact answer the boundary
# just refused is what the caller receives — inside an envelope labelled error,
# which reads as if the refusal worked.
#
# So a refusal writes a NON-EMPTY placeholder over EVERY field that can carry
# released text. Non-empty is the load-bearing property: it is what stops the
# fallback resolving past the refusal. Clearing "the output field" is not
# enough; the whole set has to go, which is why the sets live here as named
# constants rather than as a literal list at each boundary, where they could
# drift apart one field at a time.
#
# A refusal message names the boundary that refused and the rule that was
# broken. It never quotes the content it matched. That restraint is structural,
# not stylistic: the framework's output gate scans every value of the delta a
# node returns, so a message quoting a credential-shaped match makes that gate
# raise on the REFUSAL itself — and a node that raises has its delta discarded,
# leaving the un-cleared answer in state to be shipped by the fallback. A
# refusal that quotes what it caught destroys itself.

# Written over the output fields when a boundary withholds an answer. Any
# non-empty string works; this one tells the caller what happened and carries
# no bracketed digits, so re-checking it can never read it as a citation marker.
WITHHELD_OUTPUT = (
    "[OUTPUT WITHHELD - the generated answer did not satisfy this agent's "
    "output contract and was not released. No partial answer is returned. "
    "The audit trail records which boundary refused it and why.]"
)

# The same, for the credential layer, which can say something more useful about
# what to do next. Two messages, one contract: both are non-empty, both name
# only the boundary and the rule, neither reproduces what was matched.
BLOCKED_OUTPUT = (
    "[OUTPUT BLOCKED — disallowed content detected. Review the generated output "
    "and retry without credential-like strings.]"
)

_PLACEHOLDER_PREFIXES = ("[OUTPUT WITHHELD", "[OUTPUT BLOCKED")


def is_withheld(value: Any) -> bool:
    """True when `value` is a placeholder a boundary wrote, not answer text.

    Used at the envelope to tell "a boundary already withheld this" apart from
    "answer text is still sitting here", so the last-line check can neutralise
    the second without overwriting the more specific message of the first.
    """
    return isinstance(value, str) and value.startswith(_PLACEHOLDER_PREFIXES)


# Fields that can carry released answer text in the OUTER state. `result` is the
# fallback the envelope reads when `formatted_output` is falsy;
# `regulation_answer` is the same text under its domain name.
OUTER_OUTPUT_FIELDS = ("formatted_output", "result", "regulation_answer")

# The same, inside the domain workflow, before the answer reaches outer state.
INNER_OUTPUT_FIELDS = ("formatted_answer",)


def withheld_delta(stub: str, fields: Sequence[str]) -> Dict[str, Any]:
    """Return the state delta that withholds an answer at a boundary.

    `stub` is written over every field in `fields` and must be non-empty — an
    empty or None placeholder re-opens the fallback this exists to close. The
    structured citation list is dropped in the same delta: it is the other
    half of the answer, and half an answer released under an error status is
    still a release.
    """
    if not stub:
        raise ValueError("withheld_delta: the placeholder must be non-empty")
    delta: Dict[str, Any] = {field: stub for field in fields}
    delta["citations"] = None
    return delta


# The standing, mandatory legal-advice disclaimer. Defined here rather than in
# the node that renders it, so the check and the composition cannot drift apart.
LEGAL_DISCLAIMER = (
    "This answer is generated from a seeded government law and regulation "
    "knowledge base for informational purposes only and does not constitute "
    "legal advice or a formal regulatory determination. Verify against the "
    "primary statutory text and consult your legal or compliance team before "
    "acting on it."
)

# A distinctive, stable phrase from the disclaimer. Matching on this rather than
# on the whole paragraph keeps the check robust to line wrapping while still
# being specific enough that ordinary answer prose cannot satisfy it.
_DISCLAIMER_MARKER = "does not constitute legal advice"

# Numbered citation markers in the answer body: [1], [2], ...
_CITATION_MARKER_RE = re.compile(r"\[(\d{1,3})\]")

# Markdown structure characters removed when caller text is echoed back into
# the rendered answer.
_ECHO_STRIP_RE = re.compile(r"[`*_#|<>]")
_WHITESPACE_RE = re.compile(r"\s+")

# A bracketed number in caller text would read as a citation marker. These are
# removed to a fixed point rather than in a single pass, because one pass over
# a nested form such as "[1[2]]" leaves a valid marker behind. Brackets that do
# not enclose a number are left alone — the redaction marker the platform writes
# over detected personal data is one of them, and it should stay legible.
_ECHO_MARKER_RE = re.compile(r"\[\s*\d{1,3}\s*\]")
_MAX_ECHO_MARKER_PASSES = 5

# Longest caller question echoed into the answer lead sentence.
MAX_ECHO_CHARS = 200


def sanitize_echo(text: Any) -> str:
    """Make caller text safe to render inside the answer.

    The question is echoed back so the answer reads as a reply to it, which
    means caller text becomes report content. Collapsing all whitespace to
    single spaces prevents a multi-line payload from forging headings or list
    items; removing markdown characters and bracketed numbers prevents it from
    forging a citation marker that no retrieved passage backs.
    """
    if not isinstance(text, str) or not text:
        return ""
    cleaned = _ECHO_STRIP_RE.sub(" ", text)
    for _ in range(_MAX_ECHO_MARKER_PASSES):
        stripped = _ECHO_MARKER_RE.sub(" ", cleaned)
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if len(cleaned) > MAX_ECHO_CHARS:
        cleaned = cleaned[:MAX_ECHO_CHARS].rstrip() + "..."
    return cleaned


def citation_markers(text: Any) -> Set[int]:
    """Return the set of citation reference numbers cited in `text`."""
    if not isinstance(text, str) or not text:
        return set()
    return {int(m) for m in _CITATION_MARKER_RE.findall(text)}


def check_citations_wellformed(citations: Any) -> Optional[str]:
    """Every citation entry must name a real, attributed source."""
    if citations is None:
        return None
    if not isinstance(citations, list):
        return "citations_not_a_list"
    for entry in citations:
        if not isinstance(entry, dict):
            return "citation_not_a_record"
        if not str(entry.get("title", "")).strip():
            return "citation_without_title"
        if not str(entry.get("source", "")).strip():
            return "citation_without_source"
        if not str(entry.get("id", "")).strip():
            return "citation_without_source_id"
    return None


def check_grounding(answer: Any, citations: Any) -> Optional[str]:
    """Every cited provision must be traceable to a retrieved passage.

    Returns the name of the first violation found, or None when the answer is
    fully grounded. Two ways an answer can fail: it cites a marker no citation
    record backs, or a citation record does not name its source.
    """
    malformed = check_citations_wellformed(citations)
    if malformed:
        return malformed

    refs = set()
    for entry in citations or []:
        try:
            refs.add(int(entry.get("ref")))
        except (TypeError, ValueError):
            return "citation_without_reference_number"

    cited = citation_markers(answer)
    if cited - refs:
        return "ungrounded_citation_marker"
    return None


def check_sources_were_retrieved(citations: Any, retrieved: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Every citation must name a passage that retrieval actually returned.

    Only checkable where the retrieved passage set is in scope (inside the
    domain workflow). `retrieved=None` means "not in scope here" and skips.
    """
    if retrieved is None or not isinstance(citations, list):
        return None
    available = {str(doc.get("id", "")) for doc in retrieved if isinstance(doc, dict)}
    for entry in citations:
        if not isinstance(entry, dict):
            return "citation_not_a_record"
        if str(entry.get("id", "")) not in available:
            return "citation_source_was_not_retrieved"
    return None


def check_disclaimer(text: Any) -> Optional[str]:
    """The standing legal-advice disclaimer must be present in the answer."""
    if not isinstance(text, str) or _DISCLAIMER_MARKER not in text:
        return "missing_legal_disclaimer"
    return None


def check_rendered_answer(answer: Any, citations: Any) -> Optional[str]:
    """Both outer-boundary invariants over the rendered answer, in one call."""
    return check_disclaimer(answer) or check_grounding(answer, citations)
