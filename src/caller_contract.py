"""AgentCore Platform v1.0"""

# GOV-C2-005 — the caller-data contract.
#
# Everything a caller can influence arrives through one of two channels: the
# question string, or the `input_context` mapping the platform forwards from the
# invocation. This module is the single place where those values are turned into
# values the pipeline is allowed to act on. PreProcessNode owns the contract and
# calls into here; no other module re-derives these rules.
#
# Three rules hold for every field:
#
#   finite and bounded   Numbers are parsed with an explicit finite check before
#                        the range check. `float("nan")` and `float("inf")` parse
#                        cleanly and arrive verbatim through JSON request bodies,
#                        and every comparison against NaN evaluates False — so a
#                        NaN relevance floor would keep every passage rather than
#                        raise, which is a silent failure on exactly the decision
#                        this template exists to make.
#   inert                Strings that survive into state are restricted to a
#                        short lowercase identifier. Free text from the context
#                        channel never reaches the rendered answer, so a caller
#                        cannot use an option field to inject report content, and
#                        no identifier a caller pastes into an option can survive
#                        into a stored field.
#   fail closed          A value that fails its bounds raises. The request is
#                        rejected naming the FIELD; the offending value is never
#                        echoed into an error, a log line or an audit event.

import math
import re
from typing import Any, Callable, Dict, Mapping, Optional

# KB categories the seeded corpus uses (config/kb/gov_regulations_kb.json).
CATEGORY_VALUES = frozenset({"act", "cabinet_order", "ministerial_ordinance", "administrative_guidance"})

# Bounds for the caller-supplied retrieval overrides.
TOP_K_MIN = 1
TOP_K_MAX = 20
SCORE_THRESHOLD_MIN = 0.0
SCORE_THRESHOLD_MAX = 1.0

# Every caller string that is kept must match this. Short, lowercase, no
# punctuation and no whitespace — nothing that can carry markup or a sentence.
IDENTIFIER_RE = re.compile(r"^[a-z0-9_]{1,32}$")

# Defensive structural cap on the context mapping itself.
MAX_CONTEXT_KEYS = 32

# Option keys this template understands. Anything else in the mapping is
# ignored rather than rejected — the platform is free to add its own envelope
# keys, and rejecting them would couple this template to a platform version.
_OPTION_KEYS = ("category", "top_k", "score_threshold", "channel")

# Attack forms refused before the question reaches any processing. Each is a
# PHRASE, not a keyword: "record management system requirements" and "what does
# the Act instruct" are ordinary questions in this domain and must pass through
# untouched, so a bare "system" or "instruction" must never match.
_INJECTION_PATTERNS = [
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(?:previous|prior|above|earlier|all|your)\b[^.\n]{0,40}"
            r"\b(?:instruction|instructions|prompt|prompts|rule|rules|direction|directions)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_disclosure",
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|disclose|display)\b[^.\n]{0,40}"
            r"\b(?:system|initial|original|hidden)\b[^.\n]{0,20}\bprompt\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\byou\s+are\s+now\b"
            r"|\bact\s+as\s+(?:a\s+|an\s+)?(?:different|new|unrestricted|unfiltered)\b"
            r"|\bpretend\s+(?:to\s+be|you\s+are)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "chat_role_marker",
        re.compile(
            r"<\|\s*(?:im_start|im_end|system|endoftext)\s*\|>" r"|^[ \t]*(?:system|assistant)[ \t]*:[ \t]",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
]


class CallerContractError(ValueError):
    """A caller-supplied value failed its bounds.

    Carries the FIELD name only. The rejected value is deliberately absent so
    it cannot travel into an error response, a log line or an audit event.
    """

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"caller option '{field}' rejected: {reason}")


def detect_injection(text: Any) -> Optional[str]:
    """Return the name of the first refused pattern in `text`, else None."""
    if not isinstance(text, str) or not text:
        return None
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return name
    return None


def scan_context_for_injection(context: Any) -> Optional[str]:
    """Return the first refused pattern found in any string value of `context`.

    The context channel is scanned as thoroughly as the question is: a payload
    that would be refused in the question must not simply be moved into an
    option field to get through.
    """
    if not isinstance(context, Mapping):
        return None
    for key, value in list(context.items())[:MAX_CONTEXT_KEYS]:
        found = detect_injection(key if isinstance(key, str) else "") or detect_injection(value)
        if found:
            return found
    return None


def finite_in_range(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
    integer: bool = False,
) -> float:
    """Parse a caller number that must be finite and inside [minimum, maximum].

    Rejects, in order: booleans (`True` is an `int` in Python, so `top_k: true`
    would otherwise pass as 1), non-numeric values, non-finite values
    (NaN / +Infinity / -Infinity, whether they arrive as raw JSON literals or as
    the strings "NaN" / "Infinity"), non-integral values where an integer is
    required, and finally out-of-range magnitudes.
    """
    if isinstance(value, bool):
        raise CallerContractError(field, "must be a number")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise CallerContractError(field, "must be a number")
        try:
            number = float(stripped)
        except ValueError:
            raise CallerContractError(field, "must be a number") from None
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise CallerContractError(field, "must be a number")

    if not math.isfinite(number):
        raise CallerContractError(field, "must be a finite number")
    if integer and number != int(number):
        raise CallerContractError(field, "must be a whole number")
    if number < minimum or number > maximum:
        raise CallerContractError(field, f"must be between {minimum} and {maximum}")
    return number


def parse_identifier(value: Any, *, field: str) -> str:
    """Parse a caller string that is kept, restricting it to an inert identifier."""
    if not isinstance(value, str):
        raise CallerContractError(field, "must be a string")
    candidate = value.strip().lower()
    if not IDENTIFIER_RE.match(candidate):
        raise CallerContractError(field, "must be 1-32 characters of a-z, 0-9 or underscore")
    return candidate


def parse_category(value: Any, *, field: str = "category") -> str:
    """Parse the KB category filter — an inert identifier from a closed set."""
    candidate = parse_identifier(value, field=field)
    if candidate not in CATEGORY_VALUES:
        raise CallerContractError(field, "is not a recognised regulation category")
    return candidate


def parse_top_k(value: Any, *, field: str = "top_k") -> int:
    """Parse the caller's cap on how many passages are cited."""
    return int(finite_in_range(value, field=field, minimum=TOP_K_MIN, maximum=TOP_K_MAX, integer=True))


def parse_score_threshold(value: Any, *, field: str = "score_threshold") -> float:
    """Parse the caller's relevance floor."""
    return finite_in_range(
        value,
        field=field,
        minimum=SCORE_THRESHOLD_MIN,
        maximum=SCORE_THRESHOLD_MAX,
    )


def parse_caller_options(raw: Any) -> Dict[str, Any]:
    """Validate every option this template accepts from a caller mapping.

    Absent options are simply absent from the result — the pipeline then uses
    its configured tuning. A PRESENT option that fails its bounds raises, so a
    caller never receives an answer computed from a value the template quietly
    replaced with something else.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise CallerContractError("input_context", "must be a mapping")
    if len(raw) > MAX_CONTEXT_KEYS:
        raise CallerContractError("input_context", f"must carry at most {MAX_CONTEXT_KEYS} keys")

    parsers: Dict[str, Callable[[Any], Any]] = {
        "category": parse_category,
        "top_k": parse_top_k,
        "score_threshold": parse_score_threshold,
        "channel": lambda v: parse_identifier(v, field="channel"),
    }
    options: Dict[str, Any] = {}
    for key in _OPTION_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if value is None:
            continue
        options[key] = parsers[key](value)
    return options
