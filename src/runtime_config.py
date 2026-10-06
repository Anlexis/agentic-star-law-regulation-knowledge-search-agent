"""AgentCore Platform v1.0"""

# Single reader for config/config.yaml — the runtime-parameter file that sits
# beside the static manifest (config/agent.yaml).
#
# The manifest is a registry entry: identity, entry point, trust level and the
# compile-time `requires` declarations. It carries no runtime tuning, so code
# that wants `max_retry`, `timeout_s` or the `retrieval` block must read THIS
# file. Every consumer in the repo goes through load_runtime_config() so there
# is exactly one place where the file path and the failure behaviour are
# defined.
#
# Failure behaviour: an unreadable or malformed file yields an empty mapping.
# Callers then fall back to their own documented defaults rather than crashing
# a deployment over a config read, and the caller-visible behaviour is the
# shipped default tuning — never silently different numbers.

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

# src/runtime_config.py -> parents[1] = repo root.
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_runtime_config() -> Dict[str, Any]:
    """Return config/config.yaml as a mapping (empty when unreadable).

    Cached: the file is deployment-static, and both the HTTP entry point and
    the graph read it. The returned mapping is shared, so callers copy before
    mutating; nothing in this repo mutates it.
    """
    try:
        import yaml

        loaded = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


# Retrieval tuning used when config/config.yaml cannot be read, or when a value
# in it is unusable. These mirror the `retrieval` block shipped in that file, so
# a config problem degrades to the documented shipped behaviour — never to an
# accidental retune.
DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "top_k": 4,
    "score_threshold": 0.25,
    "kb_path": "config/kb/gov_regulations_kb.json",
}

# Bounds the tuning must satisfy, whatever the file says.
_TOP_K_BOUNDS = (1, 20)
_SCORE_THRESHOLD_BOUNDS = (0.0, 1.0)


def _bounded_number(value: Any, low: float, high: float, default: float) -> float:
    """Return `value` when it is a finite number inside [low, high], else `default`.

    Deployment config is not caller data, so an unusable value falls back to the
    shipped default rather than rejecting the request. It is still checked for
    finiteness explicitly: NaN compares False against every bound, so a clamp
    written as min/max would pass it straight through to the code that decides
    which passages are relevant enough to cite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number) or number < low or number > high:
        return default
    return number


def resolve_retrieval_tuning(raw: Any) -> Dict[str, Any]:
    """Normalise a `retrieval` mapping into finite, in-bounds tuning values.

    `raw` is whatever was seeded into state from config/config.yaml. Anything
    missing or unusable is replaced by the corresponding DEFAULT_RETRIEVAL entry,
    so the returned mapping is always complete and always safe to act on.
    """
    source = raw if isinstance(raw, dict) else {}
    kb_path = source.get("kb_path")
    return {
        "top_k": int(_bounded_number(source.get("top_k"), *_TOP_K_BOUNDS, float(DEFAULT_RETRIEVAL["top_k"]))),
        "score_threshold": _bounded_number(
            source.get("score_threshold"),
            *_SCORE_THRESHOLD_BOUNDS,
            float(DEFAULT_RETRIEVAL["score_threshold"]),
        ),
        "kb_path": kb_path if isinstance(kb_path, str) and kb_path.strip() else str(DEFAULT_RETRIEVAL["kb_path"]),
    }
