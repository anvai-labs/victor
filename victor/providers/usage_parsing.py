"""Single-sourced provider usage parsing via the sandhi binding (ADR-0047 D10a step 2).

Routes the metering-critical usage/cache-split extraction through ``sandhi_gateway``'s
fixture-proven parsers (Sandhi TD-0001) instead of per-adapter hand parsing — the
"three parsers = three chances to mis-meter" consolidation. The binding is required by the
typed provider runtime; every helper remains defensive and returns ``None`` on failure so legacy callers keep
their local inline parser as the fallback. This is parser compatibility only, never a transport
fallback. Metering must never break chat.

Semantic mapping (victor convention vs sandhi neutral units)
============================================================

sandhi returns ``{tokens_in, tokens_out, cache_creation_tokens, cache_read_tokens}`` with
``tokens_in`` normalized to **fresh-only** (cached tokens excluded). Victor's ``usage``
dict convention is provider-inconsistent today and is preserved exactly:

===============================  =====================  ===============================
victor key                       slug ``anthropic``     slug ``openai`` (compat family)
===============================  =====================  ===============================
``prompt_tokens``                ``tokens_in`` (the     ``tokens_in + cache_read``
                                 SDK ``input_tokens``   (**FULL** prompt incl. cached —
                                 is already fresh-only) context-window/budget logic
                                                        depends on the full count)
``completion_tokens``            ``tokens_out``         ``tokens_out``
``total_tokens``                 prompt + completion    raw ``total_tokens`` if present,
                                                        else prompt + completion
``cache_creation_input_tokens``  iff raw block carries  iff > 0
                                 the field
``cache_read_input_tokens``      iff raw block carries  iff > 0
                                 the field
===============================  =====================  ===============================

Slug ``ollama`` maps eval counts 1:1 (no cache split). The fresh-only neutral basis
enters victor only at the neutral-event layer (``SandhiMeter``), never this dict.

``google``/``vertex`` are intentionally NOT routed: their protobuf ``usage_metadata``
carries pythonized field names, so building sandhi's camelCase envelope by hand would
itself be a hand-written mapping — a fourth parser in disguise.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

try:  # optional dependency (victor[sandhi])
    import sandhi_gateway as _sg  # type: ignore[import-untyped]
except Exception:  # pragma: no cover — absent without the extra
    _sg = None

# victor provider name -> sandhi parser slug for providers with dedicated parsers.
# Anything else that adopts routing uses "openai" (the sandhi default dispatch).
# "google"/"vertex" intentionally absent — see module docstring.
SANDHI_SLUGS: Dict[str, str] = {
    "anthropic": "anthropic",
    "ollama": "ollama",
    "bedrock": "bedrock",
}

# Slugs whose raw usage block sits at the top level of the response (no {"usage": ...}
# envelope). Matches sandhi's parser input shapes (sandhi-core/src/usage.rs).
_TOP_LEVEL_SLUGS = frozenset({"ollama"})


def _coerce(usage_obj: Any) -> Optional[Dict[str, Any]]:
    """Raw usage block as a plain dict, or None. Accepts dicts and pydantic SDK objects."""
    try:
        if usage_obj is None:
            return None
        if isinstance(usage_obj, Mapping):
            return dict(usage_obj)
        model_dump = getattr(usage_obj, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            return dict(dumped) if isinstance(dumped, Mapping) else None
        return None
    except Exception:  # defensive — never fail a call on metering
        return None


def sandhi_parse_usage(slug: str, usage_obj: Any) -> Optional[Dict[str, int]]:
    """Neutral sandhi usage for a raw usage block, or None.

    Returns ``{tokens_in, tokens_out, cache_creation_tokens, cache_read_tokens}``
    (``tokens_in`` fresh-only). ``None`` when the binding is absent, the block cannot
    be coerced, or the parse fails. Never raises.
    """
    if _sg is None:
        return None
    block = _coerce(usage_obj)
    if block is None:
        return None
    try:
        payload = block if slug in _TOP_LEVEL_SLUGS else {"usage": block}
        parsed = _sg.parse_usage(slug, json.dumps(payload))
        return {
            "tokens_in": int(parsed.get("tokens_in", 0) or 0),
            "tokens_out": int(parsed.get("tokens_out", 0) or 0),
            "cache_creation_tokens": int(parsed.get("cache_creation_tokens", 0) or 0),
            "cache_read_tokens": int(parsed.get("cache_read_tokens", 0) or 0),
        }
    except Exception as exc:
        logger.debug("sandhi parse_usage failed for slug=%s (local parser fallback): %s", slug, exc)
        return None


def parse_usage_dict(slug: str, usage_obj: Any) -> Optional[Dict[str, int]]:
    """Victor-convention usage dict via sandhi's parser, or None (caller uses its local parser).

    See the module docstring for the pinned per-slug mapping. Never raises.
    """
    parsed = sandhi_parse_usage(slug, usage_obj)
    if parsed is None:
        return None
    block = _coerce(usage_obj) or {}

    completion = parsed["tokens_out"]
    creation = parsed["cache_creation_tokens"]
    read = parsed["cache_read_tokens"]

    if slug == "anthropic":
        prompt = parsed["tokens_in"]  # SDK input_tokens is already fresh-only
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
        if "cache_creation_input_tokens" in block:
            usage["cache_creation_input_tokens"] = creation
        if "cache_read_input_tokens" in block:
            usage["cache_read_input_tokens"] = read
        return usage

    if slug in _TOP_LEVEL_SLUGS:  # ollama: eval counts, no cache split
        prompt = parsed["tokens_in"]
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    # OpenAI-compat family (and default dispatch): reconstruct the FULL prompt count.
    prompt = parsed["tokens_in"] + read
    try:
        total = int(block["total_tokens"])
    except (KeyError, TypeError, ValueError):
        total = prompt + completion
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    if creation:
        usage["cache_creation_input_tokens"] = creation
    if read:
        usage["cache_read_input_tokens"] = read
    return usage


def usage_dict_from_neutral(
    neutral_usage: Any,
    raw_usage: Any,
    *,
    slug: str = "openai",
) -> Optional[Dict[str, int]]:
    """Map usage already measured by Sandhi into Victor's compatibility shape.

    Unlike :func:`parse_usage_dict`, this does not call back through the binding.
    The transport path has already parsed and trusted these neutral counts at the
    source; reparsing the raw response would duplicate the metering-critical work.
    """
    neutral = _coerce(neutral_usage)
    if neutral is None:
        return None
    try:
        fresh = int(neutral.get("tokens_in", 0) or 0)
        completion = int(neutral.get("tokens_out", 0) or 0)
        creation = int(neutral.get("cache_creation_tokens", 0) or 0)
        read = int(neutral.get("cache_read_tokens", 0) or 0)
        reasoning = int(neutral.get("reasoning_tokens", 0) or 0)
    except (TypeError, ValueError):
        return None

    raw = _coerce(raw_usage) or {}
    if slug == "anthropic":
        prompt = fresh
    elif slug in _TOP_LEVEL_SLUGS:
        prompt = fresh
    else:
        prompt = fresh + read

    try:
        total = int(raw["total_tokens"])
    except (KeyError, TypeError, ValueError):
        total = prompt + completion
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    if creation:
        usage["cache_creation_input_tokens"] = creation
    if read:
        usage["cache_read_input_tokens"] = read
    if reasoning:
        # Observability key. Whether these are folded into completion_tokens
        # (OpenAI/Anthropic) or reported separately (Gemini thoughtsTokenCount)
        # is decided at cost time via the same reasoning > completion heuristic
        # sandhi's billable() uses — not here.
        usage["reasoning_tokens"] = reasoning
    return usage
