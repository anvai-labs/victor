# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token usage display bar for the TUI.

Full-fidelity panel: absolute context limit with a proportional gauge,
per-turn token breakdown (prompt / cached / reasoning / generative),
cache hit rate, and cost (last turn + session cumulative).
"""

from __future__ import annotations

from typing import Optional

from textual.widgets import Static


def _human_tokens(n: float) -> str:
    """Render a token count compactly (e.g. ``12.4k``)."""
    if n < 1000:
        return f"{int(n)}"
    return f"{n / 1000:.1f}k"


def _ctx_color(pct: float) -> str:
    if pct < 60:
        return "green"
    if pct < 85:
        return "yellow"
    return "red"


def _gauge(pct: float, width: int = 14) -> str:
    """Render a proportional unicode gauge; solid fill = used share."""
    filled = max(0, min(width, round(pct / 100.0 * width)))
    return "▓" * filled + "░" * (width - filled)


class TokenUsageBar(Static):
    """Widget displaying context, token breakdown, cache and cost telemetry."""

    DEFAULT_CSS = """
    TokenUsageBar {
        height: 2;
        padding: 0 1;
        background: $panel;
        color: $foreground;
    }
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self._session_cost_usd: float = 0.0
        self._session_requests: int = 0

    # ── public API ────────────────────────────────────────────────
    def set_usage(
        self,
        *,
        # context — absolute limit + used, plus derived/override pct
        context_used_tokens: Optional[int] = None,
        context_limit_tokens: Optional[int] = None,
        context_used_pct: Optional[float] = None,
        # last-turn breakdown
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        cache_write_tokens: Optional[int] = None,
        cache_hit_rate: Optional[float] = None,
        total_tokens: Optional[int] = None,
        # money + requests
        cost_usd: Optional[float] = None,
        session_cost_usd: Optional[float] = None,
        request_count: Optional[int] = None,
        model: Optional[str] = None,
    ) -> None:
        """Update the token usage display.

        Args:
            context_used_tokens: Tokens currently resident in the context window.
            context_limit_tokens: Absolute context-window limit for this model.
            context_used_pct: Percentage of context window used (fallback).
            prompt_tokens: Last-turn input tokens.
            completion_tokens: Last-turn output tokens (total).
            reasoning_tokens: Last-turn reasoning/thinking tokens (subset of output).
            cached_tokens: Last-turn cache-read input tokens (subset of input).
            cache_write_tokens: Last-turn cache-write input tokens.
            cache_hit_rate: Fraction of input served from cache (0..1).
            total_tokens: Last-turn total tokens (computed when omitted).
            cost_usd: Last-turn cost in USD.
            session_cost_usd: Session cumulative cost (accumulated locally when omitted).
            request_count: Last-turn provider request count (accumulated when omitted).
            model: Current model label.
        """
        # session accumulation (caller may pass authoritative totals instead)
        if session_cost_usd is not None:
            self._session_cost_usd = float(session_cost_usd)
        elif cost_usd is not None:
            self._session_cost_usd += float(cost_usd)
        if request_count is not None:
            self._session_requests += int(request_count)

        line1 = self._render_line1(
            context_used_tokens, context_limit_tokens, context_used_pct, model
        )
        line2 = self._render_line2(
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            cached_tokens,
            cache_write_tokens,
            cache_hit_rate,
            total_tokens,
            cost_usd,
        )

        if not line1 and not line2:
            self.update("[dim italic]No usage data yet[/]")
        else:
            self.update(f"{line1}\n{line2}")

    # ── renderers ─────────────────────────────────────────────────
    def _render_line1(
        self,
        used: Optional[int],
        limit: Optional[int],
        pct: Optional[float],
        model: Optional[str],
    ) -> str:
        parts: list[str] = []
        if model:
            parts.append(f"[bold cyan]{model}[/]")
        if limit and limit > 0 and used is not None and used >= 0:
            ratio = min(100.0, used / limit * 100.0)
            color = _ctx_color(ratio)
            parts.append(
                f"[dim]Ctx[/] [{color}]{_gauge(ratio)}[/] "
                f"[{color}]{_human_tokens(used)}/{_human_tokens(limit)}[/] "
                f"[dim]{ratio:.0f}%[/]"
            )
        elif pct is not None:
            color = _ctx_color(pct)
            parts.append(f"[dim]Ctx[/] [{color}]{_gauge(pct)} {pct:.0f}%[/]")
        return "  ".join(parts)

    def _render_line2(
        self,
        prompt: Optional[int],
        completion: Optional[int],
        reasoning: Optional[int],
        cached: Optional[int],
        cache_write: Optional[int],
        hit_rate: Optional[float],
        total: Optional[int],
        cost: Optional[float],
    ) -> str:
        parts: list[str] = []

        # input side with cache annotation
        if prompt is not None and prompt >= 0:
            seg = f"[dim]In[/] [bold]{_human_tokens(prompt)}[/]"
            if cached:
                seg += f" [dim](cache R {_human_tokens(cached)}"
                if cache_write:
                    seg += f" · W {_human_tokens(cache_write)}"
                if hit_rate is not None and prompt > 0:
                    seg += f" · hit {hit_rate * 100:.0f}%"
                seg += ")[/]"
            parts.append(seg)

        # output side: reasoning vs generative split
        if completion is not None and completion >= 0:
            seg = f"[dim]Out[/] [bold]{_human_tokens(completion)}[/]"
            if reasoning:
                gen = max(0, completion - reasoning)
                seg += (
                    f" [dim](reason {_human_tokens(reasoning)} · " f"gen {_human_tokens(gen)})[/]"
                )
            parts.append(seg)

        if total is None and (prompt is not None or completion is not None):
            total = (prompt or 0) + (completion or 0)
        if total:
            parts.append(f"[dim]Turn[/] {_human_tokens(total)}")

        # expenses
        if cost is not None:
            parts.append(f"[dim]Turn $[/][bold]{cost:.4f}[/]")
        if self._session_cost_usd:
            parts.append(f"[dim]Session $[/][bold]{self._session_cost_usd:.4f}[/]")
        if self._session_requests:
            parts.append(f"[dim]{self._session_requests} req[/]")

        return "  ·  ".join(parts)
