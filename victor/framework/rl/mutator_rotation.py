# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Spread reflect/mutate calls across providers so one rate limit cannot stall a run.

A prompt-evolution pass makes several LLM calls per section, back to back, all to
one provider. On a hosted plan that reliably earns a 429 partway through, and a
429 in the mutate call is not a degraded result — it is *no* result: `mutate()`
falls back to returning its input, and whatever reformatting runs afterwards
becomes the candidate's entire diff. Two full runs were spent diagnosing that as
a strategy problem.

Rotation is per unit of work (a section), so consecutive calls land on different
providers and no single quota absorbs the whole pass. A provider that answers 429
is *benched* for the remainder of the run rather than merely skipped once —
retrying a provider that just told us to back off wastes the call and, worse,
deepens the limit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Substrings that mean "this provider is refusing for now", matched
# case-insensitively against the exception text. Kept broad: a missed match only
# costs one wasted call on the next rotation, while a false positive would bench
# a healthy provider.
RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit", "too many requests", "quota")


def is_rate_limit(error: object) -> bool:
    """True when an error looks like a provider throttling us."""
    text = str(error or "").lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


@dataclass(frozen=True)
class MutatorSpec:
    """One (provider, model) the mutator may run on."""

    provider: str
    model: str
    label: str = ""
    base_url: str = ""

    def display(self) -> str:
        return self.label or f"{self.provider}/{self.model}"


@dataclass
class MutatorRotation:
    """Round-robin over mutator specs, benching those that rate-limit.

    Deliberately not a load balancer: there is no latency or cost model here,
    only "spread the calls and stop asking a provider that said no".
    """

    specs: List[MutatorSpec]
    _cursor: int = field(default=0, init=False)
    _benched: set = field(default_factory=set, init=False)

    @classmethod
    def from_specs(cls, specs: Iterable[MutatorSpec]) -> "MutatorRotation":
        deduped: List[MutatorSpec] = []
        seen = set()
        for spec in specs:
            key = (spec.provider, spec.model)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(spec)
        return cls(specs=deduped)

    def __bool__(self) -> bool:
        return bool(self.specs)

    @property
    def available(self) -> List[MutatorSpec]:
        """Specs not benched by a rate limit this run."""
        return [s for s in self.specs if (s.provider, s.model) not in self._benched]

    def next_spec(self) -> Optional[MutatorSpec]:
        """The spec to use for the next unit of work, or None if all are benched.

        Returning None is meaningful: the caller should stop rather than fall
        through to whatever the session default is, which is how a 2B local model
        ended up rewriting production prompts unnoticed.
        """
        usable = self.available
        if not usable:
            return None
        spec = usable[self._cursor % len(usable)]
        self._cursor += 1
        return spec

    def bench(self, spec: MutatorSpec, reason: object = "") -> None:
        """Take a spec out of rotation for the rest of this run."""
        key = (spec.provider, spec.model)
        if key in self._benched:
            return
        self._benched.add(key)
        logger.warning(
            "Mutator %s benched for this run (%s); %d provider(s) still available.",
            spec.display(),
            str(reason)[:160] or "rate limited",
            len(self.available),
        )

    def note_failure(self, spec: MutatorSpec, error: object) -> bool:
        """Record a failure. Returns True when the spec was benched for throttling.

        Non-throttling failures leave the spec in rotation — a timeout or a
        transient network error says nothing about quota, and benching on those
        would strand a run on its last provider.
        """
        if is_rate_limit(error):
            self.bench(spec, error)
            return True
        logger.debug("Mutator %s failed (not a rate limit): %s", spec.display(), error)
        return False

    def summary(self) -> str:
        """One line for run output: who is in rotation and who was benched."""
        if not self.specs:
            return "no mutator rotation configured"
        benched = [s.display() for s in self.specs if (s.provider, s.model) in self._benched]
        active = [s.display() for s in self.available]
        text = f"rotation: {', '.join(active) if active else 'none available'}"
        if benched:
            text += f" | benched: {', '.join(benched)}"
        return text


def build_rotation(entries: Sequence[tuple]) -> MutatorRotation:
    """Build a rotation from ``(provider, model, label)`` tuples, skipping blanks."""
    specs = [
        MutatorSpec(provider=p, model=m, label=(lbl or ""))
        for p, m, lbl in ((e + ("",))[:3] if len(e) < 3 else e for e in entries)
        if p and m
    ]
    return MutatorRotation.from_specs(specs)


__all__ = [
    "RATE_LIMIT_MARKERS",
    "MutatorRotation",
    "MutatorSpec",
    "build_rotation",
    "is_rate_limit",
]
