# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Timeout calculation strategies for agentic benchmark execution.

Provides configurable timeout policies that ensure sufficient time per turn
while respecting total task timeout constraints.

SOLID Compliance:
- OCP: New timeout policies can be added without modifying existing code
- SRP: Each policy has a single responsibility
- DIP: Policies depend on TimeoutPolicy protocol, not concrete implementations

Usage:
    from victor.evaluation.timeout_calculator import SafeTimeoutPolicy

    policy = SafeTimeoutPolicy()
    turn_timeout = policy.calculate(total_timeout=300, max_turns=10)
    # Returns max(30, 180) = 180 seconds
"""

from dataclasses import dataclass, field
from typing import Protocol


class TimeoutPolicy(Protocol):
    """Strategy protocol for calculating per-turn timeouts (OCP).

    Implementations define how to calculate the timeout for each turn
    given the total task timeout and maximum number of turns.
    """

    def calculate(self, total_timeout: int, max_turns: int) -> int:
        """Calculate per-turn timeout in seconds.

        Args:
            total_timeout: Total timeout for the entire task in seconds
            max_turns: Maximum number of turns expected

        Returns:
            Timeout in seconds for a single turn
        """
        ...


class NaiveTimeoutPolicy:
    """Simple division policy - divides total timeout by turns.

    Warning: May result in very short per-turn timeouts for high turn counts.
    Not recommended for use with slower models (e.g., DeepSeek, Mixtral).
    """

    def calculate(self, total_timeout: int, max_turns: int) -> int:
        """Simple division of total timeout by max turns."""
        if max_turns <= 0:
            return total_timeout
        return max(1, total_timeout // max_turns)


class SafeTimeoutPolicy:
    """Ensures minimum viable timeout per turn.

    Guarantees at least MIN_TURN_TIMEOUT seconds per turn, even if
    this means the total task duration could exceed total_timeout
    in worst case (all turns used).

    This is the recommended policy for benchmarks where model latency
    varies significantly (e.g., cloud APIs, slower local models).
    """

    MIN_TURN_TIMEOUT = 180  # 3 minutes minimum per turn

    def __init__(self, min_turn_timeout: int = 180):
        """Initialize with configurable minimum timeout.

        Args:
            min_turn_timeout: Minimum seconds per turn (default: 180)
        """
        self.min_turn_timeout = min_turn_timeout

    def calculate(self, total_timeout: int, max_turns: int) -> int:
        """Calculate per-turn timeout with minimum enforcement.

        Args:
            total_timeout: Total timeout for the entire task
            max_turns: Maximum number of turns expected

        Returns:
            max(naive_timeout, min_turn_timeout)
        """
        if max_turns <= 0:
            return max(total_timeout, self.min_turn_timeout)

        naive = total_timeout // max_turns
        return max(naive, self.min_turn_timeout)


class AdaptiveTimeoutPolicy:
    """Adaptive policy that increases timeout based on turn depth.

    Provides more time for later turns, which often involve more
    complex operations (test fixing, debugging, etc.).
    """

    def __init__(
        self,
        base_timeout: int = 120,
        growth_factor: float = 1.2,
        max_timeout: int = 600,
    ):
        """Initialize adaptive timeout policy.

        Args:
            base_timeout: Base timeout for first turn
            growth_factor: Multiplicative factor per turn
            max_timeout: Maximum timeout cap
        """
        self.base_timeout = base_timeout
        self.growth_factor = growth_factor
        self.max_timeout = max_timeout

    def calculate(self, total_timeout: int, max_turns: int) -> int:
        """Calculate base timeout (grows with turn number).

        Note: This returns the base timeout. Actual timeout for turn N
        should be: min(base * growth_factor^N, max_timeout)
        """
        return min(self.base_timeout, total_timeout, self.max_timeout)

    def calculate_for_turn(self, turn: int) -> int:
        """Calculate timeout for a specific turn.

        Args:
            turn: Turn number (0-indexed)

        Returns:
            Timeout in seconds for this turn
        """
        timeout = int(self.base_timeout * (self.growth_factor**turn))
        return min(timeout, self.max_timeout)


@dataclass
class AgenticTimeoutConfig:
    """Configuration for agentic task timeouts with safety bounds.

    Provides a unified configuration for timeout handling in benchmark
    evaluation, with sensible defaults and policy-based calculation.
    """

    total_timeout: int = 300  # Total task timeout in seconds
    max_turns: int = 10  # Maximum expected turns
    min_turn_timeout: int = 180  # Minimum per-turn timeout
    policy: TimeoutPolicy = field(default_factory=SafeTimeoutPolicy)

    def __post_init__(self):
        """Initialize policy if needed."""
        if self.policy is None:
            self.policy = SafeTimeoutPolicy(min_turn_timeout=self.min_turn_timeout)

    @property
    def turn_timeout(self) -> int:
        """Calculate per-turn timeout using configured policy."""
        return self.policy.calculate(self.total_timeout, self.max_turns)
