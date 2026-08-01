"""Prompt optimization strategies.

Each strategy implements the ``PromptOptimizationStrategy`` protocol
(``reflect`` + ``mutate``) and is selected per section via config keys
(see ``victor.config.prompt_optimization_settings``). The descriptions below
state what each actually does today, not the paper it is named after — the
config keys (``miprov2``, ``cot_distillation``, ``prefpo``) are stable and
load-bearing, so the classes keep their names while the docs stay honest.

- MIPROv2Strategy (key ``miprov2``): query-aware KNN few-shot retriever. Mines
  successful traces, ranks by embedding similarity to the current query, and
  emits a bounded few-shot block. This is few-shot *demonstration mining*, not
  the MIPROv2 instruction/Bayesian proposal search.
- CoTDistillationStrategy (key ``cot_distillation``): distils a reasoning
  scaffold from a strong source trace's real tool trajectory into guidance for
  a weaker provider (prompt-layer transfer, not student-model training).
- PrefPOStrategy (key ``prefpo``): deterministic, section-scoped pairwise
  refiner. Proposes a challenger, judges it on failure coverage vs. bloat, and
  emits an additive candidate only when the challenger wins. No LLM, no learned
  preference model — a heuristic adaptation of the PrefPO framing.
"""

from victor.framework.rl.learners.strategies.miprov2_strategy import (
    MIPROv2Strategy,
)
from victor.framework.rl.learners.strategies.cot_distillation_strategy import (
    CoTDistillationStrategy,
)
from victor.framework.rl.learners.strategies.prefpo_strategy import (
    PrefPOStrategy,
)

__all__ = ["MIPROv2Strategy", "CoTDistillationStrategy", "PrefPOStrategy"]
