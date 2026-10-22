"""Reinforcement-learning event vocabulary shared below the framework.

The RL hook registry and its event types are emitted by infrastructure
(caches, tool selectors) and the runtime alike; the learners and the
coordinator that consume them stay in ``victor.framework.rl``, which wires
the coordinator into this registry and re-exports it unchanged.
"""
