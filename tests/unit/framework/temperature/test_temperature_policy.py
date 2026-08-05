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

"""Tests for the unified temperature policy (ADR-013): sources, modifiers, resolver, ratchet."""

from types import SimpleNamespace

from victor.framework.temperature import (
    GlobalDefaultSource,
    ModelBoundsModifier,
    ProfileBaseSource,
    ProfilePerTaskSource,
    RatchetState,
    RatchetStateRegistry,
    RecoveryAdjustModifier,
    SettingsPerTaskSource,
    SpinRatchetModifier,
    SpinSignal,
    TaskHintConstantSource,
    TemperatureContext,
    TemperatureRequest,
    TemperatureResolver,
    build_default_resolver,
)


def _req(**kw) -> TemperatureRequest:
    return TemperatureRequest(**kw)


# --- sources -------------------------------------------------------------------------------------


def test_profile_per_task_source():
    s = ProfilePerTaskSource()
    assert s.resolve(_req(task_type="plan", profile_task_temperatures={"plan": 0.5})) == 0.5
    assert s.resolve(_req(task_type="plan")) is None  # defer when absent


def test_settings_per_task_source():
    s = SettingsPerTaskSource()
    assert (
        s.resolve(_req(task_type="analyze", settings_task_temperatures={"analyze": 0.55})) == 0.55
    )
    assert s.resolve(_req(task_type="analyze")) is None


def test_task_hint_constant_source_reads_provider():
    provider = SimpleNamespace(
        get_hint=lambda t: (SimpleNamespace(temperature_override=0.1) if t == "debug" else None)
    )
    s = TaskHintConstantSource(provider)
    assert s.resolve(_req(task_type="debug")) == 0.1
    assert s.resolve(_req(task_type="unknown")) is None
    assert TaskHintConstantSource(None).resolve(_req(task_type="debug")) is None


def test_profile_base_and_global_default_sources():
    assert ProfileBaseSource().resolve(_req(profile_temperature=0.7)) == 0.7
    assert ProfileBaseSource().resolve(_req()) is None
    assert GlobalDefaultSource(0.6).resolve(_req()) == 0.6


# --- resolver precedence + totality --------------------------------------------------------------


def test_resolver_precedence_profile_beats_settings_beats_hint_beats_base():
    provider = SimpleNamespace(get_hint=lambda t: SimpleNamespace(temperature_override=0.3))
    resolver = build_default_resolver(hint_provider=provider)
    # profile-per-task wins over everything
    r = resolver.resolve(
        _req(
            task_type="plan",
            model_name="claude",
            profile_temperature=0.7,
            profile_task_temperatures={"plan": 0.5},
            settings_task_temperatures={"plan": 0.55},
        )
    )
    assert r.value == 0.5 and r.source_name == "profile_per_task"
    # without profile map: settings wins
    r2 = resolver.resolve(
        _req(
            task_type="plan",
            model_name="claude",
            profile_temperature=0.7,
            settings_task_temperatures={"plan": 0.55},
        )
    )
    assert r2.value == 0.55 and r2.source_name == "settings_per_task"
    # without settings: task-hint constant
    r3 = resolver.resolve(_req(task_type="plan", model_name="claude", profile_temperature=0.7))
    assert r3.value == 0.3 and r3.source_name == "task_hint_constant"


def test_resolver_falls_back_to_profile_base_then_global():
    resolver = build_default_resolver(hint_provider=None)
    r = resolver.resolve(_req(task_type="x", model_name="claude", profile_temperature=0.42))
    assert r.value == 0.42 and r.source_name == "profile_base"
    r2 = resolver.resolve(_req(task_type="x", model_name="claude"))
    assert r2.value == 0.6 and r2.source_name == "global_default"


def test_resolver_total_function_with_no_sources():
    resolver = TemperatureResolver(sources=[], modifiers=[], global_default=0.6)
    assert resolver.resolve(_req()).value == 0.6


# --- ratchet modifier + state --------------------------------------------------------------------


def test_ratchet_state_advances_and_resets():
    st = RatchetState()
    st.record_turn(SpinSignal(spin_state="warning"))
    st.record_turn(SpinSignal(spin_state="stuck"))
    assert st.steps == 2
    st.record_turn(SpinSignal(made_progress=True))  # progress resets
    assert st.steps == 0
    st.record_turn(SpinSignal(plateaued=True))
    assert st.steps == 1
    st.record_turn(SpinSignal(spin_state="normal"))  # normal does not advance
    assert st.steps == 1


def test_ratchet_modifier_steps_and_caps():
    mod = SpinRatchetModifier(step=0.05, cap=0.9)
    st = RatchetState(steps=2)
    ctx = TemperatureContext(ratchet_state=st)
    value, reason = mod.adjust(0.6, _req(), ctx)
    assert abs(value - 0.70) < 1e-9 and "ratchet" in reason
    # cap honoured
    st.steps = 100
    capped, _ = mod.adjust(0.6, _req(), ctx)
    assert capped == 0.9


def test_ratchet_modifier_idempotent_and_disabled():
    mod = SpinRatchetModifier(step=0.05, cap=0.9)
    ctx = TemperatureContext(ratchet_state=RatchetState(steps=3))
    a = mod.adjust(0.6, _req(), ctx)[0]
    b = mod.adjust(0.6, _req(), ctx)[0]  # no double-step
    assert a == b
    assert mod.adjust(0.6, _req(), TemperatureContext())[0] == 0.6  # no state -> passthrough
    off = SpinRatchetModifier(enabled=False)
    assert off.adjust(0.6, _req(), ctx)[0] == 0.6


def test_registry_record_turn_isolates_sessions():
    reg = RatchetStateRegistry()
    reg.record_turn("s1", SpinSignal(spin_state="warning"))
    reg.record_turn("s1", SpinSignal(spin_state="stuck"))
    reg.record_turn("s2", SpinSignal(spin_state="warning"))
    assert reg.get_or_create("s1").steps == 2
    assert reg.get_or_create("s2").steps == 1
    reg.discard("s1")
    assert reg.get_or_create("s1").steps == 0


# --- bounds + recovery modifiers -----------------------------------------------------------------


def test_model_bounds_clamps():
    mod = ModelBoundsModifier()
    # qwen bounds (0.3, 0.9)
    assert mod.adjust(0.95, _req(model_name="qwen3.5"), TemperatureContext())[0] == 0.9
    assert mod.adjust(0.1, _req(model_name="qwen3.5"), TemperatureContext())[0] == 0.3
    # unknown model -> full range, unchanged
    assert mod.adjust(0.95, _req(model_name="zzz"), TemperatureContext())[0] == 0.95


def test_recovery_modifier_passthrough_and_delegate():
    adjuster = SimpleNamespace(get_adjusted_temperature=lambda ctx, sid: (0.85, "STUCK_LOOP +0.2"))
    mod = RecoveryAdjustModifier(adjuster)
    # no recovery context -> passthrough
    assert mod.adjust(0.6, _req(), TemperatureContext())[0] == 0.6
    # active recovery -> delegate
    v, reason = mod.adjust(
        0.6, _req(), TemperatureContext(recovery_context=object(), session_id="s")
    )
    assert v == 0.85 and "recovery" in reason


def test_full_resolver_ratchet_then_bounds_ordering():
    # base 0.6 (global), +ratchet 2*0.05=0.10 -> 0.70, qwen bounds keep it (<=0.9)
    resolver = build_default_resolver(hint_provider=None, ratchet_step=0.05, ratchet_cap=0.9)
    ctx = TemperatureContext(ratchet_state=RatchetState(steps=2))
    r = resolver.resolve(_req(task_type="x", model_name="qwen3.5"), ctx)
    assert abs(r.value - 0.70) < 1e-9
    names = [m[0] for m in r.modifier_trace]
    assert names == ["spin_ratchet", "model_bounds"]  # recovery absent (no adjuster)


def test_kimi_k3_temperature_is_pinned_to_one():
    """kimi-k3 rejects any temperature != 1.0; its bounds collapse to a point so
    the terminal clamp forces 1.0 regardless of profile/default/spin (no profile
    need carry temperature: 1.0)."""
    from victor.framework.temperature.defaults import model_bounds

    assert model_bounds("kimi-k3") == (1.0, 1.0)

    mod = ModelBoundsModifier()
    # A low profile/default value is forced up to 1.0.
    assert mod.adjust(0.6, _req(model_name="kimi-k3"), TemperatureContext())[0] == 1.0
    # A high value is forced down to 1.0.
    assert mod.adjust(2.0, _req(model_name="kimi-k3"), TemperatureContext())[0] == 1.0


def test_kimi_k3_pin_overrides_profile_and_ratchet_end_to_end():
    """Even an explicit profile temperature and an active spin ratchet cannot move
    kimi-k3 off 1.0 — the terminal model_bounds clamp runs last."""
    resolver = build_default_resolver(hint_provider=None, ratchet_step=0.05, ratchet_cap=0.9)
    ctx = TemperatureContext(ratchet_state=RatchetState(steps=3))
    r = resolver.resolve(_req(task_type="x", model_name="kimi-k3", profile_temperature=0.3), ctx)
    assert r.value == 1.0
    assert r.modifier_trace[-1][0] == "model_bounds"
