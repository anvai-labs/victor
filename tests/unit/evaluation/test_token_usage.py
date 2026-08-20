# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for extended TokenUsage tracking across providers."""

from victor.evaluation.protocol import TokenUsage


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_basic_creation(self):
        t = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert t.input_tokens == 100
        assert t.output_tokens == 50
        assert t.total_tokens == 150

    def test_extended_fields_default_zero(self):
        t = TokenUsage()
        assert t.cached_tokens == 0
        assert t.cache_miss_tokens == 0
        assert t.reasoning_tokens == 0
        assert t.cost_usd_micros == 0

    def test_addition(self):
        t1 = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cached_tokens=10,
            reasoning_tokens=20,
            cost_usd_micros=1000,
        )
        t2 = TokenUsage(
            input_tokens=200,
            output_tokens=80,
            total_tokens=280,
            cached_tokens=30,
            reasoning_tokens=40,
            cost_usd_micros=2000,
        )
        result = t1 + t2
        assert result.input_tokens == 300
        assert result.output_tokens == 130
        assert result.total_tokens == 430
        assert result.cached_tokens == 40
        assert result.reasoning_tokens == 60
        assert result.cost_usd_micros == 3000

    def test_iadd(self):
        t = TokenUsage(input_tokens=100, cached_tokens=10)
        t += TokenUsage(input_tokens=50, cached_tokens=5)
        assert t.input_tokens == 150
        assert t.cached_tokens == 15


class TestFromProviderUsage:
    """Tests for TokenUsage.from_provider_usage() normalization."""

    def test_deepseek_format(self):
        usage = {"prompt_tokens": 308, "completion_tokens": 9, "total_tokens": 317}
        raw = {
            "prompt_tokens": 308,
            "completion_tokens": 9,
            "total_tokens": 317,
            "prompt_tokens_details": {"cached_tokens": 0},
            "prompt_cache_hit_tokens": 50,
            "prompt_cache_miss_tokens": 258,
        }
        t = TokenUsage.from_provider_usage(usage, raw)
        assert t.input_tokens == 308
        assert t.output_tokens == 9
        assert t.cached_tokens == 50
        assert t.cache_miss_tokens == 258

    def test_xai_format(self):
        usage = {"prompt_tokens": 315, "completion_tokens": 10, "total_tokens": 784}
        raw = {
            "prompt_tokens_details": {"text_tokens": 315, "cached_tokens": 5},
            "completion_tokens_details": {"reasoning_tokens": 459},
            "cost_in_usd_ticks": 3278750,
        }
        t = TokenUsage.from_provider_usage(usage, raw)
        assert t.input_tokens == 315
        assert t.total_tokens == 784
        assert t.cached_tokens == 5
        assert t.reasoning_tokens == 459
        assert t.cost_usd_micros == 3278750

    def test_openai_format(self):
        usage = {"prompt_tokens": 314, "completion_tokens": 5, "total_tokens": 319}
        raw = {
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        t = TokenUsage.from_provider_usage(usage, raw)
        assert t.input_tokens == 314
        assert t.cached_tokens == 0
        assert t.reasoning_tokens == 0

    def test_anthropic_format(self):
        usage = {"prompt_tokens": 311, "completion_tokens": 10, "total_tokens": 321}
        raw = {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 100,
            "input_tokens": 311,
            "output_tokens": 10,
        }
        t = TokenUsage.from_provider_usage(usage, raw)
        assert t.input_tokens == 311
        assert t.cached_tokens == 100

    def test_no_raw_usage(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        t = TokenUsage.from_provider_usage(usage)
        assert t.input_tokens == 100
        assert t.cached_tokens == 0
        assert t.reasoning_tokens == 0
