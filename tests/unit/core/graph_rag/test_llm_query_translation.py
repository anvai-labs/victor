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

"""LLM-based query translation (closes the PH3-006 TODO).

Pins the injectable completion contract: without an LLM the translator is
honestly template-only (the old stub claimed availability it never had), with
one it translates and every failure shape falls back instead of erroring on
the query path.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.core.graph_rag.query_translation import (
    LLMBasedTranslator,
    TranslationResult,
)


def _store():
    store = MagicMock()
    store.stats = AsyncMock(return_value={"nodes": 100, "edges": 250})
    return store


@pytest.mark.asyncio
async def test_no_llm_routes_to_template_fallback():
    translator = LLMBasedTranslator()  # no llm injected
    result = await translator.translate("find callers of parse_json", _store())
    assert result.metadata["llm_used"] is False
    # The old stub set _llm_available = True while never calling anything.
    assert translator._llm_available is False


@pytest.mark.asyncio
async def test_llm_translates_bare_json():
    llm = AsyncMock(return_value='{"type": "callers", "params": {"function": "parse_json"}}')
    translator = LLMBasedTranslator(llm=llm)

    result = await translator.translate("who calls parse_json?", _store())

    llm.assert_awaited_once()
    prompt = llm.await_args.args[0]
    assert "who calls parse_json?" in prompt
    assert "100" in prompt  # graph stats reached the prompt

    assert result.metadata["llm_used"] is True
    assert result.metadata["query_type"] == "callers"
    assert result.parameters == {"function": "parse_json"}
    assert result.graph_query == "callers(function='parse_json')"
    assert result.confidence == pytest.approx(0.9)
    assert result.is_successful()


@pytest.mark.asyncio
async def test_llm_translates_fenced_json_with_prose():
    llm = AsyncMock(
        return_value=(
            "Sure! Here is the structured query:\n"
            '```json\n{"type": "impact", "params": {"target": "auth.py"}}\n```\n'
            "Let me know if you need anything else."
        )
    )
    translator = LLMBasedTranslator(llm=llm)
    result = await translator.translate("what breaks if I change auth.py", _store())
    assert result.metadata["llm_used"] is True
    assert result.metadata["query_type"] == "impact"
    assert result.parameters == {"target": "auth.py"}


@pytest.mark.asyncio
async def test_unknown_query_type_falls_back():
    llm = AsyncMock(return_value='{"type": "drop_tables", "params": {}}')
    translator = LLMBasedTranslator(llm=llm)
    result = await translator.translate("find callers of parse_json", _store())
    assert result.metadata["llm_used"] is False
    assert result.metadata["llm_error"] == "unparseable LLM response"


@pytest.mark.asyncio
async def test_unparseable_response_falls_back():
    llm = AsyncMock(return_value="I'm sorry, I can't help with that.")
    translator = LLMBasedTranslator(llm=llm)
    result = await translator.translate("find callers of parse_json", _store())
    assert result.metadata["llm_used"] is False
    assert result.metadata["llm_error"] == "unparseable LLM response"
    # The fallback still produced a usable translation path.
    assert isinstance(result, TranslationResult)


@pytest.mark.asyncio
async def test_llm_call_error_falls_back():
    llm = AsyncMock(side_effect=RuntimeError("provider down"))
    translator = LLMBasedTranslator(llm=llm)
    result = await translator.translate("find callers of parse_json", _store())
    assert result.metadata["llm_used"] is False
    assert "provider down" in result.metadata["llm_error"]


@pytest.mark.asyncio
async def test_stats_failure_does_not_block_translation():
    store = MagicMock()
    store.stats = AsyncMock(side_effect=RuntimeError("no stats"))
    llm = AsyncMock(return_value='{"type": "semantic_search", "params": {"query": "auth"}}')
    translator = LLMBasedTranslator(llm=llm)
    result = await translator.translate("auth handling", store)
    assert result.metadata["llm_used"] is True
    assert result.metadata["query_type"] == "semantic_search"


def test_is_successful_recognizes_llm_results():
    # LLM results have neither a matched template nor the fallback flag.
    result = TranslationResult(original_query="q", graph_query="callers(function='f')")
    assert result.is_successful()
    assert not TranslationResult(original_query="q").is_successful()
