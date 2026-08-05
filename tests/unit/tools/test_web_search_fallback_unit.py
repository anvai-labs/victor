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

"""Unit tests for the DuckDuckGo multi-endpoint fallback and robustness helpers
added to web_search_tool.py (UA rotation, lite endpoint, browser fallback,
trafilatura content extraction)."""

import asyncio

import pytest

from victor.tools import web_search_tool as wst
from victor.tools.web_search_tool import (
    _DDG_HTML_URL,
    _DDG_LITE_URL,
    _extract_content,
    _parse_ddg_lite_results,
    _pick_user_agent,
    _USER_AGENTS,
)


class TestUserAgentRotation:
    """Tests for the rotating User-Agent pool."""

    def test_pick_returns_a_pool_member(self):
        assert _pick_user_agent() in _USER_AGENTS

    def test_backwards_compat_alias_is_first_entry(self):
        # Static imports of _USER_AGENT must keep working.
        assert wst._USER_AGENT == _USER_AGENTS[0]


class TestParseDdgLiteResults:
    """Tests for the lite.duckduckgo.com fallback parser."""

    def test_parses_table_rows(self):
        html = """
        <html><body><table>
          <tr class="result-link"><td><a class="result-link" href="https://a.example">First</a></td></tr>
          <tr class="result-snippet"><td>Snippet one</td></tr>
          <tr class="result-link"><td><a class="result-link" href="https://b.example">Second</a></td></tr>
          <tr class="result-snippet"><td>Snippet two</td></tr>
        </table></body></html>
        """
        results = _parse_ddg_lite_results(html, max_results=5)
        assert len(results) == 2
        assert results[0]["title"] == "First"
        assert results[0]["url"] == "https://a.example"
        assert results[0]["snippet"] == "Snippet one"
        assert results[1]["title"] == "Second"

    def test_empty_html_returns_empty(self):
        assert _parse_ddg_lite_results("<html></html>", max_results=5) == []

    def test_falls_back_to_plain_anchor(self):
        # Row without the .result-link anchor class still yields via <a>.
        html = '<tr class="result-link"><td><a href="https://c.example">Plain</a></td></tr>'
        results = _parse_ddg_lite_results(html, max_results=5)
        assert len(results) == 1
        assert results[0]["url"] == "https://c.example"


class TestExtractContentTrafilaturaFallback:
    """Ensure _extract_content degrades to BeautifulSoup when trafilatura is absent."""

    def test_bs4_path_still_works(self):
        # A simple article; trafilatura may or may not be installed, but the BS4
        # fallback must always return content for a well-formed article.
        html = (
            """
        <html><body>
          <article>
            <p>This is the main article body content. """
            + "x" * 200
            + """</p>
          </article>
        </body></html>
        """
        )
        text = _extract_content(html, max_length=5000)
        assert "main article body content" in text

    def test_returns_empty_for_empty_html(self):
        assert _extract_content("<html><body></body></html>", max_length=5000) == ""


class TestDdgSearchFallback:
    """Tests for _ddg_search endpoint ordering and browser fallback."""

    @pytest.mark.asyncio
    async def test_returns_primary_results_when_html_succeeds(self, monkeypatch):
        """When the primary html endpoint yields results, lite/browser are skipped."""
        call_log = []

        async def fake_request_text(method, url, **kwargs):
            call_log.append(url)
            if url == _DDG_HTML_URL:
                return (
                    200,
                    '<div class="result"><a class="result__a" href="https://x.example">Hit</a></div>',
                )
            return 200, ""

        monkeypatch.setattr(wst, "_request_text", fake_request_text)

        async def no_render(*a, **k):
            call_log.append("BROWSER")
            return []

        monkeypatch.setattr(wst, "_ddg_render_results", no_render)

        results = await wst._ddg_search("query", "wt-wt", 5, "-1", web_config={})
        assert len(results) == 1
        assert results[0]["title"] == "Hit"
        # Lite endpoint and browser must NOT be reached.
        assert _DDG_LITE_URL not in call_log
        assert "BROWSER" not in call_log

    @pytest.mark.asyncio
    async def test_falls_through_to_lite_when_html_empty(self, monkeypatch):
        """Empty primary results -> lite endpoint is tried."""
        call_log = []

        async def fake_request_text(method, url, **kwargs):
            call_log.append(url)
            if url == _DDG_LITE_URL:
                return (
                    200,
                    '<tr class="result-link"><td><a class="result-link" '
                    'href="https://y.example">LiteHit</a></td></tr>',
                )
            return 200, ""  # empty primary

        monkeypatch.setattr(wst, "_request_text", fake_request_text)
        monkeypatch.setattr(
            wst, "_ddg_render_results", _make_async_return([])  # browser not reached
        )

        results = await wst._ddg_search("query", "wt-wt", 5, "-1", web_config={})
        assert len(results) == 1
        assert results[0]["title"] == "LiteHit"
        assert _DDG_HTML_URL in call_log and _DDG_LITE_URL in call_log

    @pytest.mark.asyncio
    async def test_browser_fallback_when_http_endpoints_empty(self, monkeypatch):
        """Both HTTP endpoints empty -> browser fallback returns results."""
        call_log = []

        async def fake_request_text(method, url, **kwargs):
            call_log.append(url)
            return 200, ""  # both empty

        monkeypatch.setattr(wst, "_request_text", fake_request_text)

        async def fake_render(query, region, max_results, **k):
            call_log.append("BROWSER")
            return [{"title": "Rendered", "url": "https://z.example", "snippet": ""}]

        monkeypatch.setattr(wst, "_ddg_render_results", fake_render)

        results = await wst._ddg_search("query", "wt-wt", 5, "-1", web_config={})
        assert len(results) == 1
        assert results[0]["title"] == "Rendered"
        assert "BROWSER" in call_log

    @pytest.mark.asyncio
    async def test_all_empty_returns_empty(self, monkeypatch):
        async def fake_request_text(method, url, **kwargs):
            return 200, ""

        monkeypatch.setattr(wst, "_request_text", fake_request_text)
        monkeypatch.setattr(wst, "_ddg_render_results", _make_async_return([]))
        results = await wst._ddg_search("query", "wt-wt", 5, "-1", web_config={})
        assert results == []

    @pytest.mark.asyncio
    async def test_http_exception_does_not_abort_search(self, monkeypatch):
        """An exception on the primary endpoint must not abort the whole search."""

        async def fake_request_text(method, url, **kwargs):
            if url == _DDG_HTML_URL:
                raise RuntimeError("boom")
            return 200, (
                '<tr class="result-link"><td><a class="result-link" '
                'href="https://recovered.example">Recovered</a></td></tr>'
            )

        monkeypatch.setattr(wst, "_request_text", fake_request_text)
        monkeypatch.setattr(wst, "_ddg_render_results", _make_async_return([]))

        results = await wst._ddg_search("query", "wt-wt", 5, "-1", web_config={})
        assert len(results) == 1
        assert results[0]["title"] == "Recovered"


def _make_async_return(value):
    async def _ret(*a, **k):
        return value

    return _ret
