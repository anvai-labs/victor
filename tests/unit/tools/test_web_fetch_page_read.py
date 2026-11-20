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

"""Tests for the readability-oriented page-read path of ``web_fetch`` (FOU-A5).

Covers: happy-path HTML extraction with title, download size cap,
timeout, and the non-HTML content-type guard. All HTTP is mocked with
respx — no real network. The page-read path needs no API key.
"""

import httpx
import pytest
import respx

from victor.tools import web_search_tool as wst
from victor.tools.web_search_tool import web_fetch

PAGE_URL = "https://example.com/article"


@pytest.fixture(autouse=True)
def _clear_web_caches():
    from victor.tools import web_search_tool as wst

    wst._GENERIC_WEB_CACHE = None
    wst._RATE_LIMITER = None
    yield
    wst._GENERIC_WEB_CACHE = None
    wst._RATE_LIMITER = None


def _article_html(title: str = "Victor Research Guide") -> str:
    body = " ".join("Victor research tools combine search adapters with readable page extraction.")
    return (
        "<html><head><title>"
        f"{title}"
        "</title></head><body>"
        "<nav>Home About Login</nav>"
        f"<article><h1>Research</h1><p>{body * 6}</p></article>"
        "<footer>copyright</footer>"
        "</body></html>"
    )


class TestHappyPath:
    @respx.mock
    async def test_html_page_returns_title_and_main_text(self):
        respx.get(PAGE_URL).mock(
            return_value=httpx.Response(
                200,
                html=_article_html(),
                headers={"Content-Type": "text/html; charset=utf-8"},
            )
        )

        result = await web_fetch(url=PAGE_URL)

        assert result["success"] is True
        assert result["title"] == "Victor Research Guide"
        # Extractors (trafilatura) may alter intra-word spacing on synthetic
        # HTML; compare letters-only so the assertion tests extraction, not
        # whitespace fidelity.
        assert "Victorresearchtools" in result["content"].replace(" ", "")
        assert len(result["content"]) > 200  # main text, not just the nav
        assert result["render"] == "http"

    @respx.mock
    async def test_missing_title_yields_empty_title_without_crashing(self):
        respx.get(PAGE_URL).mock(
            return_value=httpx.Response(
                200,
                html=_article_html().replace("<title>Victor Research Guide</title>", ""),
                headers={"Content-Type": "text/html"},
            )
        )

        result = await web_fetch(url=PAGE_URL)

        assert result["success"] is True
        assert result["title"] == ""

    @respx.mock
    async def test_content_type_absent_still_reads_page(self):
        respx.get(PAGE_URL).mock(return_value=httpx.Response(200, html=_article_html(), headers={}))

        result = await web_fetch(url=PAGE_URL)

        assert result["success"] is True


class TestContentTypeGuard:
    @respx.mock
    async def test_pdf_content_type_rejected(self):
        respx.get(PAGE_URL).mock(
            return_value=httpx.Response(
                200,
                content=b"%PDF-1.4 fake binary",
                headers={"Content-Type": "application/pdf"},
            )
        )

        result = await web_fetch(url=PAGE_URL)

        assert result["success"] is False
        assert result["content_type"] == "application/pdf"
        assert "content type" in result["error"].lower()

    @respx.mock
    async def test_binary_octet_stream_rejected(self):
        respx.get(PAGE_URL).mock(
            return_value=httpx.Response(
                200,
                content=b"\x00\x01binary",
                headers={"Content-Type": "application/octet-stream"},
            )
        )

        result = await web_fetch(url=PAGE_URL)

        assert result["success"] is False
        assert "content type" in result["error"].lower()


class TestSizeCap:
    @respx.mock
    async def test_download_over_cap_returns_structured_error(self, monkeypatch):
        monkeypatch.setattr(wst, "_PAGE_DOWNLOAD_CAP_BYTES", 512)
        respx.get(PAGE_URL).mock(
            return_value=httpx.Response(
                200,
                html=_article_html() * 10,
                headers={"Content-Type": "text/html"},
            )
        )

        result = await web_fetch(url=PAGE_URL)

        assert result["success"] is False
        assert result.get("download_cap_exceeded") is True
        assert "cap" in result["error"].lower()

    @respx.mock
    async def test_download_under_cap_succeeds(self, monkeypatch):
        monkeypatch.setattr(wst, "_PAGE_DOWNLOAD_CAP_BYTES", 512 * 1024)
        respx.get(PAGE_URL).mock(
            return_value=httpx.Response(
                200,
                html=_article_html(),
                headers={"Content-Type": "text/html"},
            )
        )

        result = await web_fetch(url=PAGE_URL)

        assert result["success"] is True


class TestTimeout:
    @respx.mock
    async def test_timeout_returns_structured_error(self):
        respx.get(PAGE_URL).mock(side_effect=httpx.ConnectTimeout("took too long"))

        result = await web_fetch(url=PAGE_URL)

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @respx.mock
    async def test_http_error_status_returns_structured_error(self):
        respx.get(PAGE_URL).mock(return_value=httpx.Response(503, text="unavailable"))

        result = await web_fetch(url=PAGE_URL)

        assert result["success"] is False
        assert "503" in result["error"]


class TestRegistration:
    async def test_web_fetch_needs_no_key_and_is_discoverable(self):
        from victor.agent.shared_tool_registry import SharedToolRegistry

        registry = SharedToolRegistry.get_instance()
        try:
            tools = registry.get_tool_classes()
            assert "web_fetch" in tools
        finally:
            SharedToolRegistry.reset_instance()
