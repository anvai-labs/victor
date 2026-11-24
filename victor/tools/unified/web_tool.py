import argparse
import sys
from io import StringIO

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified.parser import split_command


async def fetch_url(url: str):
    """Fetch a URL through the production web_fetch tool."""
    from victor.tools.web_search_tool import web_fetch

    result = await web_fetch(url=url, render="auto")
    if isinstance(result, dict):
        if result.get("success") is False:
            raise RuntimeError(result.get("error", "fetch failed"))
        return result.get("content") or result.get("results") or result.get("text") or str(result)
    return result


async def search_web(query: str):
    """Search the web through the production web_search tool."""
    from victor.tools.web_search_tool import web_search

    result = await web_search(query=query)
    if isinstance(result, dict):
        if result.get("success") is False:
            raise RuntimeError(result.get("error", "web search failed"))
        return result.get("results") or str(result)
    return result


async def render_page(url: str):
    """Render a page through the production web_fetch browser path."""
    from victor.tools.web_search_tool import web_fetch

    result = await web_fetch(url=url, render="browser")
    if isinstance(result, dict):
        if result.get("success") is False:
            raise RuntimeError(result.get("error", "render failed"))
        return result.get("content") or result.get("results") or result.get("text") or str(result)
    return result


async def browser_page(url: str):
    """Read a page through the production browser tool (FOU-A5 phase 2)."""
    from victor.tools.browser_tool import browser_tool

    result = await browser_tool(url=url)
    if isinstance(result, dict):
        if result.get("success") is False:
            raise RuntimeError(result.get("error", "browser read failed"))
        return {
            "title": result.get("title", ""),
            "url": result.get("url", url),
            "content": result.get("content", ""),
            "links": result.get("links", []),
        }
    return result


class UnifiedWebParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_web_parser() -> UnifiedWebParser:
    parser = UnifiedWebParser(
        prog="web", description="Unified web operations.", exit_on_error=False
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    # `fetch` subcommand
    fetch_parser = subparsers.add_parser("fetch", help="Fetch a URL as markdown")
    fetch_parser.add_argument("url", help="The URL to fetch")

    # `search` subcommand
    search_parser = subparsers.add_parser("search", help="Search the web")
    search_parser.add_argument("query", help="The search query")

    # `render` subcommand
    render_parser = subparsers.add_parser("render", help="Render a page with a browser")
    render_parser.add_argument("url", help="The URL to render")

    # `browser` subcommand (FOU-A5 phase 2): full browser read (title/text/links)
    browser_parser = subparsers.add_parser(
        "browser", help="Read a JS-required page in a headless browser"
    )
    browser_parser.add_argument("url", help="The URL to read in the browser")

    return parser


@tool(
    name="web",
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.HIGH,
)
async def web_tool(cmd: str) -> str:
    """Unified web tool.
    Example commands:
      web fetch "https://google.com"
      web search "python async tutorials"
      web render "https://example.com"
      web browser "https://example.com/spa"
    """
    parser = create_web_parser()

    try:
        args_list = split_command(cmd)
        if args_list and args_list[0] == "web":
            args_list = args_list[1:]
        parsed_args = parser.parse_args(args_list)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error: {e}"

    if parsed_args.subcommand == "fetch":
        try:
            return str(await fetch_url(parsed_args.url))
        except Exception as e:
            return f"### ❌ ERROR\nFetch failed: {e}"

    elif parsed_args.subcommand == "search":
        try:
            results = await search_web(parsed_args.query)
            if not isinstance(results, list):
                return str(results)

            out = []
            for r in results:
                title = r.get("title", "No Title")
                url = r.get("url", "#")
                snippet = r.get("snippet", "")
                out.append(f"### [{title}]({url})\n{snippet}\n")

            if not out:
                return "No results found."
            return "\n".join(out)
        except Exception as e:
            return f"### ❌ ERROR\nWeb search failed: {e}"

    elif parsed_args.subcommand == "render":
        try:
            return str(await render_page(parsed_args.url))
        except Exception as e:
            return f"### ❌ ERROR\nRender failed: {e}"

    elif parsed_args.subcommand == "browser":
        try:
            result = await browser_page(parsed_args.url)
            if not isinstance(result, dict):
                return str(result)
            lines = [f"### {result.get('title') or result.get('url', '')}", ""]
            lines.append(result.get("content") or "(no text extracted)")
            links = result.get("links") or []
            if links:
                lines.append("")
                lines.append("Links:")
                for link in links[:20]:
                    lines.append(
                        f"- [{link.get('text') or link.get('href', '')}]({link.get('href', '')})"
                    )
            return "\n".join(lines)
        except Exception as e:
            return f"### ❌ ERROR\nBrowser read failed: {e}"
    else:
        old_stdout = sys.stdout
        sys.stdout = capture = StringIO()
        parser.print_help()
        sys.stdout = old_stdout
        return f"### ❌ ERROR\nInvalid subcommand '{parsed_args.subcommand}'.\n\n```text\n{capture.getvalue()}```"
