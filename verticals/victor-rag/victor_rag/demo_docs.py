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

"""Project Documentation RAG Demo - Ingest and query project docs.

This demo shows how to use the RAG vertical to ingest project
documentation and answer questions about the codebase.

Usage:
    python -m victor.rag.demo_docs [--path /path/to/docs] [--pattern "*.md"]

Example:
    # Ingest Victor's documentation
    python -m victor.rag.demo_docs

    # Ingest custom docs
    python -m victor.rag.demo_docs --path ./my-project/docs --pattern "*.rst"

    # Query the documentation
    python -m victor.rag.demo_docs --query "How do I add a new provider?"
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default documentation locations
DEFAULT_DOC_PATTERNS = [
    ("*.md", "Markdown files"),
    ("docs/**/*.md", "Documentation folder"),
    ("*.rst", "ReStructuredText files"),
    ("README*", "README files"),
    ("CHANGELOG*", "Changelog files"),
    ("CONTRIBUTING*", "Contributing guides"),
]


async def ingest_project_docs(
    project_path: Optional[Path] = None,
    patterns: Optional[List[str]] = None,
    recursive: bool = True,
) -> Dict[str, int]:
    """Ingest project documentation into RAG store.

    Args:
        project_path: Path to project root (default: current directory)
        patterns: Glob patterns to match (default: common doc patterns)
        recursive: Whether to search recursively

    Returns:
        Dict mapping pattern to chunk counts
    """
    from victor_rag.document_store import Document, DocumentStore

    if project_path is None:
        project_path = Path.cwd()
    else:
        project_path = Path(project_path)

    if patterns is None:
        patterns = ["*.md", "*.rst", "*.txt"]

    store = DocumentStore()
    await store.initialize()

    results: Dict[str, int] = {}
    processed_files: set = set()

    for pattern in patterns:
        logger.info(f"Searching for {pattern}...")

        if recursive:
            files = list(project_path.rglob(pattern))
        else:
            files = list(project_path.glob(pattern))

        # Filter files
        files = [
            f
            for f in files
            if f.is_file()
            and str(f) not in processed_files
            and not any(
                skip in str(f)
                for skip in [
                    "__pycache__",
                    ".git",
                    "node_modules",
                    ".venv",
                    "venv",
                    ".tox",
                    ".pytest_cache",
                    "dist",
                    "build",
                    ".egg-info",
                    ".victor/",
                    "archive/",
                ]
            )
        ]

        if not files:
            logger.info(f"  No files found matching {pattern}")
            continue

        pattern_chunks = 0

        for file_path in files:
            try:
                # Read content
                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    logger.warning(f"  Skipping binary file: {file_path}")
                    continue

                if not content.strip():
                    continue

                # Determine doc type
                suffix = file_path.suffix.lower()
                if suffix in {".md", ".markdown"}:
                    doc_type = "markdown"
                elif suffix in {".rst"}:
                    doc_type = "markdown"  # Close enough for chunking
                elif suffix in {".py", ".js", ".ts"}:
                    doc_type = "code"
                else:
                    doc_type = "text"

                # Create document
                rel_path = file_path.relative_to(project_path)
                doc_id = f"doc_{str(rel_path).replace('/', '_').replace('.', '_')}"

                doc = Document(
                    id=doc_id,
                    content=content,
                    source=str(file_path),
                    doc_type=doc_type,
                    metadata={
                        "project": project_path.name,
                        "relative_path": str(rel_path),
                        "file_type": suffix,
                    },
                )

                # Ingest
                chunks = await store.add_document(doc)
                pattern_chunks += len(chunks)
                processed_files.add(str(file_path))

                logger.info(f"  Ingested {rel_path} ({len(chunks)} chunks)")

            except Exception as e:
                logger.error(f"  Failed to process {file_path}: {e}")

        results[pattern] = pattern_chunks

    return results


async def query_docs(query: str, top_k: int = 5) -> None:
    """Query ingested documentation.

    Args:
        query: Search query
        top_k: Number of results to return
    """
    from victor_rag.document_store import DocumentStore

    store = DocumentStore()
    await store.initialize()

    logger.info(f"\nSearching for: {query}\n")

    results = await store.search(query, k=top_k)

    if not results:
        logger.info("No results found. Have you ingested any documentation?")
        return

    print("=" * 80)
    print(f"Found {len(results)} relevant chunks:\n")

    for i, result in enumerate(results, 1):
        metadata = result.metadata
        rel_path = metadata.get("relative_path", "Unknown")
        project = metadata.get("project", "N/A")

        print(f"[{i}] {rel_path}")
        print(f"    Project: {project}")
        print(f"    Score: {result.score:.4f}")
        print("    Content preview:")
        preview = result.content[:400].replace("\n", " ")
        print(f"    {preview}...")
        print()

    print("=" * 80)


async def show_stats() -> None:
    """Show RAG store statistics."""
    from victor_rag.document_store import DocumentStore

    store = DocumentStore()
    await store.initialize()

    stats = await store.get_stats()

    print("\n" + "=" * 50)
    print("RAG Store Statistics")
    print("=" * 50)
    print(f"Total documents: {stats.get('total_documents', 0)}")
    print(f"Total chunks: {stats.get('total_chunks', 0)}")
    print(f"Store location: {stats.get('store_path', 'N/A')}")

    docs = await store.list_documents()
    if docs:
        print("\nDocuments by project:")
        by_project: Dict[str, int] = {}
        for doc in docs:
            project = doc.metadata.get("project", "Other")
            by_project[project] = by_project.get(project, 0) + 1
        for project, count in sorted(by_project.items()):
            print(f"  {project}: {count} documents")

    print("=" * 50)


async def ingest_victor_docs() -> Dict[str, int]:
    """Ingest Victor project documentation."""
    # Find Victor project root
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists() and (current / "victor").exists():
            break
        current = current.parent

    if not (current / "pyproject.toml").exists():
        raise RuntimeError("Could not find Victor project root")

    logger.info(f"Found Victor project at: {current}")

    # Ingest documentation
    return await ingest_project_docs(
        project_path=current,
        patterns=[
            "*.md",
            "docs/**/*.md",
            "examples/**/*.md",
        ],
        recursive=True,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Project Documentation RAG Demo - Ingest and query project docs"
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="Path to project root (default: current directory for custom, Victor root for default)",
    )
    parser.add_argument(
        "--pattern",
        "-P",
        type=str,
        action="append",
        help="Glob pattern to match (can be specified multiple times)",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Query the ingested documentation instead of ingesting",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Show RAG store statistics",
    )
    parser.add_argument(
        "--victor",
        "-v",
        action="store_true",
        help="Ingest Victor's own documentation",
    )

    args = parser.parse_args()

    if args.stats:
        asyncio.run(show_stats())
        return

    if args.query:
        asyncio.run(query_docs(args.query))
        return

    print(f"\n{'=' * 60}")
    print("Project Documentation RAG Demo")
    print(f"{'=' * 60}")

    if args.victor or (not args.path and not args.pattern):
        # Ingest Victor docs
        print("Mode: Ingest Victor Documentation")
        results = asyncio.run(ingest_victor_docs())
    else:
        # Ingest custom docs
        project_path = Path(args.path) if args.path else Path.cwd()
        patterns = args.pattern or ["*.md"]
        print(f"Path: {project_path}")
        print(f"Patterns: {patterns}")
        results = asyncio.run(
            ingest_project_docs(
                project_path=project_path,
                patterns=patterns,
            )
        )

    print(f"\n{'=' * 60}")
    print("Ingestion Complete!")
    print(f"{'=' * 60}")

    total_chunks = 0
    for pattern, chunks in results.items():
        print(f"  {pattern}: {chunks} chunks")
        total_chunks += chunks

    print(f"\n  Total: {total_chunks} chunks")
    print(f"{'=' * 60}")
    print("\nYou can now query the documentation using victor CLI:")
    print('  victor rag query "How do I add a new provider?"')
    print('  victor rag query "What tools are available?" --synthesize')
    print("")
    print("  # View statistics and list documents:")
    print("  victor rag stats")
    print("  victor rag list")
    print()


if __name__ == "__main__":
    main()
