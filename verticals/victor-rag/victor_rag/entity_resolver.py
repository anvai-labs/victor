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

"""Entity Resolver - Dynamic entity resolution from document store metadata.

Resolves entities mentioned in queries against the document store metadata,
extracting associated information like tickers, company names, sectors, etc.

This replaces hardcoded dictionaries with dynamic database queries.

Example:
    resolver = EntityResolver(document_store)
    await resolver.initialize()

    # Resolve entities from a query
    entities = await resolver.resolve_entities("Compare Apple and Microsoft revenue")
    # Returns: [
    #   EntityInfo(name="Apple", ticker="AAPL", sector="Technology", ...),
    #   EntityInfo(name="Microsoft", ticker="MSFT", sector="Technology", ...),
    # ]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from victor_rag.document_store import DocumentStore

logger = logging.getLogger(__name__)


@dataclass
class EntityInfo:
    """Information about a resolved entity.

    Attributes:
        name: Canonical entity name (e.g., "Apple Inc")
        ticker: Stock ticker symbol (e.g., "AAPL")
        sector: Industry sector (e.g., "Technology")
        aliases: Alternative names/references
        metadata: Additional metadata from documents
        doc_ids: Document IDs containing this entity
        confidence: Resolution confidence (0.0-1.0)
    """

    name: str
    ticker: Optional[str] = None
    sector: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def get_search_terms(self) -> List[str]:
        """Get all search terms for this entity.

        Returns:
            List of terms to use in search queries
        """
        terms = [self.name]
        if self.ticker:
            terms.append(self.ticker)
        terms.extend(self.aliases)
        return list(set(terms))


@dataclass
class EntityIndex:
    """Index of entities extracted from document metadata.

    Provides fast lookup by name, ticker, or alias.
    """

    # Primary index: canonical name -> EntityInfo
    by_name: Dict[str, EntityInfo] = field(default_factory=dict)

    # Secondary indexes for fast lookup
    by_ticker: Dict[str, str] = field(default_factory=dict)  # ticker -> canonical name
    by_alias: Dict[str, str] = field(default_factory=dict)  # alias -> canonical name
    by_sector: Dict[str, List[str]] = field(default_factory=dict)  # sector -> [names]

    def add(self, entity: EntityInfo) -> None:
        """Add entity to the index."""
        canonical = entity.name.lower()
        self.by_name[canonical] = entity

        if entity.ticker:
            self.by_ticker[entity.ticker.upper()] = canonical
            self.by_ticker[entity.ticker.lower()] = canonical

        for alias in entity.aliases:
            self.by_alias[alias.lower()] = canonical

        if entity.sector:
            if entity.sector not in self.by_sector:
                self.by_sector[entity.sector] = []
            self.by_sector[entity.sector].append(canonical)

    def lookup(self, term: str) -> Optional[EntityInfo]:
        """Look up entity by any identifier.

        Args:
            term: Name, ticker, or alias to look up

        Returns:
            EntityInfo if found, None otherwise
        """
        term_lower = term.lower()
        term_upper = term.upper()

        # Try direct name match
        if term_lower in self.by_name:
            return self.by_name[term_lower]

        # Try ticker
        if term_upper in self.by_ticker:
            canonical = self.by_ticker[term_upper]
            return self.by_name.get(canonical)

        # Try alias
        if term_lower in self.by_alias:
            canonical = self.by_alias[term_lower]
            return self.by_name.get(canonical)

        return None

    def fuzzy_match(self, term: str, threshold: float = 0.8) -> List[EntityInfo]:
        """Find entities with fuzzy matching.

        Args:
            term: Term to match
            threshold: Minimum similarity threshold

        Returns:
            List of matching entities sorted by confidence
        """
        matches = []
        term_lower = term.lower()

        for canonical, entity in self.by_name.items():
            # Check name similarity
            similarity = self._string_similarity(term_lower, canonical)
            if similarity >= threshold:
                entity.confidence = similarity
                matches.append(entity)
                continue

            # Check aliases
            for alias in entity.aliases:
                similarity = self._string_similarity(term_lower, alias.lower())
                if similarity >= threshold:
                    entity.confidence = similarity
                    matches.append(entity)
                    break

        return sorted(matches, key=lambda e: e.confidence, reverse=True)

    def _string_similarity(self, a: str, b: str) -> float:
        """Calculate string similarity using character overlap."""
        if not a or not b:
            return 0.0

        # Simple Jaccard similarity on character bigrams
        def bigrams(s: str) -> Set[str]:
            return set(s[i : i + 2] for i in range(len(s) - 1))

        a_bigrams = bigrams(a)
        b_bigrams = bigrams(b)

        if not a_bigrams or not b_bigrams:
            return 1.0 if a == b else 0.0

        intersection = len(a_bigrams & b_bigrams)
        union = len(a_bigrams | b_bigrams)

        return intersection / union if union > 0 else 0.0


class EntityResolver:
    """Resolves entities from queries using document store metadata.

    Builds an index from document metadata and provides fast entity resolution.
    Supports exact matching, ticker lookup, and fuzzy matching.
    """

    def __init__(self, document_store: Optional["DocumentStore"] = None):
        """Initialize entity resolver.

        Args:
            document_store: Document store to query for metadata
        """
        self._store = document_store
        self._index = EntityIndex()
        self._initialized = False

        # Common financial terms for context detection
        self._financial_terms = {
            "revenue",
            "sales",
            "income",
            "profit",
            "earnings",
            "growth",
            "margin",
            "eps",
            "debt",
            "assets",
            "cash",
            "expenses",
            "costs",
            "liabilities",
            "equity",
        }

        # Comparison keywords
        self._comparison_keywords = {
            "compare",
            "versus",
            "vs",
            "difference",
            "between",
            "against",
            "relative",
            "compared",
        }

    async def initialize(self) -> None:
        """Initialize the entity index from document store metadata."""
        if self._initialized:
            return

        if not self._store:
            logger.warning("No document store provided, using empty index")
            self._initialized = True
            return

        try:
            # Get all documents from store
            docs = await self._store.list_documents()

            # Group by entity (e.g., company)
            entities_seen: Dict[str, EntityInfo] = {}

            for doc in docs:
                metadata = doc.metadata or {}

                # Extract entity information from document metadata
                ticker = metadata.get("symbol") or metadata.get("ticker")
                company = metadata.get("company") or metadata.get("name")
                sector = metadata.get("sector")

                if not ticker and not company:
                    continue

                # Use ticker as canonical identifier if available
                canonical_key = (ticker or company or "").lower()

                if canonical_key in entities_seen:
                    # Update existing entity
                    entity = entities_seen[canonical_key]
                    entity.doc_ids.append(doc.id)

                    # Merge metadata
                    for key, value in metadata.items():
                        if key not in entity.metadata and value:
                            entity.metadata[key] = value
                else:
                    # Create new entity
                    aliases = []

                    # Add company name variations as aliases
                    if company:
                        aliases.append(company)
                        # Add short name (e.g., "Apple" from "Apple Inc")
                        short_name = self._extract_short_name(company)
                        if short_name and short_name != company:
                            aliases.append(short_name)

                    entity = EntityInfo(
                        name=company or ticker or "Unknown",
                        ticker=ticker,
                        sector=sector,
                        aliases=aliases,
                        metadata=dict(metadata),
                        doc_ids=[doc.id],
                    )
                    entities_seen[canonical_key] = entity

            # Build index
            for entity in entities_seen.values():
                self._index.add(entity)

            logger.info(f"Entity index built with {len(entities_seen)} entities")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize entity index: {e}")
            self._initialized = True  # Mark as initialized to avoid repeated failures

    def _extract_short_name(self, full_name: str) -> str:
        """Extract short company name from full name.

        Args:
            full_name: Full company name (e.g., "Apple Inc")

        Returns:
            Short name (e.g., "Apple")
        """
        # Remove common suffixes
        suffixes = [
            " Inc",
            " Inc.",
            " Corporation",
            " Corp",
            " Corp.",
            " Company",
            " Co",
            " Co.",
            " LLC",
            " Ltd",
            " Ltd.",
            " PLC",
            " plc",
            " NV",
            " SA",
            " AG",
            " SE",
            " Holdings",
            " Group",
            " International",
        ]

        name = full_name
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]

        return name.strip()

    async def resolve_entities(
        self,
        query: str,
        use_fuzzy: bool = True,
        fuzzy_threshold: float = 0.7,
    ) -> List[EntityInfo]:
        """Resolve entities mentioned in a query.

        Args:
            query: User query to analyze
            use_fuzzy: Whether to use fuzzy matching for unresolved terms
            fuzzy_threshold: Minimum similarity for fuzzy matches

        Returns:
            List of resolved entities
        """
        await self.initialize()

        resolved: List[EntityInfo] = []
        resolved_names: Set[str] = set()

        # Tokenize query into potential entity mentions
        tokens = self._extract_potential_entities(query)

        for token in tokens:
            # Skip if already resolved
            if token.lower() in resolved_names:
                continue

            # Try exact/alias lookup first
            entity = self._index.lookup(token)
            if entity:
                resolved.append(entity)
                resolved_names.add(entity.name.lower())
                if entity.ticker:
                    resolved_names.add(entity.ticker.lower())
                continue

            # Try fuzzy matching
            if use_fuzzy:
                fuzzy_matches = self._index.fuzzy_match(token, fuzzy_threshold)
                for entity in fuzzy_matches:
                    if entity.name.lower() not in resolved_names:
                        resolved.append(entity)
                        resolved_names.add(entity.name.lower())
                        break  # Take best match only

        return resolved

    def _extract_potential_entities(self, query: str) -> List[str]:
        """Extract potential entity mentions from query.

        Args:
            query: User query

        Returns:
            List of potential entity tokens
        """
        # Remove punctuation but keep apostrophes in words
        clean_query = re.sub(r"[^\w\s'-]", " ", query)

        # Split into words
        words = clean_query.split()

        # Filter out common words and keep potential entities
        # (capitalized words, all-caps words, known patterns)
        potential = []

        for i, word in enumerate(words):
            # Strip possessive forms: "Tesla's" → "Tesla", "Apple's" → "Apple"
            clean_word = re.sub(r"'s$", "", word)
            word_lower = clean_word.lower()

            # Skip comparison keywords and financial terms
            if word_lower in self._comparison_keywords:
                continue
            if word_lower in self._financial_terms:
                continue

            # Skip very short words unless they look like tickers
            if len(clean_word) < 3 and not clean_word.isupper():
                continue

            # Skip common words
            if word_lower in {
                "the",
                "and",
                "for",
                "with",
                "what",
                "how",
                "why",
                "is",
                "are",
                "was",
                "were",
            }:
                continue

            # Include if:
            # - Capitalized (proper noun)
            # - All uppercase (potential ticker)
            # - In our index
            if clean_word[0].isupper() or clean_word.isupper() or self._index.lookup(clean_word):
                potential.append(clean_word)

        return potential

    def analyze_query(self, query: str, entities: List[EntityInfo]) -> Dict[str, Any]:
        """Analyze query with resolved entities.

        Args:
            query: Original query
            entities: Resolved entities

        Returns:
            Analysis dict with query metadata
        """
        query_lower = query.lower()

        # Detect comparison
        is_comparison = any(kw in query_lower for kw in self._comparison_keywords)

        # If multiple entities mentioned, treat as comparison even without keyword
        if len(entities) > 1:
            is_comparison = True

        # Detect financial terms
        financial_terms = [t for t in self._financial_terms if t in query_lower]

        # Calculate recommended k
        if is_comparison:
            recommended_k = max(10, len(entities) * 5)
        elif entities:
            recommended_k = 5
        else:
            recommended_k = 5

        # Build expanded query from entity search terms
        expansion_terms = []
        for entity in entities:
            expansion_terms.extend(entity.get_search_terms())

        return {
            "entities": entities,
            "entity_names": [e.name for e in entities],
            "is_comparison": is_comparison,
            "financial_terms": financial_terms,
            "recommended_k": recommended_k,
            "expansion_terms": list(set(expansion_terms)),
            "entity_count": len(entities),
        }

    def get_entity_count(self) -> int:
        """Get total number of indexed entities."""
        return len(self._index.by_name)

    def get_sectors(self) -> List[str]:
        """Get list of all sectors in the index."""
        return list(self._index.by_sector.keys())

    def get_entities_by_sector(self, sector: str) -> List[EntityInfo]:
        """Get all entities in a sector.

        Args:
            sector: Sector name

        Returns:
            List of entities in that sector
        """
        names = self._index.by_sector.get(sector, [])
        return [self._index.by_name[name] for name in names if name in self._index.by_name]

    def refresh(self) -> None:
        """Mark index as stale to force rebuild on next access."""
        self._initialized = False
        self._index = EntityIndex()
