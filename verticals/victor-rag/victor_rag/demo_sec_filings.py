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

"""SEC Filing RAG Demo - Ingest 10-K/10-Q filings for S&P 500 stocks.

This demo shows how to use the RAG vertical to ingest and query
SEC filings for S&P 500 companies.

Usage:
    victor rag demo-sec [OPTIONS]               # Via CLI
    python -m victor.rag.demo_sec_filings       # Direct

Examples:
    # Ingest latest 10-K filings for top 50 S&P 500 companies
    victor rag demo-sec --preset top50

    # Ingest only Apple 10-Q filings
    victor rag demo-sec --company AAPL --filing-type 10-Q

    # Clear all SEC filings from RAG store
    victor rag demo-sec --clear

    # Query the ingested filings using CLI
    victor rag query "What is Apple's revenue?"
    victor rag query "Compare NVIDIA and AMD revenue growth" --synthesize

    # Search without synthesis
    victor rag search "risk factors technology sector"

    # List all ingested documents
    victor rag list

    # Show RAG store stats
    victor rag stats
"""

import argparse
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Top 100 S&P 500 companies by market cap (as of 2025) with SEC CIK numbers
# CIK numbers are padded to 10 digits with leading zeros
SP500_COMPANIES: Dict[str, Dict[str, str]] = {
    # === Magnificent 7 / Big Tech ===
    "AAPL": {"name": "Apple Inc", "cik": "0000320193", "sector": "Technology"},
    "MSFT": {
        "name": "Microsoft Corporation",
        "cik": "0000789019",
        "sector": "Technology",
    },
    "GOOGL": {
        "name": "Alphabet Inc (Google)",
        "cik": "0001652044",
        "sector": "Technology",
    },
    "AMZN": {
        "name": "Amazon.com Inc",
        "cik": "0001018724",
        "sector": "Consumer Discretionary",
    },
    "NVDA": {"name": "NVIDIA Corporation", "cik": "0001045810", "sector": "Technology"},
    "META": {"name": "Meta Platforms Inc", "cik": "0001326801", "sector": "Technology"},
    "TSLA": {
        "name": "Tesla Inc",
        "cik": "0001318605",
        "sector": "Consumer Discretionary",
    },
    # === Financial Services ===
    "BRK.B": {
        "name": "Berkshire Hathaway Inc",
        "cik": "0001067983",
        "sector": "Financials",
    },
    "JPM": {"name": "JPMorgan Chase & Co", "cik": "0000019617", "sector": "Financials"},
    "V": {"name": "Visa Inc", "cik": "0001403161", "sector": "Financials"},
    "MA": {"name": "Mastercard Inc", "cik": "0001141391", "sector": "Financials"},
    "BAC": {
        "name": "Bank of America Corp",
        "cik": "0000070858",
        "sector": "Financials",
    },
    "WFC": {"name": "Wells Fargo & Co", "cik": "0000072971", "sector": "Financials"},
    "GS": {
        "name": "Goldman Sachs Group Inc",
        "cik": "0000886982",
        "sector": "Financials",
    },
    "MS": {"name": "Morgan Stanley", "cik": "0000895421", "sector": "Financials"},
    "AXP": {"name": "American Express Co", "cik": "0000004962", "sector": "Financials"},
    "C": {"name": "Citigroup Inc", "cik": "0000831001", "sector": "Financials"},
    "BLK": {"name": "BlackRock Inc", "cik": "0001364742", "sector": "Financials"},
    "SCHW": {
        "name": "Charles Schwab Corp",
        "cik": "0000316709",
        "sector": "Financials",
    },
    "SPGI": {"name": "S&P Global Inc", "cik": "0000064040", "sector": "Financials"},
    # === Healthcare ===
    "UNH": {
        "name": "UnitedHealth Group Inc",
        "cik": "0000731766",
        "sector": "Healthcare",
    },
    "JNJ": {"name": "Johnson & Johnson", "cik": "0000200406", "sector": "Healthcare"},
    "LLY": {"name": "Eli Lilly and Co", "cik": "0000059478", "sector": "Healthcare"},
    "PFE": {"name": "Pfizer Inc", "cik": "0000078003", "sector": "Healthcare"},
    "ABBV": {"name": "AbbVie Inc", "cik": "0001551152", "sector": "Healthcare"},
    "MRK": {"name": "Merck & Co Inc", "cik": "0000310158", "sector": "Healthcare"},
    "TMO": {
        "name": "Thermo Fisher Scientific",
        "cik": "0000097745",
        "sector": "Healthcare",
    },
    "ABT": {"name": "Abbott Laboratories", "cik": "0000001800", "sector": "Healthcare"},
    "DHR": {"name": "Danaher Corporation", "cik": "0000313616", "sector": "Healthcare"},
    "BMY": {
        "name": "Bristol-Myers Squibb Co",
        "cik": "0000014272",
        "sector": "Healthcare",
    },
    "AMGN": {"name": "Amgen Inc", "cik": "0000318154", "sector": "Healthcare"},
    "GILD": {
        "name": "Gilead Sciences Inc",
        "cik": "0000882095",
        "sector": "Healthcare",
    },
    "MDT": {"name": "Medtronic PLC", "cik": "0001613103", "sector": "Healthcare"},
    "ISRG": {
        "name": "Intuitive Surgical Inc",
        "cik": "0001035267",
        "sector": "Healthcare",
    },
    "VRTX": {
        "name": "Vertex Pharmaceuticals",
        "cik": "0000875320",
        "sector": "Healthcare",
    },
    # === Consumer ===
    "PG": {
        "name": "Procter & Gamble Co",
        "cik": "0000080424",
        "sector": "Consumer Staples",
    },
    "KO": {"name": "Coca-Cola Co", "cik": "0000021344", "sector": "Consumer Staples"},
    "PEP": {"name": "PepsiCo Inc", "cik": "0000077476", "sector": "Consumer Staples"},
    "COST": {
        "name": "Costco Wholesale Corp",
        "cik": "0000909832",
        "sector": "Consumer Staples",
    },
    "WMT": {"name": "Walmart Inc", "cik": "0000104169", "sector": "Consumer Staples"},
    "MCD": {
        "name": "McDonald's Corp",
        "cik": "0000063908",
        "sector": "Consumer Discretionary",
    },
    "NKE": {
        "name": "Nike Inc",
        "cik": "0000320187",
        "sector": "Consumer Discretionary",
    },
    "SBUX": {
        "name": "Starbucks Corp",
        "cik": "0000829224",
        "sector": "Consumer Discretionary",
    },
    "HD": {
        "name": "Home Depot Inc",
        "cik": "0000354950",
        "sector": "Consumer Discretionary",
    },
    "LOW": {
        "name": "Lowe's Companies Inc",
        "cik": "0000060667",
        "sector": "Consumer Discretionary",
    },
    "TGT": {
        "name": "Target Corp",
        "cik": "0000027419",
        "sector": "Consumer Discretionary",
    },
    "NFLX": {
        "name": "Netflix Inc",
        "cik": "0001065280",
        "sector": "Consumer Discretionary",
    },
    "DIS": {
        "name": "Walt Disney Co",
        "cik": "0001744489",
        "sector": "Communication Services",
    },
    "CMCSA": {
        "name": "Comcast Corp",
        "cik": "0001166691",
        "sector": "Communication Services",
    },
    # === Industrials ===
    "CAT": {"name": "Caterpillar Inc", "cik": "0000018230", "sector": "Industrials"},
    "BA": {"name": "Boeing Co", "cik": "0000012927", "sector": "Industrials"},
    "HON": {
        "name": "Honeywell International",
        "cik": "0000773840",
        "sector": "Industrials",
    },
    "UPS": {
        "name": "United Parcel Service",
        "cik": "0001090727",
        "sector": "Industrials",
    },
    "RTX": {"name": "RTX Corporation", "cik": "0000101829", "sector": "Industrials"},
    "LMT": {
        "name": "Lockheed Martin Corp",
        "cik": "0000936468",
        "sector": "Industrials",
    },
    "GE": {"name": "General Electric Co", "cik": "0000040545", "sector": "Industrials"},
    "DE": {"name": "Deere & Company", "cik": "0000315189", "sector": "Industrials"},
    "MMM": {"name": "3M Company", "cik": "0000066740", "sector": "Industrials"},
    "UNP": {"name": "Union Pacific Corp", "cik": "0000100885", "sector": "Industrials"},
    # === Energy ===
    "XOM": {"name": "Exxon Mobil Corp", "cik": "0000034088", "sector": "Energy"},
    "CVX": {"name": "Chevron Corp", "cik": "0000093410", "sector": "Energy"},
    "COP": {"name": "ConocoPhillips", "cik": "0001163165", "sector": "Energy"},
    "SLB": {"name": "Schlumberger NV", "cik": "0000087347", "sector": "Energy"},
    "EOG": {"name": "EOG Resources Inc", "cik": "0000821189", "sector": "Energy"},
    # === Technology (continued) ===
    "AVGO": {"name": "Broadcom Inc", "cik": "0001730168", "sector": "Technology"},
    "ORCL": {"name": "Oracle Corporation", "cik": "0001341439", "sector": "Technology"},
    "CSCO": {"name": "Cisco Systems Inc", "cik": "0000858877", "sector": "Technology"},
    "ACN": {"name": "Accenture PLC", "cik": "0001467373", "sector": "Technology"},
    "CRM": {"name": "Salesforce Inc", "cik": "0001108524", "sector": "Technology"},
    "ADBE": {"name": "Adobe Inc", "cik": "0000796343", "sector": "Technology"},
    "AMD": {
        "name": "Advanced Micro Devices",
        "cik": "0000002488",
        "sector": "Technology",
    },
    "INTC": {"name": "Intel Corporation", "cik": "0000050863", "sector": "Technology"},
    "IBM": {
        "name": "International Business Machines",
        "cik": "0000051143",
        "sector": "Technology",
    },
    "QCOM": {"name": "Qualcomm Inc", "cik": "0000804328", "sector": "Technology"},
    "TXN": {
        "name": "Texas Instruments Inc",
        "cik": "0000097476",
        "sector": "Technology",
    },
    "INTU": {"name": "Intuit Inc", "cik": "0000896878", "sector": "Technology"},
    "NOW": {"name": "ServiceNow Inc", "cik": "0001373715", "sector": "Technology"},
    "AMAT": {
        "name": "Applied Materials Inc",
        "cik": "0000006951",
        "sector": "Technology",
    },
    "MU": {
        "name": "Micron Technology Inc",
        "cik": "0000723125",
        "sector": "Technology",
    },
    "LRCX": {"name": "Lam Research Corp", "cik": "0000707549", "sector": "Technology"},
    "ADI": {"name": "Analog Devices Inc", "cik": "0000006281", "sector": "Technology"},
    "SNPS": {"name": "Synopsys Inc", "cik": "0000883241", "sector": "Technology"},
    "KLAC": {"name": "KLA Corporation", "cik": "0000319201", "sector": "Technology"},
    "PANW": {
        "name": "Palo Alto Networks Inc",
        "cik": "0001327567",
        "sector": "Technology",
    },
    # === Utilities & Real Estate ===
    "NEE": {"name": "NextEra Energy Inc", "cik": "0000753308", "sector": "Utilities"},
    "DUK": {"name": "Duke Energy Corp", "cik": "0001326160", "sector": "Utilities"},
    "SO": {"name": "Southern Company", "cik": "0000092122", "sector": "Utilities"},
    "AMT": {
        "name": "American Tower Corp",
        "cik": "0001053507",
        "sector": "Real Estate",
    },
    "PLD": {"name": "Prologis Inc", "cik": "0001045609", "sector": "Real Estate"},
    "CCI": {"name": "Crown Castle Inc", "cik": "0001051470", "sector": "Real Estate"},
    # === Additional Top Companies ===
    "T": {"name": "AT&T Inc", "cik": "0000732717", "sector": "Communication Services"},
    "VZ": {
        "name": "Verizon Communications",
        "cik": "0000732712",
        "sector": "Communication Services",
    },
    "TMUS": {
        "name": "T-Mobile US Inc",
        "cik": "0001283699",
        "sector": "Communication Services",
    },
    "PM": {
        "name": "Philip Morris International",
        "cik": "0001413329",
        "sector": "Consumer Staples",
    },
    "MO": {
        "name": "Altria Group Inc",
        "cik": "0000764180",
        "sector": "Consumer Staples",
    },
    "CL": {
        "name": "Colgate-Palmolive Co",
        "cik": "0000021665",
        "sector": "Consumer Staples",
    },
    "EL": {
        "name": "Estee Lauder Companies",
        "cik": "0001001250",
        "sector": "Consumer Staples",
    },
    "MDLZ": {
        "name": "Mondelez International",
        "cik": "0001103982",
        "sector": "Consumer Staples",
    },
    "CB": {"name": "Chubb Limited", "cik": "0000896159", "sector": "Financials"},
    "PNC": {
        "name": "PNC Financial Services",
        "cik": "0000713676",
        "sector": "Financials",
    },
    "TFC": {
        "name": "Truist Financial Corp",
        "cik": "0000092230",
        "sector": "Financials",
    },
    "USB": {"name": "US Bancorp", "cik": "0000036104", "sector": "Financials"},
}

# Preset company groups
COMPANY_PRESETS: Dict[str, List[str]] = {
    "faang": ["META", "AMZN", "AAPL", "NFLX", "GOOGL"],
    "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "top10": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "BRK.B",
        "JPM",
        "V",
    ],
    "top25": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "BRK.B",
        "JPM",
        "V",
        "UNH",
        "MA",
        "JNJ",
        "XOM",
        "PG",
        "HD",
        "CVX",
        "LLY",
        "AVGO",
        "COST",
        "ABBV",
        "MRK",
        "PFE",
        "KO",
        "PEP",
    ],
    "top50": list(SP500_COMPANIES.keys())[:50],
    "top100": list(SP500_COMPANIES.keys()),
    "tech": [s for s, info in SP500_COMPANIES.items() if info["sector"] == "Technology"],
    "healthcare": [s for s, info in SP500_COMPANIES.items() if info["sector"] == "Healthcare"],
    "financials": [s for s, info in SP500_COMPANIES.items() if info["sector"] == "Financials"],
    "energy": [s for s, info in SP500_COMPANIES.items() if info["sector"] == "Energy"],
}


@dataclass
class Filing:
    """SEC Filing metadata."""

    company: str
    cik: str
    filing_type: str
    filing_date: str
    accession_number: str
    primary_doc_url: str


class SECFilingFetcher:
    """Fetch SEC filings from EDGAR."""

    BASE_URL = "https://www.sec.gov"
    EDGAR_API = "https://data.sec.gov"

    def __init__(self):
        self.headers = {
            "User-Agent": "VictorRAG/1.0 (research@example.com)",
            "Accept": "application/json",
        }
        # Create SSL context that doesn't verify certificates (for demo purposes)
        import ssl

        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE

    async def get_company_filings(
        self,
        cik: str,
        filing_type: str = "10-K",
        count: int = 1,
    ) -> List[Filing]:
        """Get recent filings for a company.

        Args:
            cik: SEC CIK number (with leading zeros)
            filing_type: Filing type (10-K, 10-Q)
            count: Number of filings to retrieve

        Returns:
            List of Filing objects
        """
        # Remove leading zeros for API call
        cik_no_zeros = cik.lstrip("0")

        # Fetch company submissions
        url = f"{self.EDGAR_API}/submissions/CIK{cik}.json"

        connector = aiohttp.TCPConnector(ssl=self._ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch filings for CIK {cik}: {response.status}")
                    return []

                data = await response.json()

        # Parse recent filings
        filings = []
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form == filing_type and len(filings) < count:
                accession = accessions[i].replace("-", "")
                filings.append(
                    Filing(
                        company=data.get("name", "Unknown"),
                        cik=cik,
                        filing_type=form,
                        filing_date=dates[i],
                        accession_number=accessions[i],
                        primary_doc_url=f"{self.BASE_URL}/Archives/edgar/data/{cik_no_zeros}/{accession}/{primary_docs[i]}",
                    )
                )

        return filings

    async def fetch_filing_content(self, filing: Filing) -> str:
        """Fetch the full text content of a filing.

        Args:
            filing: Filing metadata

        Returns:
            Text content of the filing
        """
        connector = aiohttp.TCPConnector(ssl=self._ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(
                filing.primary_doc_url,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch filing: {response.status}")

                content = await response.text()

                # Extract text from HTML
                return self._extract_text_from_html(content)

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML filing."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)

            return "\n".join(lines)

        except ImportError:
            # Fallback: basic regex-based extraction
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()


async def clear_sec_filings() -> int:
    """Clear all SEC filings from the RAG store.

    Returns:
        Number of documents removed
    """
    from victor_rag.document_store import DocumentStore

    store = DocumentStore()
    await store.initialize()

    docs = await store.list_documents()
    sec_docs = [doc for doc in docs if doc.id.startswith("sec_")]

    count = 0
    for doc in sec_docs:
        try:
            await store.delete_document(doc.id)
            count += 1
            logger.info(f"Deleted: {doc.id}")
        except Exception as e:
            logger.error(f"Failed to delete {doc.id}: {e}")

    return count


async def ingest_sec_filings(
    companies: Optional[List[str]] = None,
    filing_type: str = "10-K",
    count: int = 1,
    max_concurrent: int = 5,
) -> Dict[str, int]:
    """Ingest SEC filings into RAG store.

    Args:
        companies: List of company symbols (default: FAANG)
        filing_type: Filing type to ingest
        count: Number of filings per company
        max_concurrent: Maximum concurrent downloads

    Returns:
        Dict mapping company symbols to chunk counts
    """
    from victor_rag.document_store import Document, DocumentStore

    if companies is None:
        companies = COMPANY_PRESETS["faang"]

    # Validate company symbols
    valid_companies = []
    for symbol in companies:
        if symbol in SP500_COMPANIES:
            valid_companies.append(symbol)
        else:
            logger.warning(f"Unknown company symbol: {symbol}")

    if not valid_companies:
        logger.error("No valid company symbols provided")
        return {}

    fetcher = SECFilingFetcher()
    store = DocumentStore()
    await store.initialize()

    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_company(symbol: str) -> Tuple[str, int]:
        async with semaphore:
            company_info = SP500_COMPANIES[symbol]
            logger.info(f"Fetching {filing_type} filings for {company_info['name']}...")

            try:
                # Get filing metadata
                filings = await fetcher.get_company_filings(
                    cik=company_info["cik"],
                    filing_type=filing_type,
                    count=count,
                )

                if not filings:
                    logger.warning(f"No {filing_type} filings found for {symbol}")
                    return symbol, 0

                total_chunks = 0

                for filing in filings:
                    logger.info(f"  Downloading {filing.filing_type} from {filing.filing_date}...")

                    # Fetch content
                    content = await fetcher.fetch_filing_content(filing)

                    if not content:
                        logger.warning(f"  Empty content for {filing.accession_number}")
                        continue

                    # Create document
                    doc_id = f"sec_{symbol}_{filing.filing_type}_{filing.filing_date}"
                    doc = Document(
                        id=doc_id,
                        content=content,
                        source=filing.primary_doc_url,
                        doc_type="html",  # Use HTML-aware chunking
                        metadata={
                            "company": company_info["name"],
                            "symbol": symbol,
                            "cik": company_info["cik"],
                            "sector": company_info["sector"],
                            "filing_type": filing.filing_type,
                            "filing_date": filing.filing_date,
                            "accession_number": filing.accession_number,
                        },
                    )

                    # Ingest
                    chunks = await store.add_document(doc)
                    total_chunks += len(chunks)
                    logger.info(f"  Ingested {len(chunks)} chunks ({len(content):,} chars)")

                return symbol, total_chunks

            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                return symbol, 0

    # Process companies concurrently
    tasks = [process_company(symbol) for symbol in valid_companies]
    for completed in asyncio.as_completed(tasks):
        symbol, chunks = await completed
        results[symbol] = chunks

    return results


async def query_filings(
    query: str,
    top_k: int = 5,
    synthesize: bool = False,
    provider: str = "ollama",
    model: str = None,
    filter_sector: str = None,
    filter_symbol: str = None,
) -> None:
    """Query ingested SEC filings.

    Args:
        query: Search query
        top_k: Number of results to return
        synthesize: Use LLM to synthesize answer
        provider: LLM provider for synthesis
        model: Model name for synthesis
        filter_sector: Filter by sector
        filter_symbol: Filter by company symbol
    """
    from victor_rag.tools.query import RAGQueryTool

    tool = RAGQueryTool()

    logger.info(f"\nQuerying: {query}\n")
    if synthesize:
        logger.info(f"Using {provider} for answer synthesis...")

    # Build metadata filter
    metadata_filter = {}
    if filter_sector:
        metadata_filter["sector"] = filter_sector
    if filter_symbol:
        metadata_filter["symbol"] = filter_symbol.upper()

    result = await tool.execute(
        question=query,
        k=top_k,
        synthesize=synthesize,
        provider=provider if synthesize else None,
        model=model if synthesize else None,
        metadata_filter=metadata_filter if metadata_filter else None,
    )

    if not result.success:
        logger.error(f"Query failed: {result.output}")
        return

    print("=" * 80)
    print(result.output)
    print("=" * 80)


async def show_stats() -> None:
    """Show RAG store statistics."""
    from victor_rag.document_store import DocumentStore

    store = DocumentStore()
    await store.initialize()

    stats = await store.get_stats()

    print("\n" + "=" * 60)
    print("RAG Store Statistics - SEC Filings")
    print("=" * 60)
    print(f"Total documents: {stats.get('total_documents', 0)}")
    print(f"Total chunks: {stats.get('total_chunks', 0)}")
    print(f"Store location: {stats.get('store_path', 'N/A')}")

    docs = await store.list_documents()
    sec_docs = [doc for doc in docs if doc.id.startswith("sec_")]

    if sec_docs:
        print(f"\nSEC Filings: {len(sec_docs)} documents")

        # By company
        by_company: Dict[str, int] = {}
        for doc in sec_docs:
            symbol = doc.metadata.get("symbol", "Other")
            by_company[symbol] = by_company.get(symbol, 0) + 1

        print("\nBy Company:")
        for symbol, count in sorted(by_company.items()):
            name = SP500_COMPANIES.get(symbol, {}).get("name", symbol)
            print(f"  {symbol}: {count} filings - {name}")

        # By sector
        by_sector: Dict[str, int] = {}
        for doc in sec_docs:
            sector = doc.metadata.get("sector", "Other")
            by_sector[sector] = by_sector.get(sector, 0) + 1

        print("\nBy Sector:")
        for sector, count in sorted(by_sector.items(), key=lambda x: -x[1]):
            print(f"  {sector}: {count} filings")

    print("=" * 60)


def list_companies(preset: str = None, sector: str = None) -> None:
    """List available companies."""
    print("\n" + "=" * 60)

    if preset:
        if preset not in COMPANY_PRESETS:
            print(f"Unknown preset: {preset}")
            print(f"Available presets: {', '.join(COMPANY_PRESETS.keys())}")
            return

        symbols = COMPANY_PRESETS[preset]
        print(f"Companies in '{preset}' preset ({len(symbols)} companies):")
        print("-" * 60)
    elif sector:
        symbols = [s for s, info in SP500_COMPANIES.items() if info["sector"] == sector]
        if not symbols:
            print(f"Unknown sector: {sector}")
            sectors = set(info["sector"] for info in SP500_COMPANIES.values())
            print(f"Available sectors: {', '.join(sorted(sectors))}")
            return
        print(f"Companies in '{sector}' sector ({len(symbols)} companies):")
        print("-" * 60)
    else:
        print(f"All S&P 500 Companies ({len(SP500_COMPANIES)} companies):")
        print("-" * 60)
        symbols = list(SP500_COMPANIES.keys())

    for symbol in symbols:
        info = SP500_COMPANIES.get(symbol, {})
        print(f"  {symbol:8} {info.get('name', 'N/A'):40} [{info.get('sector', 'N/A')}]")

    print("\n" + "=" * 60)
    print("\nAvailable Presets:")
    for name, companies in COMPANY_PRESETS.items():
        print(f"  {name:12} - {len(companies)} companies")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SEC Filing RAG Demo - Ingest and query 10-K/10-Q filings for S&P 500 stocks"
    )
    parser.add_argument(
        "--company",
        "-c",
        type=str,
        action="append",
        help="Company symbol(s) to process (can be specified multiple times)",
    )
    parser.add_argument(
        "--preset",
        "-P",
        type=str,
        choices=list(COMPANY_PRESETS.keys()),
        help="Use a preset company group (faang, mag7, top10, top25, top50, top100, tech, healthcare, financials, energy)",
    )
    parser.add_argument(
        "--filing-type",
        "-t",
        type=str,
        default="10-K",
        choices=["10-K", "10-Q"],
        help="Filing type to ingest (default: 10-K)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=1,
        help="Number of filings per company (default: 1)",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Query the ingested filings instead of ingesting",
    )
    parser.add_argument(
        "--synthesize",
        "-S",
        action="store_true",
        help="Use LLM to synthesize answer (requires provider)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default="ollama",
        help="LLM provider for synthesis (default: ollama)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model name for synthesis (provider-specific)",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Show RAG store statistics",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available companies",
    )
    parser.add_argument(
        "--sector",
        type=str,
        help="Filter by sector (for --list or --query)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all SEC filings from RAG store",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent downloads (default: 5)",
    )

    args = parser.parse_args()

    if args.list:
        list_companies(preset=args.preset, sector=args.sector)
        return

    if args.stats:
        asyncio.run(show_stats())
        return

    if args.clear:
        print("\nClearing SEC filings from RAG store...")
        count = asyncio.run(clear_sec_filings())
        print(f"Removed {count} SEC filing documents")
        return

    if args.query:
        asyncio.run(
            query_filings(
                args.query,
                synthesize=args.synthesize,
                provider=args.provider,
                model=args.model,
                filter_sector=args.sector,
                filter_symbol=args.company[0] if args.company else None,
            )
        )
        return

    # Determine companies to process
    if args.preset:
        companies = COMPANY_PRESETS[args.preset]
    elif args.company:
        companies = [c.upper() for c in args.company]
    else:
        # Default to FAANG for quick demo
        companies = COMPANY_PRESETS["faang"]

    print(f"\n{'=' * 60}")
    print("SEC Filing RAG Demo - S&P 500")
    print(f"{'=' * 60}")
    print(f"Filing Type: {args.filing_type}")
    print(f"Companies: {len(companies)} ({args.preset or 'custom'})")
    print(f"Count per company: {args.count}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"{'=' * 60}\n")

    results = asyncio.run(
        ingest_sec_filings(
            companies=companies,
            filing_type=args.filing_type,
            count=args.count,
            max_concurrent=args.max_concurrent,
        )
    )

    print(f"\n{'=' * 60}")
    print("Ingestion Complete!")
    print(f"{'=' * 60}")

    total_chunks = 0
    for symbol, chunks in sorted(results.items()):
        company_name = SP500_COMPANIES.get(symbol, {}).get("name", symbol)
        print(f"  {symbol:8} {company_name:40} {chunks:5} chunks")
        total_chunks += chunks

    print(f"\n  {'TOTAL':8} {'':40} {total_chunks:5} chunks")
    print(f"{'=' * 60}")
    print("\nYou can now query the filings using the Victor CLI:")
    print("")
    print("  # Search only (no LLM):")
    print('  victor rag search "revenue growth"')
    print("")
    print("  # Query with context:")
    print('  victor rag query "What is Apple\'s revenue?"')
    print("")
    print("  # Query with LLM synthesis:")
    print('  victor rag query "Compare NVIDIA and AMD revenue" --synthesize')
    print('  victor rag query "Risk factors in tech sector" -S -p anthropic')
    print("")
    print("  # List ingested documents:")
    print("  victor rag list")
    print("")
    print("  # Show statistics:")
    print("  victor rag stats")
    print("")
    print("  # Advanced: Use this script with filtering:")
    print('  victor rag demo-sec -q "Risk factors" --sector Technology')
    print()


if __name__ == "__main__":
    main()
