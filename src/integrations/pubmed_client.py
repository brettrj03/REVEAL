from typing import List, Dict, Any
import asyncio
import httpx
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import os
import warnings

# Placeholder email patterns that indicate unconfigured credentials
_PLACEHOLDER_EMAIL_PATTERNS = [
    "your.email@example.com",
    "your-email@example.com",
    "example@example.com",
    "user@example.com",
]


def _is_placeholder_email(email: str) -> bool:
    """Check if the email appears to be a placeholder value."""
    if not email:
        return True
    email_lower = email.lower().strip()
    return email_lower in _PLACEHOLDER_EMAIL_PATTERNS or "@example.com" in email_lower


def check_ncbi_credentials() -> dict:
    """
    Check the current NCBI credentials configuration.

    Returns a dict with:
        - email_ok: bool - True if email is properly configured
        - api_key_ok: bool - True if API key is properly configured (or not set)
        - email_value: str - The current email value
        - has_api_key: bool - True if an API key is configured
        - warning_message: str | None - A warning message if there are issues
    """
    email = os.getenv("NCBI_EMAIL", "")
    api_key = os.getenv("NCBI_API_KEY", "")

    email_is_placeholder = _is_placeholder_email(email)

    # Check if the API key env var exists but is empty
    api_key_set_but_empty = "NCBI_API_KEY" in os.environ and not os.environ["NCBI_API_KEY"].strip()

    warning_parts = []
    if email_is_placeholder:
        warning_parts.append(
            "NCBI_EMAIL is still set to the placeholder value. "
            "PubMed searches will fail."
        )

    if api_key_set_but_empty:
        warning_parts.append(
            "NCBI_API_KEY is set but empty. "
            "If you don't have an NCBI API key, remove or comment out the NCBI_API_KEY line entirely."
        )

    warning_message = " ".join(warning_parts) if warning_parts else None

    return {
        "email_ok": not email_is_placeholder,
        "api_key_ok": not api_key_set_but_empty,
        "email_value": email,
        "has_api_key": bool(api_key and api_key.strip()),
        "warning_message": warning_message,
    }


class PubMedClient:
    """Client for NCBI E-utilities PubMed API."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str | None = None, api_key: str | None = None):
        # Try environment variables first, then fall back to parameters
        self.email = email or os.getenv("NCBI_EMAIL") or "your.email@example.com"

        # Handle API key: only use it if it's a non-empty string
        raw_api_key = api_key or os.getenv("NCBI_API_KEY")
        self.api_key = raw_api_key if raw_api_key and raw_api_key.strip() else None

        # Warn if email looks like a placeholder
        if _is_placeholder_email(self.email):
            warnings.warn(
                "NCBI_EMAIL appears to be a placeholder value. "
                "PubMed API requests may fail with 400 Bad Request. "
                "Please set NCBI_EMAIL to your real email address in your .env file.",
                UserWarning,
                stacklevel=2
            )

    async def count(self, query: str) -> int:
        """Return total hits for query using ESearch."""
        url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "rettype": "count",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        print(f"🔍 PUBMED QUERY: {query}")
        try:
            await asyncio.sleep(0.35)  # Stay under NCBI rate limit (3 req/sec)
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

                # Parse XML response
                root = ET.fromstring(response.text)
                count_elem = root.find(".//Count")
                if count_elem is not None and count_elem.text:
                    count = int(count_elem.text)
                    print(f"   ✓ Result: {count} papers")
                    return count

                print("   ❌ Error: No count in response")
                return 0
        except Exception as exc:
            print(f"   ❌ Error: {exc}")
            return 0

    async def search_pmids(self, query: str, retmax: int = 20, sort: str = "relevance") -> List[str]:
        """Return list of PMIDs for query using ESearch.

        Args:
            query: PubMed search query
            retmax: Maximum number of PMIDs to return
            sort: Sort order - "relevance" (Best Match) or "pub_date" (newest first)
        """
        url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "xml",
            "email": self.email,
            "sort": sort,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        for attempt in range(3):
            try:
                await asyncio.sleep(0.35)  # Stay under NCBI rate limit (3 req/sec)
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()

                    # Parse XML response
                    root = ET.fromstring(response.text)
                    id_elements = root.findall(".//Id")
                    return [elem.text for elem in id_elements if elem.text]
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 1.0 * (2 ** attempt)  # 1s, 2s
                print(f"    PubMed request failed (attempt {attempt+1}/3), retrying in {wait}s: {type(e).__name__}")
                await asyncio.sleep(wait)

    async def fetch_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Return list of records with pmid, title, abstract, year, journal, publication_types using EFetch."""
        if not pmids:
            return []

        url = f"{self.BASE_URL}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        for attempt in range(3):
            try:
                await asyncio.sleep(0.35)  # Stay under NCBI rate limit (3 req/sec)
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()

                    # Parse XML response
                    root = ET.fromstring(response.text)
                    records = []

                    for article in root.findall(".//PubmedArticle"):
                        record = self._parse_article(article)
                        if record:
                            records.append(record)

                    return records
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 1.0 * (2 ** attempt)  # 1s, 2s
                print(f"    PubMed request failed (attempt {attempt+1}/3), retrying in {wait}s: {type(e).__name__}")
                await asyncio.sleep(wait)

    def _parse_article(self, article: ET.Element) -> Dict[str, Any] | None:
        """Parse a single PubmedArticle XML element."""
        try:
            # PMID
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None or not pmid_elem.text:
                return None
            pmid = pmid_elem.text

            # Title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None and title_elem.text else "No title"

            # Abstract (combine all AbstractText elements)
            abstract_parts = []
            for abs_elem in article.findall(".//AbstractText"):
                if abs_elem.text:
                    label = abs_elem.get("Label")
                    if label:
                        abstract_parts.append(f"{label}: {abs_elem.text}")
                    else:
                        abstract_parts.append(abs_elem.text)
            abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available"

            # Year
            year_elem = article.find(".//PubDate/Year")
            year = year_elem.text if year_elem is not None and year_elem.text else "n.d."

            # Journal
            journal_elem = article.find(".//Journal/Title")
            if journal_elem is None:
                journal_elem = article.find(".//Journal/ISOAbbreviation")
            journal = journal_elem.text if journal_elem is not None and journal_elem.text else "Unknown journal"

            # Publication types
            pub_types = []
            for pt_elem in article.findall(".//PublicationType"):
                if pt_elem.text:
                    pub_types.append(pt_elem.text)

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "publication_types": pub_types,
            }
        except Exception as e:
            # If parsing fails for this article, skip it
            return None
