"""
Format the simplified literature analysis section with top papers.
"""

from typing import Dict, Any


def format_literature_analysis_section(gene_symbol: str, literature_data: Dict[str, Any]) -> str:
    """
    Format the new simplified literature analysis section.

    Shows query used, total papers found, and top 10 ranked papers.

    Args:
        gene_symbol: Gene symbol
        literature_data: Dictionary with 'top_papers', 'total_papers_found', 'query_used'

    Returns:
        Formatted literature section as string
    """
    sections = []
    sections.append("\n" + "=" * 80)
    sections.append("LITERATURE ANALYSIS")
    sections.append("=" * 80)

    # Get data
    top_papers = literature_data.get('top_papers', [])
    total_found = literature_data.get('total_papers_found', 0)
    query_used = literature_data.get('query_used', 'N/A')

    sections.append(f"\nTotal papers in context: {total_found:,}")

    # Show query (truncated if too long)
    if len(query_used) > 150:
        sections.append(f"Query: {query_used[:147]}...")
    else:
        sections.append(f"Query: {query_used}")

    sections.append("")

    if not top_papers:
        sections.append("⚠️  No highly relevant papers found in this specific context")
        sections.append("")
        return "\n".join(sections)

    sections.append(f"TOP {len(top_papers)} MOST RELEVANT PAPERS:\n")

    for i, paper in enumerate(top_papers, 1):
        # Extract first author
        authors = paper.get('authors', 'Unknown')
        if isinstance(authors, list):
            first_author = authors[0] if authors else 'Unknown'
        else:
            first_author = authors.split(',')[0].strip() if authors else 'Unknown'

        # Relevance stars
        relevance_score = paper.get('relevance_score', 3)
        stars = "★" * min(int(relevance_score), 5)

        # Format paper
        sections.append(f"{i}. {paper.get('title', 'No title')}")
        sections.append(f"   {first_author} et al., {paper.get('journal', 'Unknown')} ({paper.get('year', 'N/A')})")
        sections.append(f"   PMID: {paper.get('pmid', 'N/A')} | Relevance: {stars} ({relevance_score}/5)")

        # Key finding if available
        key_finding = paper.get('key_finding')
        if key_finding:
            sections.append("")
            sections.append("   Key Finding:")
            # Wrap long findings
            if len(key_finding) > 76:
                # Simple wrap at word boundaries
                words = key_finding.split()
                line = "   "
                for word in words:
                    if len(line) + len(word) + 1 > 76:
                        sections.append(line)
                        line = "   " + word
                    else:
                        line += (" " + word) if line != "   " else word
                if line.strip():
                    sections.append(line)
            else:
                sections.append(f"   {key_finding}")

        sections.append("")

    return "\n".join(sections)
