"""Formatting utilities for top papers sections in gene reports."""

from typing import List, Dict, Any


def format_top_papers(papers: List[Dict[str, Any]], gene: str, max_papers: int = 5) -> str:
    """
    Format the top papers section for gene report.

    Args:
        papers: List of paper dictionaries from PubMedQueryAgent
        gene: Gene symbol
        max_papers: Maximum number of papers to display (default: 5)

    Returns:
        Formatted string for report
    """
    if not papers:
        return f"""
TOP RELEVANT PUBLICATIONS:
No highly relevant publications found for {gene} in the specific context of this study.
This may indicate a novel research area or that this gene has not been extensively
studied in this particular biological context.
"""

    sections = ["\nTOP RELEVANT PUBLICATIONS:\n"]

    for i, paper in enumerate(papers[:max_papers], 1):
        # Format relevance score with visual indicator
        relevance = paper.get('relevance_score', 0)
        if relevance >= 9:
            relevance_indicator = "⭐⭐⭐ Highly Relevant"
        elif relevance >= 7:
            relevance_indicator = "⭐⭐ Very Relevant"
        elif relevance >= 5:
            relevance_indicator = "⭐ Relevant"
        else:
            relevance_indicator = "Mentioned"

        # Truncate abstract to reasonable length
        abstract = paper.get('abstract', 'No abstract available')
        if len(abstract) > 400:
            abstract = abstract[:397] + "..."

        # Build PubMed URL
        pmid = paper.get('pmid', 'N/A')
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != 'N/A' else 'N/A'

        section = f"""
Paper #{i} - {relevance_indicator} (Score: {relevance:.1f}/10)
{'─' * 80}
Title: {paper.get('title', 'No title')}
Journal: {paper.get('journal', 'Unknown')} ({paper.get('year', 'N/A')})
PMID: {pmid}
URL: {pubmed_url}

Relevance: {paper.get('reason', 'No explanation provided')}

Abstract: {abstract}
"""
        sections.append(section)

    return "\n".join(sections)


def format_top_papers_summary(gene: str, top_papers_data: Dict[str, Any]) -> str:
    """
    Format a summary section for top papers retrieval.

    Args:
        gene: Gene symbol
        top_papers_data: Dictionary containing top_papers list and metadata

    Returns:
        Formatted summary string
    """
    papers = top_papers_data.get('top_papers', [])
    query_level = top_papers_data.get('query_level_used', 'unknown')
    total_found = top_papers_data.get('total_papers_found', 0)

    if not papers:
        return f"\nNo relevant publications retrieved for {gene}\n"

    summary = f"""
Top Papers Summary:
  Papers retrieved: {len(papers)}
  Total papers found: {total_found}
  Query level used: {query_level}
"""

    return summary
