"""BM25 pre-ranking of papers before expensive LLM ranking.

Quickly filters out obviously irrelevant papers based on how well
their abstracts match the user's experimental context terms.
This runs BEFORE the LLM ranking step to reduce cost and time.
"""

import logging
import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> List[str]:
    """Simple whitespace tokenisation with lowercasing.

    Strips basic punctuation so that 'MHC-II,' matches 'MHC-II'.
    """
    # Lowercase and split on whitespace
    tokens = text.lower().split()
    # Strip leading/trailing punctuation from each token
    tokens = [re.sub(r'^[^\w]+|[^\w]+$', '', t) for t in tokens]
    # Remove empty tokens
    return [t for t in tokens if t]


def prerank_papers(
    papers: List[Dict[str, Any]],
    context_terms: List[str],
    gene_symbol: str = "",
    top_n: int = 15,
    abstract_key: str = "abstract",
) -> List[Dict[str, Any]]:
    """Pre-rank papers using BM25 against experimental context terms.

    Args:
        papers: List of paper dicts, each must have an abstract field.
        context_terms: List of context terms extracted from user query
                      (e.g., ["myeloid cells", "MHC class II", "antigen presentation",
                       "CIITA", "MYC"]).
        gene_symbol: Gene symbol for logging purposes.
        top_n: Number of papers to keep after ranking.
        abstract_key: Key in paper dict that holds the abstract text.

    Returns:
        Top N papers sorted by BM25 relevance score.
    """
    if not papers:
        return []

    if not context_terms:
        logger.warning(
            f"No context terms provided for BM25 ranking of {gene_symbol}, "
            f"returning all {len(papers)} papers unranked."
        )
        return papers

    # If we have fewer papers than top_n, just return them all
    if len(papers) <= top_n:
        logger.info(
            f"BM25 [{gene_symbol}]: Only {len(papers)} papers, "
            f"keeping all (threshold: {top_n})."
        )
        return papers

    # Build corpus from abstracts
    corpus = []
    valid_indices = []
    for i, paper in enumerate(papers):
        abstract = paper.get(abstract_key, "") or ""
        # Also include title if available for better matching
        title = paper.get("title", "") or ""
        text = f"{title} {abstract}".strip()
        if text:
            corpus.append(_tokenise(text))
            valid_indices.append(i)
        else:
            # Papers without abstracts get lowest priority
            valid_indices.append(i)
            corpus.append([])

    # Build query from context terms
    query = _tokenise(" ".join(context_terms))

    # Score with BM25
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)

    # Pair papers with scores and sort
    scored_papers = list(zip(valid_indices, scores))
    scored_papers.sort(key=lambda x: x[1], reverse=True)

    # Log what we're keeping vs cutting
    kept = scored_papers[:top_n]
    cut = scored_papers[top_n:]

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            f"BM25 [{gene_symbol}]: Ranked {len(papers)} papers, "
            f"keeping top {top_n}."
        )
        if kept:
            top_score = kept[0][1]
            bottom_kept_score = kept[-1][1]
            logger.info(f"  Score range (kept): {top_score:.2f} - {bottom_kept_score:.2f}")
        if cut:
            top_cut_score = cut[0][1]
            logger.info(f"  Highest cut score: {top_cut_score:.2f}")

    if logger.isEnabledFor(logging.DEBUG):
        for idx, score in kept[:5]:
            title = papers[idx].get("title", "No title")[:80]
            logger.debug(f"  KEPT (score {score:.2f}): {title}")
        for idx, score in cut[:3]:
            title = papers[idx].get("title", "No title")[:80]
            logger.debug(f"  CUT  (score {score:.2f}): {title}")

    # Return top N papers in ranked order with BM25 rank and score attached
    ranked_papers = []
    for rank, (idx, score) in enumerate(kept, start=1):
        paper = papers[idx].copy()  # Don't modify original
        paper['bm25_rank'] = rank
        paper['bm25_score'] = score
        ranked_papers.append(paper)

    return ranked_papers


def build_context_query_terms(
    literature_context_terms: Dict[str, Any],
    user_query: str = "",
) -> List[str]:
    """Build a list of context terms for BM25 ranking from state data.

    Args:
        literature_context_terms: Dict with 'diseases', 'processes', 'cell_types',
                                  'modifiers', 'context_genes', etc.
        user_query: Original user query (optional, for extracting additional terms).

    Returns:
        List of context terms suitable for BM25 ranking.
    """
    context_terms = []

    # Add diseases
    diseases = literature_context_terms.get("diseases", [])
    if diseases:
        context_terms.extend(diseases)

    # Add biological processes
    processes = literature_context_terms.get("processes", [])
    if processes:
        context_terms.extend(processes)

    # Add cell types
    cell_types = literature_context_terms.get("cell_types", [])
    if cell_types:
        context_terms.extend(cell_types)

    # Add modifiers
    modifiers = literature_context_terms.get("modifiers", [])
    if modifiers:
        context_terms.extend(modifiers)

    # Add context genes (MYC, CIITA, etc. mentioned in user's biological question)
    context_genes = literature_context_terms.get("context_genes", [])
    if context_genes:
        context_terms.extend(context_genes)

    # Deduplicate while preserving order
    seen = set()
    unique_terms = []
    for term in context_terms:
        term_lower = term.lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(term)

    return unique_terms
