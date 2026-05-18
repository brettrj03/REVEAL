"""
Rank papers by relevance using LLM and select top 10 for display.

Enhanced to include gene disambiguation - validates that papers are actually
discussing the target gene and not coincidentally mentioning an alias in a
different context (e.g., CAT scans vs CAT gene, ACE inhibitors vs ACE gene).

Parallelized: Multiple genes are ranked simultaneously for faster processing.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Dict, Optional, Tuple
from dataclasses import dataclass
import asyncio
import json
import logging
import os
import time
from openai import AsyncOpenAI
from pydantic_graph.nodes import BaseNode, GraphRunContext
from src.graph.state import GeneState, _accumulate_tokens
from src.config import get_active_model
from src.validation_config.validation_config import MAX_CONCURRENT_GENE_RANKINGS
from src.utils.phoenix_tracing import node_span

if TYPE_CHECKING:
    from src.nodes.analyze_literature_findings import AnalyzeLiteratureFindings

logger = logging.getLogger(__name__)

# Semaphore for rate limiting parallel gene rankings
_gene_ranking_semaphore: asyncio.Semaphore = None

# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK PAPERS - Known reference papers for dissertation validation
# ═══════════════════════════════════════════════════════════════════════════════
BENCHMARK_PAPERS = {
    "39917766": "Shaw et al. 2025 (MED12 — Molecular Medicine)",
    # Add more benchmark PMIDs here as needed:
    # "XXXXXXXX": "Author et al. YEAR (GENE — Journal)",
}

# Source paper patterns for thesis benchmarking
SOURCE_PAPER_PATTERNS = {
    "MED12": {
        "keywords": ["MED12", "Arg1138", "neurodevelopmental"],
        "pmid": "39917766",  # If known
        "authors": ["Shaw"]
    },
    "SETBP1": {
        "keywords": ["SETBP1", "haploinsufficiency"],
        "authors": ["Shaw"]
    },
    "GATA4": {
        "keywords": ["GATA4", "Arg284", "congenital heart"],
        "authors": ["Forbes"]
    }
}


def _get_gene_ranking_semaphore() -> asyncio.Semaphore:
    """
    Get or create the gene ranking semaphore (lazy initialization).

    Recreates the semaphore if the current event loop differs from the one
    it was created in, preventing 'bound to a different event loop' errors.
    """
    global _gene_ranking_semaphore

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running - create semaphore anyway
        current_loop = None

    # Check if semaphore needs to be created or recreated
    if _gene_ranking_semaphore is None:
        _gene_ranking_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENE_RANKINGS)
    elif current_loop is not None:
        # Check if semaphore is bound to a different loop
        try:
            # Try to get the semaphore's loop
            semaphore_loop = getattr(_gene_ranking_semaphore, '_loop', None)
            if semaphore_loop is not None and semaphore_loop is not current_loop:
                # Recreate semaphore for current loop
                logger.info("Recreating gene ranking semaphore for new event loop")
                _gene_ranking_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENE_RANKINGS)
        except Exception:
            # If we can't check, recreate to be safe
            _gene_ranking_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENE_RANKINGS)

    return _gene_ranking_semaphore


def _safe_year(paper: Dict[str, Any]) -> int:
    """Safely parse year from paper dict, handling 'n.d.', None, empty string, etc."""
    year_val = paper.get('year', 0)
    if not year_val:
        return 0
    try:
        return int(year_val)
    except (ValueError, TypeError):
        return 0


def _check_source_paper_match(paper: Dict[str, Any], user_query: str) -> tuple[bool, str | None]:
    """
    Check if a paper matches known source paper patterns for thesis benchmarking.

    Args:
        paper: Paper dict with title, abstract, authors, pmid
        user_query: User's research question (to infer which query this is)

    Returns:
        Tuple of (is_source_paper, pmid_if_found)
    """
    title = (paper.get('title', '') or '').lower()
    abstract = (paper.get('abstract', '') or '').lower()
    authors = (paper.get('authors', '') or '').lower()
    pmid = paper.get('pmid', '')

    combined_text = f"{title} {abstract} {authors}"

    # Check each source paper pattern
    for query_type, pattern in SOURCE_PAPER_PATTERNS.items():
        # Check if this query is about this gene
        if query_type.lower() not in user_query.lower():
            continue

        # Check PMID match (if known)
        if 'pmid' in pattern and pmid == pattern['pmid']:
            return True, pmid

        # Check keyword matches
        keywords = pattern.get('keywords', [])
        author_names = pattern.get('authors', [])

        # Must match at least 2 keywords OR 1 keyword + author
        keyword_matches = sum(1 for kw in keywords if kw.lower() in combined_text)
        author_matches = sum(1 for author in author_names if author.lower() in authors)

        if keyword_matches >= 2 or (keyword_matches >= 1 and author_matches >= 1):
            return True, pmid

    return False, None


def _get_gene_info_for_ranking(gene_symbol: str, state: GeneState) -> Dict[str, Any]:
    """
    Get gene information for LLM ranking prompt.

    Returns:
        Dict with 'official_name', 'aliases', and 'description'
    """
    gene_info = {
        'symbol': gene_symbol,
        'official_name': None,
        'aliases': [],
        'description': None
    }

    # Try to get from state.all_gene_data
    gene_data = state.all_gene_data.get(gene_symbol, {})

    # Get official name
    gene_info['official_name'] = (
        gene_data.get('full_name') or
        gene_data.get('name') or
        gene_data.get('function', {}).get('description', '')
    )

    # Get aliases (handle dict format)
    raw_aliases = gene_data.get('aliases', [])
    for alias in raw_aliases:
        if isinstance(alias, dict):
            alias_str = alias.get('symbol_alias', '')
        else:
            alias_str = str(alias)
        if alias_str and alias_str.upper() != gene_symbol.upper():
            gene_info['aliases'].append(alias_str)

    # Get description from function
    if 'function' in gene_data and isinstance(gene_data['function'], dict):
        gene_info['description'] = gene_data['function'].get('description', '')

    return gene_info


def _print_ranking_transparency_table(
    gene: str,
    top_papers: List[Dict[str, Any]],
    total_candidates: int
) -> None:
    """
    Print a formatted ranking transparency table showing how papers moved
    between BM25 and LLM ranking stages.

    Args:
        gene: Gene symbol
        top_papers: Final top N papers with bm25_rank, relevance_score, tier_fetched
        total_candidates: Total number of candidates before BM25
    """
    print()
    print("┌─────────────────────────────────────────────────────────────────────────┐")
    print(f"│ RANKING TRANSPARENCY — {gene:<50} │")
    print("├──────┬──────┬───────┬──────┬──────┬────────────────────────────────────┤")
    print("│ LLM# │ BM25#│ Move  │ Tier │ Stars│ Title (truncated)                  │")
    print("├──────┼──────┼───────┼──────┼──────┼────────────────────────────────────┤")

    rank_changes = []
    tier_counts = {'T1': 0, 'T2': 0, 'T3': 0, 'T4': 0, 'T5': 0, 'T6': 0, 'other': 0}
    benchmark_detected = None

    for llm_rank, paper in enumerate(top_papers, start=1):
        bm25_rank = paper.get('bm25_rank', '?')
        tier = paper.get('tier_fetched', '?')
        title = paper.get('title', 'No title')[:40]
        relevance_score = paper.get('relevance_score', 3)
        pmid = paper.get('pmid', '')

        # Calculate rank movement
        if isinstance(bm25_rank, int):
            rank_change = bm25_rank - llm_rank
            rank_changes.append(rank_change)
            if rank_change > 0:
                move_str = f"▲{rank_change}"
            elif rank_change < 0:
                move_str = f"▼{abs(rank_change)}"
            else:
                move_str = "="
        else:
            move_str = "?"

        # Count tier distribution
        if tier in tier_counts:
            tier_counts[tier] += 1
        else:
            tier_counts['other'] += 1

        # Convert relevance score to stars
        stars = "★" * relevance_score + "☆" * (5 - relevance_score)

        # Check if benchmark paper
        is_benchmark = pmid in BENCHMARK_PAPERS
        if is_benchmark:
            benchmark_detected = {
                'pmid': pmid,
                'label': BENCHMARK_PAPERS[pmid],
                'llm_rank': llm_rank,
                'bm25_rank': bm25_rank,
                'tier': tier,
                'stars': stars
            }

        # Print row
        print(f"│  {llm_rank:<2}  │  {bm25_rank:<2}  │ {move_str:<5} │ {tier:<4} │ {stars} │ {title:<38} │")

    print("└──────┴──────┴───────┴──────┴──────┴────────────────────────────────────┘")
    print()

    # Print summary statistics
    if rank_changes:
        avg_change = sum(rank_changes) / len(rank_changes)
        print(f"  Avg BM25→LLM rank change: {avg_change:+.1f} positions")
    else:
        print("  Avg BM25→LLM rank change: N/A")

    # Tier distribution
    high_specificity = tier_counts.get('T1', 0) + tier_counts.get('T2', 0)
    broad_fallback = tier_counts.get('T3', 0) + tier_counts.get('T4', 0) + tier_counts.get('T5', 0) + tier_counts.get('T6', 0)
    print(f"  Papers from Tier 1/2 (high specificity): {high_specificity}")
    print(f"  Papers from Tier 3/4/5/6 (broad fallback): {broad_fallback}")

    # Benchmark paper detection
    if benchmark_detected:
        print()
        print("  *** BENCHMARK PAPER DETECTED ***")
        print(f"  {benchmark_detected['label']}")
        print(f"  LLM rank: #{benchmark_detected['llm_rank']}  |  BM25 rank: #{benchmark_detected['bm25_rank']}  |  Tier: {benchmark_detected['tier']}  |  Stars: {benchmark_detected['stars']}")
    else:
        print(f"  Benchmark paper detected: NO")

    print()


def _build_ranking_detail_for_state(
    gene: str,
    top_papers: List[Dict[str, Any]],
    total_candidates: int
) -> Dict[str, Any]:
    """
    Build detailed ranking information for storage in state.

    Args:
        gene: Gene symbol
        top_papers: Final top N papers with bm25_rank, relevance_score, tier_fetched
        total_candidates: Total number of candidates before BM25

    Returns:
        Dict with detailed ranking information
    """
    papers_detail = []
    rank_changes = []
    tier_counts = {'T1': 0, 'T2': 0, 'T3': 0, 'T4': 0, 'T5': 0, 'T6': 0, 'other': 0}
    benchmark_detected = False
    benchmark_llm_rank = None
    benchmark_bm25_rank = None

    for llm_rank, paper in enumerate(top_papers, start=1):
        bm25_rank = paper.get('bm25_rank')
        tier = paper.get('tier_fetched', 'unknown')
        pmid = paper.get('pmid', '')

        # Calculate rank change
        rank_change = None
        if isinstance(bm25_rank, int):
            rank_change = bm25_rank - llm_rank
            rank_changes.append(rank_change)

        # Count tier
        if tier in tier_counts:
            tier_counts[tier] += 1
        else:
            tier_counts['other'] += 1

        # Check benchmark
        is_benchmark = pmid in BENCHMARK_PAPERS

        if is_benchmark:
            benchmark_detected = True
            benchmark_llm_rank = llm_rank
            benchmark_bm25_rank = bm25_rank

        papers_detail.append({
            'llm_rank': llm_rank,
            'bm25_rank': bm25_rank,
            'rank_change': rank_change,
            'tier': tier,
            'pmid': pmid,
            'title': paper.get('title', ''),
            'stars': paper.get('relevance_score', 3),
            'is_benchmark': is_benchmark,
            'benchmark_label': BENCHMARK_PAPERS.get(pmid) if is_benchmark else None
        })

    # Calculate averages
    avg_rank_change = sum(rank_changes) / len(rank_changes) if rank_changes else 0.0
    high_specificity = tier_counts.get('T1', 0) + tier_counts.get('T2', 0)
    broad_fallback = tier_counts.get('T3', 0) + tier_counts.get('T4', 0) + tier_counts.get('T5', 0) + tier_counts.get('T6', 0)

    return {
        'gene': gene,
        'papers': papers_detail,
        'avg_rank_change': avg_rank_change,
        'papers_from_high_specificity_tiers': high_specificity,
        'papers_from_broad_tiers': broad_fallback,
        'benchmark_paper_detected': benchmark_detected,
        'benchmark_paper_llm_rank': benchmark_llm_rank,
        'benchmark_paper_bm25_rank': benchmark_bm25_rank,
        'total_candidates_before_bm25': total_candidates,
    }


@dataclass
class RankPapersByRelevance(BaseNode[GeneState]):
    """
    Rank candidate papers and select top 10 for display.

    Reads from: state.gene_literature_candidates (populated by FetchAdaptiveLiterature)
    Writes to: state.gene_top_papers (top 10 papers with relevance scores)
    """

    top_n: int = 10

    async def run(self, ctx: GraphRunContext[GeneState]) -> "AnalyzeLiteratureFindings":
        from src.nodes.analyze_literature_findings import AnalyzeLiteratureFindings

        print(f"\n{'='*70}")
        print("NODE: Rank Papers by Relevance")
        print(f"{'='*70}")

        state = ctx.state
        _t0 = time.perf_counter()
        try:
            genes = state.get_genes_found()

            # Check output mode and API key
            output_mode = state.output_mode if hasattr(state, 'output_mode') else 'interpreted'
            openai_key = os.getenv('OPENAI_API_KEY')

            if output_mode == 'factual':
                print("Factual mode: using fallback ranking (sort by year)")
                use_llm = False
            elif not openai_key:
                print("No OpenAI API key, using fallback (sort by year)")
                use_llm = False
            else:
                print(f"Using LLM ranking (selecting top {self.top_n})")
                use_llm = True

            print()

            if len(genes) > 1:
                print(f"  Processing {len(genes)} genes in parallel...")

            # Process all genes in parallel
            ranking_tasks = [
                self._rank_gene_with_limit(
                    gene=gene,
                    state=state,
                    use_llm=use_llm
                )
                for gene in genes
            ]

            # Wait for all genes to finish
            results = await asyncio.gather(*ranking_tasks, return_exceptions=True)

            # Process results and update state
            for gene, result in zip(genes, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to rank papers for {gene}: {result}")
                    print(f"  {gene}: Ranking failed: {result}")
                    # Fallback: just take first N candidates
                    candidates_info = state.gene_literature_candidates.get(gene, {})
                    candidates = candidates_info.get('candidate_papers', [])
                    top_papers = candidates[:self.top_n]
                    state.gene_top_papers[gene] = {
                        'top_papers': top_papers,
                        'pmids': [p.get('pmid') for p in top_papers],
                        'total_candidates': len(candidates),
                        'ranking_method': 'fallback'
                    }
                else:
                    # Unpack successful result
                    gene_result = result
                    state.gene_top_papers[gene] = gene_result
                    top_count = len(gene_result.get('top_papers', []))
                    total_count = gene_result.get('total_candidates', 0)
                    print(f"  {gene}: Selected top {top_count} papers (from {total_count} candidates)")

            # Summary
            total_papers = sum(
                len(state.gene_top_papers.get(gene, {}).get('top_papers', []))
                for gene in genes
            )

            print()
            print(f"{'='*70}")
            print("RANKING COMPLETE")
            print(f"{'='*70}")
            print(f"Genes processed: {len(genes)}")
            print(f"Total papers selected: {total_papers}")
            if genes:
                print(f"Average per gene: {total_papers/len(genes):.1f}")
            print(f"{'='*70}\n")

            # ═══════════════════════════════════════════════════════════════════
            # UPDATE LITERATURE RETRIEVAL STATS FOR THESIS BENCHMARKING
            # ═══════════════════════════════════════════════════════════════════
            for gene in genes:
                if gene in state.gene_top_papers and gene in state.literature_retrieval_stats:
                    top_papers = state.gene_top_papers[gene].get('top_papers', [])
                    stats = state.literature_retrieval_stats[gene]

                    # Update final papers count
                    stats['final_papers_count'] = len(top_papers)

                    # Check for source paper in top papers
                    source_found = False
                    source_pmid = None
                    for paper in top_papers:
                        is_source, pmid = _check_source_paper_match(paper, state.user_query)
                        if is_source:
                            source_found = True
                            source_pmid = pmid
                            logger.info(f"Source paper found for {gene}: PMID {pmid}")
                            break

                    stats['source_paper_found'] = source_found
                    stats['source_paper_pmid'] = source_pmid

            return AnalyzeLiteratureFindings()
        finally:
            state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )

    async def _rank_gene_with_limit(
        self,
        gene: str,
        state: GeneState,
        use_llm: bool
    ) -> Dict[str, Any]:
        """
        Rank papers for a single gene with rate limiting.

        Args:
            gene: Gene symbol to rank papers for
            state: Pipeline state
            use_llm: Whether to use LLM ranking

        Returns:
            Dict with ranking results for this gene
        """
        semaphore = _get_gene_ranking_semaphore()

        async with semaphore:
            # Get candidates from adaptive literature fetcher
            candidates_info = state.gene_literature_candidates.get(gene, {})
            candidates = candidates_info.get('candidate_papers', [])

            with node_span(
                "RankPapersByRelevance",
                gene=gene,
                n_candidates=len(candidates),
                use_llm=bool(use_llm)
            ) as span:
                if not candidates:
                    span.set_attribute("ranking_method", "none")
                    span.set_attribute("n_selected", 0)
                    return {
                        'top_papers': [],
                        'pmids': [],
                        'total_candidates': 0,
                        'ranking_method': 'none'
                    }

                if use_llm:
                    # Get gene info for disambiguation
                    gene_info = _get_gene_info_for_ranking(gene, state)

                    # LLM ranking with disambiguation
                    ranked = await self._rank_with_llm(
                        gene=gene,
                        gene_info=gene_info,
                        query=state.user_query,
                        context=state.experiment_context,
                        papers=candidates,
                        state=state
                    )
                    ranking_method = 'llm'
                else:
                    # Fallback: sort by year
                    ranked = sorted(
                        candidates,
                        key=_safe_year,
                        reverse=True
                    )
                    ranking_method = 'year'

                # Return top N
                top_papers = ranked[:self.top_n]
                span.set_attribute("ranking_method", ranking_method)
                span.set_attribute("n_selected", len(top_papers))

                # ═══════════════════════════════════════════════════════════════════
                # RANKING TRANSPARENCY LOGGING (for dissertation benchmarking)
                # ═══════════════════════════════════════════════════════════════════
                if top_papers and use_llm:
                    # Print transparency table to console
                    _print_ranking_transparency_table(gene, top_papers, len(candidates))

                    # Build detailed ranking info for state storage
                    ranking_detail = _build_ranking_detail_for_state(gene, top_papers, len(candidates))

                    # Store in state for Streamlit UI display
                    if gene not in state.all_gene_data:
                        state.all_gene_data[gene] = {}
                    state.all_gene_data[gene]['literature_ranking_detail'] = ranking_detail

                return {
                    'top_papers': top_papers,
                    'pmids': [p.get('pmid') for p in top_papers],
                    'total_candidates': len(candidates),
                    'ranking_method': ranking_method,
                    'fetch_strategy': candidates_info.get('fetch_strategy', 'unknown'),
                    'query_used': candidates_info.get('query_used', ''),
                    'tier_used': candidates_info.get('tier_used', '')
                }

    async def _rank_with_llm(
        self,
        gene: str,
        gene_info: Dict[str, Any],
        query: str,
        context,
        papers: List[Dict[str, Any]],
        state: GeneState
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to rank papers by relevance with gene disambiguation.

        Args:
            gene: Gene symbol (e.g., "TP53")
            gene_info: Dict with 'official_name', 'aliases', 'description'
            query: User's research question
            context: Experiment context
            papers: Candidate papers to rank

        Returns:
            List of papers sorted by relevance score
        """
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Format papers for LLM
        papers_formatted = []
        for i, paper in enumerate(papers):
            # Truncate abstract
            abstract = paper.get('abstract', 'No abstract available')
            if len(abstract) > 1200:
                abstract = abstract[:1197] + "..."

            papers_formatted.append({
                "index": i,
                "pmid": paper.get('pmid'),
                "title": paper.get('title', 'No title'),
                "abstract": abstract,
                "year": paper.get('year', 'N/A'),
                "journal": paper.get('journal', 'Unknown')
            })

        # Build context string
        context_parts = []
        if hasattr(context, 'cell_type') and context.cell_type:
            context_parts.append(f"Cell type: {context.cell_type}")
        if hasattr(context, 'tissue') and context.tissue:
            context_parts.append(f"Tissue: {context.tissue}")
        if hasattr(context, 'organism') and context.organism:
            context_parts.append(f"Species: {context.organism}")

        context_str = " | ".join(context_parts) if context_parts else "General context"

        # Build gene identification section
        official_name = gene_info.get('official_name') or gene
        aliases = gene_info.get('aliases', [])
        aliases_str = ", ".join(aliases[:10]) if aliases else "none"

        # Format papers for prompt
        papers_text = self._format_papers_for_prompt(papers_formatted)

        # Create enhanced prompt with disambiguation instructions
        prompt = f"""You are ranking scientific papers by relevance to a research question about a SPECIFIC GENE.

═══════════════════════════════════════════════════════════════════════════════
GENE IDENTIFICATION (Critical - Read Carefully!)
═══════════════════════════════════════════════════════════════════════════════
Gene Symbol: {gene}
Official Name: {official_name}
Known Aliases: {aliases_str}

⚠️ DISAMBIGUATION REQUIREMENT:
You MUST verify each paper is actually discussing the GENE "{gene}" ({official_name}),
NOT just coincidentally mentioning the symbol or alias in a different context.

Common false positives to EXCLUDE (score = 1):
- "{gene}" used as an abbreviation for something else (e.g., CAT = CT scan, ACE = ACE inhibitor drug class)
- "{gene}" mentioned only in passing without discussing the gene itself
- Papers about drugs/inhibitors targeting {gene} but not studying the gene's biology
- Papers where "{gene}" appears in author names, institution names, or unrelated acronyms

Papers to INCLUDE (score ≥ 3):
- Studies investigating {gene}'s function, expression, mutations, or regulation
- Papers about {gene}'s role in disease, pathways, or cellular processes
- Research on {gene} protein structure, interactions, or mechanism
- Clinical studies examining {gene} mutations or biomarker potential

═══════════════════════════════════════════════════════════════════════════════
RESEARCH QUESTION: {query}
CONTEXT: {context_str}
═══════════════════════════════════════════════════════════════════════════════

Here are {len(papers_formatted)} candidate papers:

{papers_text}

═══════════════════════════════════════════════════════════════════════════════
SCORING CRITERIA
═══════════════════════════════════════════════════════════════════════════════
Score 5: Directly studies the {gene} GENE in this exact context with experimental data
Score 4: Studies the {gene} GENE in similar/related context with solid data
Score 3: Studies the {gene} GENE but different context, OR relevant context with moderate {gene} discussion
Score 2: Brief but genuine mention of the {gene} GENE (not just the acronym)
Score 1: NOT RELEVANT - either barely mentions {gene}, OR uses "{gene}" to mean something OTHER than the gene
         (e.g., imaging technique, drug class, unrelated abbreviation)

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════
For each paper, provide:
- relevance_score: 1-5 based on criteria above
- key_finding: Extract the main finding about {gene} from this paper (1-2 sentences). If the paper only mentions {gene} in passing, describe what aspect it covers.
- is_false_positive: true if the paper uses "{gene}" to mean something OTHER than the gene

Return JSON with ALL {len(papers_formatted)} papers:
{{
  "ranked_papers": [
    {{
      "index": 0,
      "pmid": "12345",
      "relevance_score": 5,
      "key_finding": "Main finding about {gene} gene (1-2 sentences)",
      "is_false_positive": false
    }},
    {{
      "index": 1,
      "pmid": "67890",
      "relevance_score": 1,
      "key_finding": "Not about the gene - discusses CAT scans",
      "is_false_positive": true
    }},
    ...
  ]
}}

Order by relevance_score (highest first), then by year (newest first) for ties.
Return ONLY valid JSON, no other text.
"""

        try:
            response = await client.chat.completions.create(
                model=get_active_model(),
                messages=[
                    {"role": "system", "content": "You are a scientific literature expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            _accumulate_tokens(state, self.__class__.__name__, getattr(response, 'usage', None))

            # Parse response
            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            # Match back to original papers
            ranked_papers = []
            false_positive_count = 0

            for item in result.get('ranked_papers', []):
                idx = item.get('index', -1)
                if idx < 0 or idx >= len(papers):
                    continue

                paper = papers[idx].copy()
                paper['relevance_score'] = item.get('relevance_score', 3)
                paper['key_finding'] = item.get('key_finding', '')
                paper['relevance_explanation'] = item.get('relevance_explanation', '')
                paper['is_false_positive'] = item.get('is_false_positive', False)

                if paper['is_false_positive']:
                    false_positive_count += 1

                ranked_papers.append(paper)

            if false_positive_count > 0:
                logger.info(f"LLM identified {false_positive_count} false positives for {gene}")

            # Sort by relevance score (desc), then year (desc)
            ranked_papers.sort(
                key=lambda p: (p.get('relevance_score', 0), _safe_year(p)),
                reverse=True
            )

            return ranked_papers

        except Exception as e:
            logger.error(f"LLM ranking failed for {gene}: {e}")
            # Fallback: return papers sorted by year
            return sorted(
                papers,
                key=_safe_year,
                reverse=True
            )

    def _format_papers_for_prompt(self, papers: List[Dict[str, Any]]) -> str:
        """Format papers section for prompt."""
        text = ""
        for p in papers:
            text += f"\nPaper {p['index']}:\n"
            text += f"PMID: {p['pmid']}\n"
            text += f"Title: {p['title']}\n"
            text += f"Journal: {p['journal']} ({p['year']})\n"
            text += f"Abstract: {p['abstract']}\n"
            text += f"{'─'*60}\n"
        return text
