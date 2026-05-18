"""
Adaptive Literature Fetcher
Tries tiered queries until finding optimal number of papers, accumulates across tiers if needed.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Any
from dataclasses import dataclass
import time
from pydantic_graph.nodes import BaseNode, GraphRunContext
from src.graph.state import GeneState
from src.integrations.pubmed_client import PubMedClient
from src.utils.adaptive_literature_queries import (
    AdaptiveLiteratureQueryBuilder,
    get_gene_info_from_db,
    build_cascading_queries,
    CascadingTier,
)
from src.utils.bm25_preranker import prerank_papers, build_context_query_terms
from src.config import RETRIEVAL_MODE, CASCADING_TARGET_POOL

if TYPE_CHECKING:
    from src.nodes.rank_papers_by_relevance import RankPapersByRelevance


# BM25 noise words - generic terms that match thousands of papers without specificity
BM25_NOISE_WORDS = {
    # Generic protein descriptor words
    'binding', 'protein', 'proteins', 'factor', 'factors',
    'receptor', 'receptors', 'kinase', 'kinases',
    'domain', 'domains', 'associated', 'related',
    'containing', 'like', 'family', 'member',
    'subunit', 'component', 'complex', 'enzyme',
    'activator', 'inhibitor', 'regulator', 'modifier',
    'transcript', 'isoform', 'variant', 'homolog',
    # Common filler words
    'type', 'class', 'group', 'alpha', 'beta', 'gamma', 'delta',
    'the', 'and', 'for', 'with', 'from',
}


def _is_valid_bm25_term(term: str) -> bool:
    """Check if a term is valid for BM25 ranking (not noise)."""
    if not term:
        return False
    term_clean = term.strip().lower()
    # Filter single letters or numbers
    if len(term_clean) < 4:
        return False
    # Filter if it's a pure number
    if term_clean.isdigit():
        return False
    # Filter noise words
    if term_clean in BM25_NOISE_WORDS:
        return False
    return True


def _build_gene_specific_bm25_terms(gene_symbol: str, base_context_terms: list, state) -> list:
    """
    Build BM25 context terms including gene-specific aliases.

    This ensures papers using common aliases (e.g., "p53" for TP53)
    are not incorrectly filtered out by BM25.

    Filters out:
    - Generic protein descriptor words (binding, protein, factor, etc.)
    - Single letters or numbers
    - Words under 4 characters

    Args:
        gene_symbol: The gene symbol (e.g., "TP53")
        base_context_terms: Base context terms from LLM extraction
        state: Pipeline state with gene data

    Returns:
        List of context terms including gene symbol, aliases, and name tokens
    """
    gene_terms = set()

    # Always include the gene symbol itself (exempt from length filter)
    gene_terms.add(gene_symbol)
    gene_terms.add(gene_symbol.lower())

    # Try to get aliases from state.all_gene_data (already fetched)
    gene_data = state.all_gene_data.get(gene_symbol, {})

    # Add aliases from gene data
    # Aliases can be: list of strings, list of dicts with 'symbol_alias', or nested in profile
    aliases = gene_data.get('aliases', [])
    if aliases:
        for alias in aliases:
            # Handle dict format: {'symbol_alias': 'p53', 'source': 'NCBI'}
            if isinstance(alias, dict):
                alias_str = alias.get('symbol_alias', '') or alias.get('alias', '')
            else:
                alias_str = str(alias)

            # Aliases are exempt from length filter (e.g., "p53" is only 3 chars but important)
            if alias_str and len(alias_str) >= 2:
                if alias_str.lower() not in BM25_NOISE_WORDS:
                    gene_terms.add(alias_str)
                    gene_terms.add(alias_str.lower())

    # Add tokens from official gene name - WITH NOISE FILTERING
    # e.g., "GATA binding protein 4" -> keep "GATA", filter out "binding", "protein"
    full_name = gene_data.get('full_name', '') or gene_data.get('name', '')
    if full_name:
        name_tokens = full_name.lower().split()
        for token in name_tokens:
            # Clean punctuation
            token = ''.join(c for c in token if c.isalnum())
            # Apply noise filter
            if _is_valid_bm25_term(token):
                gene_terms.add(token)

    # Try database lookup for additional aliases
    try:
        gene_info = get_gene_info_from_db(gene_symbol)
        if gene_info:
            # Add database aliases (exempt from length filter)
            for alias in gene_info.aliases:
                if alias and len(alias) >= 2:
                    if alias.lower() not in BM25_NOISE_WORDS:
                        gene_terms.add(alias)
                        gene_terms.add(alias.lower())
            # Add official name tokens - WITH NOISE FILTERING
            if gene_info.official_name:
                for token in gene_info.official_name.lower().split():
                    token = ''.join(c for c in token if c.isalnum())
                    if _is_valid_bm25_term(token):
                        gene_terms.add(token)
    except Exception:
        pass  # Database lookup is optional enhancement

    # Combine with base context terms (deduplicated, with noise filtering)
    all_terms = list(gene_terms)
    for term in base_context_terms:
        term_lower = term.lower()
        if term_lower not in {t.lower() for t in all_terms}:
            # Apply noise filter to base context terms too
            if _is_valid_bm25_term(term):
                all_terms.append(term)

    return all_terms


@dataclass
class FetchAdaptiveLiterature(BaseNode[GeneState]):
    """
    Adaptive literature fetching with tiered fallback.

    Strategy:
    1. Try tiers from specific to broad
    2. Stop when finding 10-200 papers
    3. If all tiers return <10, accumulate across tiers
    4. If tier returns >200, try agent refinement
    5. Apply BM25 pre-ranking to filter candidates before LLM ranking
    6. Fetch candidates for LLM ranking
    """

    target_papers: int = 10  # Final number of papers to select
    min_candidates: int = 30  # Minimum candidates for good ranking
    bm25_top_n: int = 20  # Number of papers to keep after BM25 pre-ranking

    async def _fetch_cascading(
        self,
        gene: str,
        state: GeneState,
        client: PubMedClient,
        gene_context_terms: list,
    ) -> Dict[str, Any]:
        """
        Fetch papers using cascading specificity search.

        Starts with most specific query (all dimensions), progressively
        broadens until reaching target pool size (~200 papers).

        Args:
            gene: Gene symbol
            state: Pipeline state
            client: PubMed client
            gene_context_terms: Terms for BM25 ranking

        Returns:
            Dict with candidate papers and metadata
        """
        # Extract context dimensions from state
        context_terms = state.literature_context_terms or {}
        experiment_ctx = state.experiment_context

        # Known population terms (for filtering from tissue candidates)
        population_keywords = {
            'paediatric', 'pediatric', 'child', 'children', 'infant', 'infants',
            'neonatal', 'neonate', 'neonates', 'newborn', 'newborns',
            'adult', 'adults', 'elderly', 'geriatric', 'aged',
            'women', 'woman', 'men', 'man', 'male', 'males', 'female', 'females',
            'pregnant', 'pregnancy', 'postmenopausal', 'premenopausal',
        }

        # Get ALL disease synonyms from context (not just first one)
        diseases = list(context_terms.get('diseases', []))
        if not diseases and experiment_ctx and hasattr(experiment_ctx, 'disease') and experiment_ctx.disease:
            diseases = [experiment_ctx.disease]

        # Get ALL phenotype terms from context
        phenotypes = list(context_terms.get('phenotypes', []))

        # Get ALL tissue synonyms from context
        tissues = list(context_terms.get('tissues', []))
        if not tissues:
            # Fallback: check 'cell_types' but filter out population terms
            cell_types = context_terms.get('cell_types', [])
            tissues = [ct for ct in cell_types if ct.lower() not in population_keywords]
        # Also add from experiment context if available
        if experiment_ctx and hasattr(experiment_ctx, 'tissue') and experiment_ctx.tissue:
            exp_tissue = experiment_ctx.tissue
            if exp_tissue not in tissues:
                tissues.append(exp_tissue)

        # Get ALL population terms from context
        populations = list(context_terms.get('populations', []))
        if not populations:
            # Fallback: search in cell_types, modifiers for population terms
            for field in ['cell_types', 'modifiers']:
                for term in context_terms.get(field, []):
                    if term.lower() in population_keywords and term not in populations:
                        populations.append(term)

        # Get gene info for disambiguation
        gene_info = get_gene_info_from_db(gene)

        # Build cascading tiers with full synonym lists
        tiers = build_cascading_queries(
            gene=gene,
            diseases=diseases,
            phenotypes=phenotypes,
            tissues=tissues,
            populations=populations,
            gene_info=gene_info,
        )

        print(f"  Cascading search with {len(tiers)} tiers")
        print(f"    Diseases: {diseases or '(none)'}")
        print(f"    Phenotypes: {phenotypes or '(none)'}")
        print(f"    Tissues: {tissues or '(none)'}")
        print(f"    Populations: {populations or '(none)'}")

        # Accumulate papers across tiers until target reached
        all_papers = []
        seen_pmids = set()
        tier_contributions = []
        queries_used = []
        executed_tier_numbers = set()
        stop_reason = None  # Track why we stopped

        for tier in tiers:
            print(f"\n  Tier {tier.tier_number}: {tier.description}")
            query_display = tier.query[:100] + "..." if len(tier.query) > 100 else tier.query
            print(f"    Query: {query_display}")

            try:
                executed_tier_numbers.add(tier.tier_number)
                count = await client.count(tier.query)
                print(f"    PubMed count: {count:,}")

                if count == 0:
                    tier_contributions.append({
                        'tier': tier.tier_number,
                        'name': tier.name,
                        'description': tier.description,
                        'query': tier.query,
                        'count': 0,
                        'contributed': 0,
                        'status': 'executed_zero',
                    })
                    continue

                # Fetch papers (date sorted for diversity)
                fetch_count = min(count, CASCADING_TARGET_POOL)
                pmids = await client.search_pmids(tier.query, retmax=fetch_count, sort="pub_date")
                papers = await client.fetch_details(pmids)

                # Deduplicate and tag with tier information
                new_papers = []
                for paper in papers:
                    pmid = paper.get('pmid')
                    if pmid and pmid not in seen_pmids:
                        seen_pmids.add(pmid)
                        # Attach tier information for ranking transparency
                        paper['tier_fetched'] = f"T{tier.tier_number}"
                        paper['tier_name'] = tier.name
                        paper['tier_description'] = tier.description
                        new_papers.append(paper)

                all_papers.extend(new_papers)
                queries_used.append(f"Tier {tier.tier_number} ({tier.description}): {tier.query}")

                # Determine status based on stopping conditions
                tier_status = 'executed'
                if len(all_papers) >= CASCADING_TARGET_POOL:
                    tier_status = 'threshold_met_pool'
                    stop_reason = 'pool_threshold'
                elif count >= CASCADING_TARGET_POOL:
                    tier_status = 'threshold_met_tier'
                    stop_reason = 'tier_sufficient'

                tier_contributions.append({
                    'tier': tier.tier_number,
                    'name': tier.name,
                    'description': tier.description,
                    'query': tier.query,
                    'count': count,
                    'contributed': len(new_papers),
                    'status': tier_status,
                })

                print(f"    Contributed: {len(new_papers)} new papers (total: {len(all_papers)})")

                # Stop if we've reached target pool OR if this single tier has enough papers
                # The second condition prevents unnecessary broader tiers when a specific tier
                # already provides sufficient coverage (e.g., Tier 3 has 475 papers, we don't
                # need Tier 4's even broader results)
                if len(all_papers) >= CASCADING_TARGET_POOL:
                    print(f"    Reached target pool ({CASCADING_TARGET_POOL}), stopping")
                    break
                if count >= CASCADING_TARGET_POOL:
                    print(f"    Tier alone has {count} papers (>= {CASCADING_TARGET_POOL}), stopping cascade")
                    break

            except Exception as e:
                print(f"    Error ({type(e).__name__}): {e!r}")
                tier_contributions.append({
                    'tier': tier.tier_number,
                    'name': tier.name,
                    'description': tier.description,
                    'query': tier.query,
                    'count': 0,
                    'contributed': 0,
                    'status': 'error',
                    'error': str(e),
                })
                continue

        # Add skipped tiers (tiers that were never executed due to early stopping)
        for tier in tiers:
            if tier.tier_number not in executed_tier_numbers:
                tier_contributions.append({
                    'tier': tier.tier_number,
                    'name': tier.name,
                    'description': tier.description,
                    'query': tier.query,
                    'count': None,
                    'contributed': 0,
                    'status': 'skipped',
                })

        # Sort tier_contributions by tier number for consistent display
        tier_contributions.sort(key=lambda x: x['tier'])

        # Apply BM25 pre-ranking
        papers_before_bm25 = len(all_papers)
        if gene_context_terms and len(all_papers) > self.bm25_top_n:
            all_papers = prerank_papers(
                papers=all_papers,
                context_terms=gene_context_terms,
                gene_symbol=gene,
                top_n=self.bm25_top_n,
                abstract_key="abstract"
            )
            print(f"\n  BM25 pre-ranked: {papers_before_bm25} → {len(all_papers)} papers")

        return {
            'tier_used': 'cascading',
            'tier_contributions': tier_contributions,
            'total_count': papers_before_bm25,
            'candidate_pmids': [p.get('pmid') for p in all_papers],
            'candidate_papers': all_papers,
            'query_used': '\n'.join(queries_used),
            'fetch_strategy': 'cascading',
            'bm25_applied': len(all_papers) < papers_before_bm25,
            'papers_before_bm25': papers_before_bm25,
            'stop_reason': stop_reason,
            'dimensions': {
                'diseases': diseases,
                'phenotypes': phenotypes,
                'tissues': tissues,
                'populations': populations,
            },
        }

    async def run(self, ctx: GraphRunContext[GeneState]) -> "RankPapersByRelevance":
        _t0 = time.perf_counter()
        state = ctx.state
        try:
            from src.nodes.rank_papers_by_relevance import RankPapersByRelevance
    
            print(f"\n{'='*70}")
            print("NODE: Fetch Adaptive Literature")
            print(f"Strategy: Tiered search with accumulation (mode: {RETRIEVAL_MODE})")
            print(f"{'='*70}")
    
            client = PubMedClient()
    
            # Build base context terms for BM25 pre-ranking (from Node 6 extraction)
            base_context_terms = build_context_query_terms(
                state.literature_context_terms or {},
                state.user_query or ""
            )
            print(f"BM25 base context terms: {base_context_terms[:5]}{'...' if len(base_context_terms) > 5 else ''}")
    
            genes = state.get_genes_found()
            print(f"\nProcessing {len(genes)} genes...\n")
    
            for gene in genes:
                print(f"{'='*60}")
                print(f"GENE: {gene}")
                print(f"{'='*60}")
    
                # Build gene-specific BM25 terms including aliases
                gene_context_terms = _build_gene_specific_bm25_terms(gene, base_context_terms, state)
                print(f"  BM25 terms for {gene}: {gene_context_terms[:8]}{'...' if len(gene_context_terms) > 8 else ''}")
    
                # ═══════════════════════════════════════════════════════════════
                # CASCADING MODE: Use cascading specificity search
                # ═══════════════════════════════════════════════════════════════
                if RETRIEVAL_MODE == "cascading":
                    result = await self._fetch_cascading(
                        gene=gene,
                        state=state,
                        client=client,
                        gene_context_terms=gene_context_terms,
                    )
                    state.gene_literature_candidates[gene] = result
                    continue  # Skip the original tier-based logic
    
                # ═══════════════════════════════════════════════════════════════
                # ORIGINAL MODE: best_match or date_pool
                # ═══════════════════════════════════════════════════════════════
                tiers = state.literature_query_tiers.get(gene, [])
    
                if not tiers:
                    print(f"  No query tiers available, skipping")
                    state.gene_literature_candidates[gene] = {
                        'tier_used': 'none',
                        'total_count': 0,
                        'candidate_pmids': [],
                        'candidate_papers': [],
                        'query_used': '',
                        'fetch_strategy': 'skipped'
                    }
                    continue
    
                # Try to find optimal tier
                all_candidates = []
                queries_used = []  # Track queries for transparency
                found_optimal = False
    
                for tier in tiers:
                    print(f"\n  Tier {tier.tier_number}: {tier.description}")
                    query_display = tier.query[:100] + "..." if len(tier.query) > 100 else tier.query
                    print(f"    Query: {query_display}")
    
                    try:
                        # Get count
                        count = await client.count(tier.query)
                        print(f"    Result: {count:,} papers")
    
                        # CASE 1: Acceptable range (>= 10 papers, no upper limit)
                        # We accept any count >= 10 because:
                        # - We only fetch top 50 for LLM ranking anyway
                        # - PubMed returns papers sorted by relevance
                        # - Well-studied genes with disease context can return 10,000+ papers
                        #   (e.g., BRCA1 + breast cancer = 12,153 papers)
                        # - The top 50 by relevance are still highly specific
                        # - Upper limits were causing disease-context queries to be rejected
                        if count >= 10:
                            print(f"    In acceptable range! Fetching papers...")
    
                            # Fetch parameters depend on retrieval mode
                            if RETRIEVAL_MODE == "date_pool":
                                # Date-sorted with larger pool for BM25/LLM to filter
                                fetch_count = min(count, 200)
                                sort_order = "pub_date"
                            else:
                                # Default: Best Match sorting with smaller pool
                                fetch_count = min(count, 50)
                                sort_order = "relevance"
    
                            pmids = await client.search_pmids(tier.query, retmax=fetch_count, sort=sort_order)
                            papers = await client.fetch_details(pmids)
    
                            print(f"    Fetched {len(papers)} candidate papers")
    
                            # Apply BM25 pre-ranking to filter to most relevant papers
                            papers_before_bm25 = len(papers)
                            if gene_context_terms and len(papers) > self.bm25_top_n:
                                papers = prerank_papers(
                                    papers=papers,
                                    context_terms=gene_context_terms,
                                    gene_symbol=gene,
                                    top_n=self.bm25_top_n,
                                    abstract_key="abstract"
                                )
                                print(f"    BM25 pre-ranked: {papers_before_bm25} → {len(papers)} papers")
    
                            # Store results
                            state.gene_literature_candidates[gene] = {
                                'tier_used': tier.name,
                                'tier_number': tier.tier_number,
                                'total_count': count,
                                'candidate_pmids': [p.get('pmid') for p in papers],
                                'candidate_papers': papers,
                                'query_used': tier.query,
                                'fetch_strategy': 'single_tier',
                                'bm25_applied': len(papers) < papers_before_bm25,
                                'papers_before_bm25': papers_before_bm25,
                            }
                            found_optimal = True
                            break  # Success! Stop trying tiers
    
                        # CASE 2: Too few papers (<10)
                        elif count < 10:
                            print(f"    Only {count} papers, accumulating...")
    
                            # Fetch all papers from this tier
                            if count > 0:
                                sort_order = "pub_date" if RETRIEVAL_MODE == "date_pool" else "relevance"
                                pmids = await client.search_pmids(tier.query, retmax=count, sort=sort_order)
                                papers = await client.fetch_details(pmids)
    
                                # Deduplicate by PMID
                                existing_pmids = {p.get('pmid') for p in all_candidates}
                                new_papers = [p for p in papers if p.get('pmid') not in existing_pmids]
                                all_candidates.extend(new_papers)
                                queries_used.append(f"Tier {tier.tier_number}: {tier.query}")
    
                                print(f"    Added {len(new_papers)} new papers ({len(all_candidates)} total)")
    
                            # If we have enough candidates now, stop
                            if len(all_candidates) >= self.min_candidates:
                                print(f"    Reached {len(all_candidates)} candidates through accumulation")
    
                                # Apply BM25 pre-ranking
                                papers_before_bm25 = len(all_candidates)
                                if gene_context_terms and len(all_candidates) > self.bm25_top_n:
                                    all_candidates = prerank_papers(
                                        papers=all_candidates,
                                        context_terms=gene_context_terms,
                                        gene_symbol=gene,
                                        top_n=self.bm25_top_n,
                                        abstract_key="abstract"
                                    )
                                    print(f"    BM25 pre-ranked: {papers_before_bm25} → {len(all_candidates)} papers")
    
                                state.gene_literature_candidates[gene] = {
                                    'tier_used': 'accumulated',
                                    'tiers_combined': [t.tier_number for t in tiers[:tiers.index(tier)+1]],
                                    'total_count': papers_before_bm25,
                                    'candidate_pmids': [p.get('pmid') for p in all_candidates],
                                    'candidate_papers': all_candidates,
                                    'query_used': '\n'.join(queries_used) if queries_used else 'accumulated from multiple tiers',
                                    'fetch_strategy': 'accumulation',
                                    'bm25_applied': len(all_candidates) < papers_before_bm25,
                                    'papers_before_bm25': papers_before_bm25,
                                }
                                found_optimal = True
                                break
    
                            # Continue to next tier to accumulate more
                            continue
    
                        # Note: No upper limit on paper count - we accept any count >= 10
                        # since we only fetch top 50 papers anyway (sorted by PubMed relevance)
    
                    except Exception as e:
                        print(f"    Error: {e}")
                        continue
    
                # If no optimal tier found, use accumulated papers or last resort
                if not found_optimal:
                    if all_candidates:
                        print(f"\n  Using {len(all_candidates)} accumulated papers as fallback")
    
                        # Apply BM25 pre-ranking
                        papers_before_bm25 = len(all_candidates)
                        if gene_context_terms and len(all_candidates) > self.bm25_top_n:
                            all_candidates = prerank_papers(
                                papers=all_candidates,
                                context_terms=gene_context_terms,
                                gene_symbol=gene,
                                top_n=self.bm25_top_n,
                                abstract_key="abstract"
                            )
                            print(f"    BM25 pre-ranked: {papers_before_bm25} → {len(all_candidates)} papers")
    
                        state.gene_literature_candidates[gene] = {
                            'tier_used': 'fallback_accumulation',
                            'total_count': papers_before_bm25,
                            'candidate_pmids': [p.get('pmid') for p in all_candidates],
                            'candidate_papers': all_candidates,
                            'query_used': '\n'.join(queries_used) if queries_used else 'accumulated from multiple tiers',
                            'fetch_strategy': 'fallback',
                            'bm25_applied': len(all_candidates) < papers_before_bm25,
                            'papers_before_bm25': papers_before_bm25,
                        }
                    else:
                        # Last resort: fetch from gene-only tier with improved disambiguation
                        print(f"\n  Falling back to gene-only search...")
    
                        # Build improved gene search term using query builder
                        builder = AdaptiveLiteratureQueryBuilder()
                        gene_search = builder._build_gene_search_term(None, gene)
                        fallback_query = f'{gene_search} AND "Humans"[MeSH Terms]'
    
                        try:
                            # Use retrieval mode settings for fallback too
                            if RETRIEVAL_MODE == "date_pool":
                                fallback_retmax = 200
                                fallback_sort = "pub_date"
                            else:
                                fallback_retmax = 50
                                fallback_sort = "relevance"
    
                            pmids = await client.search_pmids(fallback_query, retmax=fallback_retmax, sort=fallback_sort)
                            papers = await client.fetch_details(pmids)
    
                            print(f"  Fetched {len(papers)} papers from fallback")
    
                            # Apply BM25 pre-ranking
                            papers_before_bm25 = len(papers)
                            if gene_context_terms and len(papers) > self.bm25_top_n:
                                papers = prerank_papers(
                                    papers=papers,
                                    context_terms=gene_context_terms,
                                    gene_symbol=gene,
                                    top_n=self.bm25_top_n,
                                    abstract_key="abstract"
                                )
                                print(f"    BM25 pre-ranked: {papers_before_bm25} → {len(papers)} papers")
    
                            state.gene_literature_candidates[gene] = {
                                'tier_used': 'tier5_gene_only',
                                'tier_number': 5,
                                'total_count': papers_before_bm25,
                                'candidate_pmids': [p.get('pmid') for p in papers],
                                'candidate_papers': papers,
                                'query_used': fallback_query,
                                'fetch_strategy': 'last_resort',
                                'bm25_applied': len(papers) < papers_before_bm25,
                                'papers_before_bm25': papers_before_bm25,
                            }
                        except Exception as e:
                            print(f"  Fallback also failed: {e}")
                            state.gene_literature_candidates[gene] = {
                                'tier_used': 'failed',
                                'total_count': 0,
                                'candidate_pmids': [],
                                'candidate_papers': [],
                                'query_used': fallback_query,
                                'fetch_strategy': 'failed',
                                'error': str(e)
                            }
    
            # Summary
            print(f"\n{'='*70}")
            print("FETCH SUMMARY")
            print(f"{'='*70}")
    
            for gene in genes:
                if gene in state.gene_literature_candidates:
                    info = state.gene_literature_candidates[gene]
                    papers_count = len(info.get('candidate_papers', []))
                    strategy = info.get('fetch_strategy', 'unknown')
                    bm25_applied = info.get('bm25_applied', False)
                    papers_before = info.get('papers_before_bm25', papers_count)
    
                    if strategy == 'cascading':
                        # Show tier contributions for cascading mode
                        tier_contribs = info.get('tier_contributions', [])
                        contrib_summary = ", ".join([
                            f"T{tc['tier']}:{tc['contributed']}"
                            for tc in tier_contribs
                            if tc.get('contributed', 0) > 0
                        ])
                        print(f"  {gene}: {papers_count} candidates (cascading: {contrib_summary})")
                        if bm25_applied:
                            print(f"         BM25: {papers_before}→{papers_count}")
                    elif bm25_applied:
                        print(f"  {gene}: {papers_count} candidates ({strategy}, BM25: {papers_before}→{papers_count})")
                    else:
                        print(f"  {gene}: {papers_count} candidates ({strategy})")
                else:
                    print(f"  {gene}: No papers found")
    
            print(f"{'='*70}\n")

            # ═══════════════════════════════════════════════════════════════════
            # POPULATE LITERATURE RETRIEVAL STATS FOR THESIS BENCHMARKING
            # ═══════════════════════════════════════════════════════════════════
            for gene in genes:
                if gene in state.gene_literature_candidates:
                    info = state.gene_literature_candidates[gene]
                    papers_count = len(info.get('candidate_papers', []))
                    bm25_applied = info.get('bm25_applied', False)
                    papers_before = info.get('papers_before_bm25', papers_count)

                    # Initialize stats for this gene
                    state.literature_retrieval_stats[gene] = {
                        'candidates_before_bm25': papers_before,
                        'candidates_after_bm25': papers_count if bm25_applied else papers_before,
                        'bm25_applied': bm25_applied,
                        'final_papers_count': 0,  # Will be updated by RankPapersByRelevance
                        'source_paper_found': False,  # Will be checked after ranking
                        'source_paper_pmid': None
                    }

            return RankPapersByRelevance()
        finally:
            state.log_node_execution(self.__class__.__name__, round(time.perf_counter() - _t0, 3))
