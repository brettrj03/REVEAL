"""
Adaptive Literature Query Builder with Tiered Fallback System
Builds multiple query tiers from specific to broad, tries them until finding optimal results.

Enhanced to avoid acronym collisions by using:
- Gene names from database (e.g., "lactase" for LCT)
- NCBI Gene Name field tags
- Biological context filters for ambiguous symbols
"""

from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import os
import sqlite3


class QueryTier(BaseModel):
    """Represents one tier in the fallback strategy"""
    tier_number: int
    name: str
    query: str
    expected_range: Tuple[int, int]  # (min_papers, max_papers)
    description: str


@dataclass
class GeneInfo:
    """Gene metadata for building better queries"""
    symbol: str
    gene_id: str
    official_name: str  # e.g., "lactase" for LCT
    aliases: List[str]

    @property
    def is_short_symbol(self) -> bool:
        """Symbols <4 chars are more likely to have acronym collisions"""
        return len(self.symbol) <= 3

    @property
    def needs_disambiguation(self) -> bool:
        """Check if this gene symbol likely needs disambiguation"""
        # Short symbols or common acronyms need extra filtering
        ambiguous_patterns = ['LCT', 'ACE', 'CAT', 'SOD', 'MET', 'KIT', 'RET', 'ALK']
        return self.is_short_symbol or self.symbol in ambiguous_patterns


@dataclass
class CascadingTier:
    """Represents one tier in the cascading specificity search"""
    tier_number: int
    name: str
    query: str
    dimensions_used: List[str]  # e.g., ['gene', 'disease', 'tissue']
    description: str


# Noise terms to exclude from queries - these add no specificity
NOISE_TERMS = {
    'genetic', 'genetics', 'genomic', 'genomics',
    'precision', 'personalized', 'personalised',
    'molecular', 'variants', 'variant', 'mutation', 'mutations',
    'biomarker', 'biomarkers', 'marker', 'markers',
    'diagnosis', 'diagnostic', 'prognosis', 'prognostic',
    'treatment', 'therapy', 'therapeutic',
    'patient', 'patients', 'clinical', 'study', 'studies',
    'analysis', 'research', 'investigation',
}


def _clean_term(term: str) -> Optional[str]:
    """Clean and validate a search term, returning None if it's noise."""
    if not term:
        return None
    term = term.strip()
    if term.lower() in NOISE_TERMS:
        return None
    if len(term) < 2:
        return None
    return term


def _format_term_for_pubmed(term: str) -> str:
    """Format a term for PubMed [Title/Abstract] search."""
    term = term.strip()
    # Quote multi-word terms
    if ' ' in term:
        return f'"{term}"[Title/Abstract]'
    return f'{term}[Title/Abstract]'


def _build_or_clause(terms: List[str], field_tag: bool = True) -> Optional[str]:
    """
    Build an OR clause from a list of terms.

    Args:
        terms: List of search terms
        field_tag: If True, add [Title/Abstract] tag to each term

    Returns:
        OR clause string or None if no valid terms
    """
    clean_terms = [_clean_term(t) for t in terms if t]
    clean_terms = [t for t in clean_terms if t]  # Remove None values

    if not clean_terms:
        return None

    if field_tag:
        formatted = [_format_term_for_pubmed(t) for t in clean_terms]
    else:
        formatted = clean_terms

    if len(formatted) == 1:
        return formatted[0]
    return f'({" OR ".join(formatted)})'


def build_cascading_queries(
    gene: str,
    diseases: Optional[List[str]] = None,
    phenotypes: Optional[List[str]] = None,
    tissues: Optional[List[str]] = None,
    populations: Optional[List[str]] = None,
    gene_info: Optional[GeneInfo] = None,
    # Legacy single-term parameters for backward compatibility
    disease: Optional[str] = None,
    tissue: Optional[str] = None,
    population: Optional[str] = None,
) -> List[CascadingTier]:
    """
    Build cascading specificity queries from most specific to least specific.

    Each dimension is built as an OR group of synonyms. The OR groups stay constant
    across tiers - only the AND/OR relationship BETWEEN dimensions changes.

    Dimension hierarchy (in order of importance):
    1. Gene — always present, never drops
    2. Disease — named conditions (e.g., "congenital heart disease")
    3. Phenotype — observable symptoms (e.g., "neurodevelopmental delay")
    4. Tissue — anatomical location (e.g., "brain", "heart")
    5. Population — demographic (e.g., "pediatric", "female")

    Tier logic (with disease present):
    - Tier 1: Gene AND Disease AND Phenotype AND Tissue AND Population (all AND)
    - Tier 2: Gene AND Disease AND Phenotype AND (Tissue OR Population)
    - Tier 3: Gene AND (Disease OR Phenotype OR Tissue OR Population) (all OR)
    - Tier 4: Gene only (last resort)

    When disease is absent, phenotype steps up:
    - Tier 1: Gene AND Phenotype AND Tissue AND Population
    - Tier 2: Gene AND Phenotype AND (Tissue OR Population)
    - Tier 3: Gene AND (Phenotype OR Tissue OR Population)
    - Tier 4: Gene only

    Args:
        gene: Gene symbol (required)
        diseases: List of disease terms/synonyms (e.g., ["congenital heart disease", "CHD"])
        phenotypes: List of phenotype terms (e.g., ["neurodevelopmental delay", "intellectual disability"])
        tissues: List of tissue terms/synonyms (e.g., ["brain", "cerebral"])
        populations: List of population terms (e.g., ["pediatric", "child"])
        gene_info: Optional GeneInfo for disambiguation
        disease: Legacy single disease term (converted to list)
        tissue: Legacy single tissue term (converted to list)
        population: Legacy single population term (converted to list)

    Returns:
        List of CascadingTier objects ordered from most to least specific
    """
    tiers = []

    # Handle legacy single-term parameters by converting to lists
    if diseases is None and disease:
        diseases = [disease]
    if tissues is None and tissue:
        tissues = [tissue]
    if populations is None and population:
        populations = [population]

    # Ensure we have lists (not None)
    diseases = diseases or []
    phenotypes = phenotypes or []
    tissues = tissues or []
    populations = populations or []

    # Build gene search term with disambiguation
    if gene_info and gene_info.official_name and gene_info.official_name != gene:
        if gene_info.needs_disambiguation:
            gene_search = (
                f'({gene}[Title/Abstract] OR '
                f'"{gene_info.official_name}"[Title/Abstract] OR '
                f'({gene}[Title/Abstract] AND (gene[Title/Abstract] OR protein[Title/Abstract])))'
            )
        else:
            gene_search = (
                f'({gene}[Title/Abstract] OR '
                f'"{gene_info.official_name}"[Title/Abstract])'
            )
    else:
        # No gene info - use symbol with biological filter for short symbols
        if len(gene) <= 3:
            gene_search = (
                f'({gene}[Title/Abstract] AND '
                f'(gene[Title/Abstract] OR protein[Title/Abstract] OR expression[Title/Abstract]))'
            )
        else:
            gene_search = f'{gene}[Title/Abstract]'

    # Build OR clauses for each dimension (these stay constant across tiers)
    disease_clause = _build_or_clause(diseases)
    phenotype_clause = _build_or_clause(phenotypes)
    tissue_clause = _build_or_clause(tissues)
    population_clause = _build_or_clause(populations)

    # Track which dimensions are available
    has_disease = disease_clause is not None
    has_phenotype = phenotype_clause is not None
    has_tissue = tissue_clause is not None
    has_population = population_clause is not None

    # Determine the "anchor" dimension for Tier 2
    # Disease is primary anchor; if absent, phenotype steps up
    has_anchor = has_disease or has_phenotype
    anchor_clause = disease_clause if has_disease else phenotype_clause
    anchor_name = 'disease' if has_disease else 'phenotype'

    # Collect all available dimensions in priority order
    all_clauses = []
    all_names = []
    if has_disease:
        all_clauses.append(disease_clause)
        all_names.append('disease')
    if has_phenotype:
        all_clauses.append(phenotype_clause)
        all_names.append('phenotype')
    if has_tissue:
        all_clauses.append(tissue_clause)
        all_names.append('tissue')
    if has_population:
        all_clauses.append(population_clause)
        all_names.append('population')

    dimension_count = len(all_clauses)

    # ═══════════════════════════════════════════════════════════════════
    # TIER 1: Gene AND Disease AND Phenotype AND Tissue AND Population
    # (or without disease: Gene AND Phenotype AND Tissue AND Population)
    # Built whenever we have at least 2 context dimensions
    # ═══════════════════════════════════════════════════════════════════
    if dimension_count >= 2:
        and_clause = ' AND '.join(all_clauses)
        query = f'{gene_search} AND {and_clause}'

        tiers.append(CascadingTier(
            tier_number=1,
            name="tier1_all_and",
            query=query,
            dimensions_used=['gene'] + all_names,
            description=f"Gene AND {' AND '.join(all_names)}"
        ))

    # ═══════════════════════════════════════════════════════════════════
    # TIER 2: Gene AND Anchor AND (lower dimensions OR)
    # Anchor = Disease if present, else Phenotype
    # Lower dimensions = everything below anchor in hierarchy
    # ═══════════════════════════════════════════════════════════════════
    if has_anchor and (has_tissue or has_population):
        # Collect dimensions below the anchor for OR grouping
        lower_parts = []
        lower_names = []

        # If disease is anchor, phenotype joins the OR group with tissue/population
        if has_disease and has_phenotype:
            lower_parts.append(phenotype_clause)
            lower_names.append('phenotype')

        if has_tissue:
            lower_parts.append(tissue_clause)
            lower_names.append('tissue')
        if has_population:
            lower_parts.append(population_clause)
            lower_names.append('population')

        if lower_parts:
            if len(lower_parts) == 1:
                or_clause = lower_parts[0]
                desc = f"Gene AND {anchor_name.title()} AND {lower_names[0].title()}"
            else:
                or_clause = f'({" OR ".join(lower_parts)})'
                desc = f"Gene AND {anchor_name.title()} AND ({' OR '.join(lower_names)})"

            query = f'{gene_search} AND {anchor_clause} AND {or_clause}'

            tiers.append(CascadingTier(
                tier_number=2,
                name="tier2_anchor_and_lower_or",
                query=query,
                dimensions_used=['gene', anchor_name] + lower_names,
                description=desc
            ))

    # ═══════════════════════════════════════════════════════════════════
    # TIER 3: Gene AND (all dimensions OR)
    # All context as OR - broadest context search
    # ═══════════════════════════════════════════════════════════════════
    if dimension_count >= 1:
        if len(all_clauses) == 1:
            or_clause = all_clauses[0]
            desc = f"Gene AND {all_names[0]}"
        else:
            or_clause = f'({" OR ".join(all_clauses)})'
            desc = f"Gene AND ({' OR '.join(all_names)})"

        query = f'{gene_search} AND {or_clause}'

        tiers.append(CascadingTier(
            tier_number=3,
            name="tier3_all_or",
            query=query,
            dimensions_used=['gene'] + all_names,
            description=desc
        ))

    # ═══════════════════════════════════════════════════════════════════
    # TIER 4: Gene only (last resort)
    # ═══════════════════════════════════════════════════════════════════
    query = f'{gene_search} AND "Humans"[MeSH Terms]'
    tiers.append(CascadingTier(
        tier_number=4,
        name="tier4_gene_only",
        query=query,
        dimensions_used=['gene'],
        description="Gene only (human filter)"
    ))

    return tiers


def get_gene_info_from_db(symbol: str, db_path: str = None) -> Optional[GeneInfo]:
    """
    Fetch gene metadata from SQLite database.

    Args:
        symbol: Gene symbol (e.g., 'LCT')
        db_path: Path to the SQLite database

    Returns:
        GeneInfo object or None if gene not found
    """
    if db_path is None:
        from src.config import DEFAULT_DB_PATH
        db_path = DEFAULT_DB_PATH
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get gene_id and check if gene exists
        cursor.execute("SELECT gene_id FROM genes WHERE gene_symbol = ?", (symbol,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        gene_id = row[0]

        # Get official name from gene_function table
        cursor.execute(
            "SELECT description FROM gene_function WHERE gene_id = ?",
            (gene_id,)
        )
        row = cursor.fetchone()
        official_name = row[0] if row else symbol

        # Clean up official name (remove source annotations)
        if '[Source:' in official_name:
            official_name = official_name.split('[Source:')[0].strip()

        # Get aliases
        cursor.execute(
            "SELECT symbol_alias FROM gene_alias WHERE gene_id = ?",
            (gene_id,)
        )
        aliases = [row[0] for row in cursor.fetchall()]

        conn.close()

        return GeneInfo(
            symbol=symbol,
            gene_id=gene_id,
            official_name=official_name,
            aliases=aliases
        )

    except Exception as e:
        print(f"  Warning: Could not fetch gene info for {symbol}: {e}")
        return None


class AdaptiveLiteratureQueryBuilder:
    """Builds tiered queries that adapt to available context"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            from src.config import DEFAULT_DB_PATH
            db_path = DEFAULT_DB_PATH
        self.db_path = db_path

    def _build_gene_search_term(self, gene_info: Optional[GeneInfo], gene_symbol: str) -> str:
        """
        Build a robust gene search term that avoids acronym collisions.

        Strategy:
        1. Use [Title/Abstract] field tag for gene symbols
        2. Include official gene name in Title/Abstract search
        3. For ambiguous symbols, add biological context filter

        Args:
            gene_info: Gene metadata from database (may be None)
            gene_symbol: The gene symbol

        Returns:
            PubMed search term string
        """
        if gene_info and gene_info.official_name and gene_info.official_name != gene_symbol:
            # We have an official name different from the symbol
            official_name = gene_info.official_name

            # Build a robust query combining multiple approaches
            # 1. Gene symbol in title/abstract
            # 2. Official name in title/abstract (catches papers using full name)
            # 3. Symbol with gene/protein context (reduces false positives for ambiguous symbols)

            if gene_info.needs_disambiguation:
                # For ambiguous symbols, be more restrictive
                return (
                    f'({gene_symbol}[Title/Abstract] OR '
                    f'"{official_name}"[Title/Abstract] OR '
                    f'({gene_symbol}[Title/Abstract] AND (gene[Title/Abstract] OR protein[Title/Abstract] OR expression[Title/Abstract])))'
                )
            else:
                # For less ambiguous symbols, be more permissive
                return (
                    f'({gene_symbol}[Title/Abstract] OR '
                    f'"{official_name}"[Title/Abstract])'
                )
        else:
            # No official name available, use symbol with biological filter
            if len(gene_symbol) <= 3:
                # Short symbol - add biological context to reduce false positives
                return (
                    f'({gene_symbol}[Title/Abstract] AND '
                    f'(gene[Title/Abstract] OR protein[Title/Abstract] OR expression[Title/Abstract]))'
                )
            else:
                # Longer symbol - less likely to have collisions
                return f'{gene_symbol}[Title/Abstract]'

    def build_all_tiers(
        self,
        gene: str,
        context_terms: dict,
        user_query: str,
        gene_info: Optional[GeneInfo] = None,
        other_genes: Optional[List[str]] = None  # Kept for backward compatibility, but ignored
    ) -> List[QueryTier]:
        """
        Build all 5 query tiers from specific to broad.

        Enhanced to avoid acronym collisions by using gene names and biological filters.

        IMPORTANT: The target gene gets a full gene search block with [Title/Abstract] tags.
        Context genes (from context_terms['context_genes']) are added as simple
        [Title/Abstract] keywords only - they should NOT get full gene search blocks.

        Args:
            gene: Gene symbol (the primary gene for this search)
            context_terms: Dict with 'diseases', 'processes', 'cell_types', 'modifiers', 'context_genes'
            user_query: Original user query for extracting function keywords
            gene_info: Optional gene metadata from database (for better queries)
            other_genes: DEPRECATED - ignored, kept for backward compatibility

        Returns:
            List of QueryTier objects (only tiers that can be built)
        """
        tiers = []

        # If gene_info not provided, try to fetch from database
        if gene_info is None:
            gene_info = get_gene_info_from_db(gene, self.db_path)

        # Build the gene search term (handles disambiguation)
        # Only the TARGET gene gets the full gene search block with [Title/Abstract]
        gene_search = self._build_gene_search_term(gene_info, gene)

        # Log what we're using
        if gene_info and gene_info.official_name != gene:
            print(f"    Using gene name: {gene_info.official_name}")

        # Extract context components
        diseases = context_terms.get('diseases', [])
        processes = context_terms.get('processes', [])
        cell_types = context_terms.get('cell_types', [])

        # Context genes from user query (MYC, CIITA, etc.) - simple [Title/Abstract] keywords only
        # These are genes mentioned in the biological question, NOT co-queried genes from the gene list
        context_genes = context_terms.get('context_genes', [])
        # Filter out the target gene from context genes (it's already the main search term)
        context_genes = [g for g in context_genes if g.upper() != gene.upper()]

        context_genes_search = None
        if context_genes:
            # Simple [Title/Abstract] keywords only - no full gene search blocks
            context_genes_search = " OR ".join([f'{g}[Title/Abstract]' for g in context_genes])
            print(f"    Context genes (as keywords): {context_genes}")

        # ═══════════════════════════════════════════════════════════
        # TIER 1: Experimental context (disease/process/cell type)
        # ═══════════════════════════════════════════════════════════
        if diseases or processes or cell_types:
            clauses = [gene_search]
            desc_parts = []
            if diseases:
                disease_str = " OR ".join([f'"{d}"[Title/Abstract]' for d in diseases[:3]])
                clauses.append(f'({disease_str})')
                desc_parts.append("disease")
            if processes:
                process_str = " OR ".join([f'"{p}"[Title/Abstract]' for p in processes[:3]])
                clauses.append(f'({process_str})')
                desc_parts.append("process")
            if cell_types:
                cell_str = " OR ".join([f'"{c}"[Title/Abstract]' for c in cell_types[:3]])
                clauses.append(f'({cell_str})')
                desc_parts.append("cell type")

            query = " AND ".join(clauses)
            tiers.append(QueryTier(
                tier_number=1,
                name="tier1_gene_exp_context",
                query=query,
                expected_range=(10, 150),
                description=f"Gene + experimental context ({'/'.join(desc_parts)})"
            ))

        # ═══════════════════════════════════════════════════════════
        # TIER 2: Gene + disease + (process OR cell type)
        # ═══════════════════════════════════════════════════════════
        if diseases and (processes or cell_types):
            disease_str = " OR ".join([f'"{d}"[Title/Abstract]' for d in diseases[:3]])

            if processes:
                context_str = " OR ".join([f'"{p}"[Title/Abstract]' for p in processes[:3]])
                context_type = "process"
            else:
                context_str = " OR ".join([f'"{c}"[Title/Abstract]' for c in cell_types[:3]])
                context_type = "cell type"

            query = f'{gene_search} AND ({disease_str}) AND ({context_str})'

            tiers.append(QueryTier(
                tier_number=2,
                name="tier2_gene_disease_context",
                query=query,
                expected_range=(20, 300),
                description=f"Gene + disease + {context_type}"
            ))

        # ═══════════════════════════════════════════════════════════
        # TIER 3: Gene + any available context
        # ═══════════════════════════════════════════════════════════
        context_components = []
        desc_parts = []

        if diseases:
            context_components.extend([f'"{d}"[Title/Abstract]' for d in diseases[:3]])
            desc_parts.append("disease")
        if processes:
            context_components.extend([f'"{p}"[Title/Abstract]' for p in processes[:3]])
            desc_parts.append("process")
        if cell_types:
            context_components.extend([f'"{c}"[Title/Abstract]' for c in cell_types[:2]])
            desc_parts.append("cell type")
        modifiers = context_terms.get('modifiers') or []
        if modifiers:
            context_components.extend([f'"{m}"[Title/Abstract]' for m in modifiers[:2]])
            desc_parts.append("modifier")

        if context_components:
            context_str = " OR ".join(context_components)
            query = f'{gene_search} AND ({context_str})'

            tiers.append(QueryTier(
                tier_number=3,
                name="tier3_gene_any_context",
                query=query,
                expected_range=(50, 500),
                description=f"Gene + any context ({'/'.join(desc_parts)})"
            ))

        # ═══════════════════════════════════════════════════════════
        # TIER 4: Cross-gene co-mentions (fallback)
        # ═══════════════════════════════════════════════════════════
        if context_genes_search:
            query = f'{gene_search} AND ({context_genes_search})'

            tiers.append(QueryTier(
                tier_number=4,
                name="tier4_gene_context_genes",
                query=query,
                expected_range=(10, 300),
                description=f"Gene + context genes ({', '.join(context_genes)})"
            ))

        # ═══════════════════════════════════════════════════════════
        # TIER 5: Broad (gene + function keywords from query)
        # ═══════════════════════════════════════════════════════════
        function_keywords = self._extract_function_keywords(user_query, gene)

        if function_keywords:
            keyword_str = " OR ".join([f'"{k}"[Title/Abstract]' for k in function_keywords[:5]])
            query = f'{gene_search} AND ({keyword_str})'

            tiers.append(QueryTier(
                tier_number=5,
                name="tier5_gene_function_keywords",
                query=query,
                expected_range=(100, 1000),
                description="Gene + function keywords"
            ))

        # ═══════════════════════════════════════════════════════════
        # TIER 6: Gene Only (always works - fallback)
        # Uses the robust gene search with disambiguation
        # ═══════════════════════════════════════════════════════════
        query = f'{gene_search} AND "Humans"[MeSH Terms]'

        tiers.append(QueryTier(
            tier_number=6,
            name="tier6_gene_only",
            query=query,
            expected_range=(200, 50000),
            description="Gene name search with human filter (fallback)"
        ))

        return tiers

    def _extract_function_keywords(self, query: str, gene: str) -> List[str]:
        """
        Extract function-related keywords from user query.

        Args:
            query: User's original query
            gene: Gene symbol (to avoid matching gene name)

        Returns:
            List of function keywords found in query
        """
        keywords = []

        # Common biological process/function terms
        function_terms = [
            'function', 'role', 'activity', 'mechanism', 'pathway',
            'regulation', 'expression', 'signaling', 'metabolism',
            'development', 'differentiation', 'proliferation',
            'apoptosis', 'interaction', 'binding', 'transport',
            'synthesis', 'degradation', 'modification', 'localization',
            'cancer', 'tumor', 'metastasis', 'inflammation',
            'immune', 'cell cycle', 'death', 'survival'
        ]

        query_lower = query.lower()

        # Extract terms that appear in query
        for term in function_terms:
            if term in query_lower and term != gene.lower():
                keywords.append(term)

        return keywords


class AgentQueryRefiner:
    """Uses LLM to refine queries when they return too many results"""

    def __init__(self, openai_api_key: str = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))

    async def refine_query(
        self,
        original_query: str,
        gene: str,
        user_query: str,
        current_count: int,
        target_range: Tuple[int, int] = (50, 200)
    ) -> Optional[str]:
        """
        Use LLM to reformulate query to be more specific.

        Args:
            original_query: Current PubMed query
            gene: Gene symbol
            user_query: User's original question
            current_count: Current number of results
            target_range: Desired range of results

        Returns:
            Refined query or None if refinement fails
        """
        min_target, max_target = target_range

        prompt = f"""You are a PubMed query expert. A search returned too many papers.

CURRENT SITUATION:
- Query: {original_query}
- Results: {current_count:,} papers
- Target: {min_target}-{max_target} papers

USER'S QUESTION: {user_query}
GENE OF INTEREST: {gene}

TASK: Reformulate this PubMed query to be MORE SPECIFIC while staying relevant.

STRATEGIES TO USE:
1. Add more specific disease subtypes (e.g., "breast cancer" → "triple-negative breast cancer")
2. Add mechanism/pathway terms from user's question
3. Add treatment/intervention context if relevant
4. Add cell type or tissue specificity
5. Use more restrictive field tags [Title/Abstract]

IMPORTANT:
- Keep the gene term: {gene}[Title/Abstract]
- Add 1-3 additional specific terms from the user's question
- Use AND to combine terms (makes it more specific)
- Do NOT use OR (that makes it broader)

Return ONLY the new PubMed query, nothing else.
Format: GENE[Title/Abstract] AND "specific term"[Title/Abstract] AND "another term"[Title/Abstract]
"""

        try:
            from src.config import get_active_model
            response = self.client.chat.completions.create(
                model=get_active_model(),
                messages=[
                    {"role": "system", "content": "You are a PubMed query expert. Return only the query, no explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=200
            )

            refined_query = response.choices[0].message.content.strip()

            # Basic validation
            if gene not in refined_query:
                return None
            if len(refined_query) < len(original_query):
                return None  # Shouldn't be shorter

            return refined_query

        except Exception as e:
            print(f"  Agent refinement failed: {e}")
            return None
