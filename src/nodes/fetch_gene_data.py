"""Fetch all gene data (Phase 1 factual collection)."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple, Dict, Any

from dataclasses import dataclass, field
from pydantic_graph.nodes import BaseNode, GraphRunContext
from src.graph.state import GeneState
from src.agents.fetcher import fetch_complete_profile
from src.utils.provenance import ProvenanceTracker
from src.models.report_components import GeneProfile, clean_gene_name
import sqlite3
import time

if TYPE_CHECKING:
    from src.nodes.analyze_network_overlap import AnalyzeNetworkOverlap
    from src.agents.gene_extractor import GeneExtractionResult


# ============================================================================
# RESOLUTION DATA CLASSES
# ============================================================================

@dataclass
class ResolvedGene:
    """A successfully resolved gene identifier."""
    ensembl_id: str
    official_symbol: str
    original_identifier: str
    identifier_type: str   # 'symbol', 'ensembl_id', 'full_name', 'alias'
    lookup_tier: str       # 'direct', 'alias', 'ensembl_id', 'description'


@dataclass
class UnresolvedIdentifier:
    """An identifier that could not be resolved."""
    identifier: str
    identifier_type: str
    reason: str


def _lookup_gene_by_tiers(conn, gene_symbol):
    """
    Lookup gene in database using tiered approach:
    1. Direct symbol match
    2. Alias match
    3. Ensembl ID match
    4. Full gene name (description) partial match
    """
    cursor = conn.cursor()
    query_value = (gene_symbol or '').strip()
    if not query_value:
        return {
            'found': False,
            'notes': 'Empty gene query provided'
        }

    # Tier 1: Direct match
    cursor.execute("""
        SELECT gene_id, gene_symbol AS symbol
        FROM genes
        WHERE gene_symbol = ? COLLATE NOCASE
    """, (query_value,))

    result = cursor.fetchone()
    if result:
        print(f"    ↪ Lookup tier: direct symbol match for '{query_value}'")
        return {
            'found': True,
            'ensembl_id': result['gene_id'],
            'official_symbol': result['symbol'],
            'lookup_tier': 'direct'
        }

    # Tier 2: Alias match
    cursor.execute("""
        SELECT g.gene_id, g.gene_symbol AS symbol
        FROM genes g
        JOIN gene_alias ga ON g.gene_id = ga.gene_id
        WHERE ga.symbol_alias = ? COLLATE NOCASE
    """, (query_value,))

    result = cursor.fetchone()
    if result:
        print(f"    ↪ Lookup tier: alias match for '{query_value}'")
        return {
            'found': True,
            'ensembl_id': result['gene_id'],
            'official_symbol': result['symbol'],
            'lookup_tier': 'alias'
        }

    # Tier 3: Ensembl ID match
    cursor.execute("""
        SELECT gene_id, gene_symbol AS symbol
        FROM genes
        WHERE gene_id = ? COLLATE NOCASE
    """, (query_value,))

    result = cursor.fetchone()
    if result:
        print(f"    ↪ Lookup tier: ensembl_id match for '{query_value}'")
        return {
            'found': True,
            'ensembl_id': result['gene_id'],
            'official_symbol': result['symbol'],
            'lookup_tier': 'ensembl_id'
        }

    # Tier 4: Full gene name (description) partial match
    like_value = f"%{query_value}%"
    cursor.execute("""
        SELECT g.gene_id, g.gene_symbol AS symbol
        FROM gene_function gf
        JOIN genes g ON gf.gene_id = g.gene_id
        WHERE gf.description LIKE ? COLLATE NOCASE
        ORDER BY LENGTH(gf.description)
        LIMIT 1
    """, (like_value,))

    result = cursor.fetchone()
    if result:
        print(f"    ↪ Lookup tier: description match for '{query_value}'")
        return {
            'found': True,
            'ensembl_id': result['gene_id'],
            'official_symbol': result['symbol'],
            'lookup_tier': 'description'
        }

    # Not found
    return {
        'found': False,
        'notes': f'Gene identifier "{query_value}" not found in database'
    }


# ============================================================================
# INDIVIDUAL LOOKUP HELPER FUNCTIONS
# ============================================================================

def _lookup_by_direct_symbol(cursor, symbol: str) -> Dict[str, Any]:
    """Tier 1: Direct symbol match in genes table."""
    cursor.execute("""
        SELECT gene_id, gene_symbol AS symbol
        FROM genes
        WHERE gene_symbol = ? COLLATE NOCASE
    """, (symbol.strip(),))
    result = cursor.fetchone()
    if result:
        return {
            'found': True,
            'ensembl_id': result['gene_id'],
            'official_symbol': result['symbol'],
            'lookup_tier': 'direct'
        }
    return {'found': False}


def _lookup_by_alias(cursor, alias: str) -> Dict[str, Any]:
    """Tier 2: Alias match via gene_alias table."""
    cursor.execute("""
        SELECT g.gene_id, g.gene_symbol AS symbol
        FROM genes g
        JOIN gene_alias ga ON g.gene_id = ga.gene_id
        WHERE ga.symbol_alias = ? COLLATE NOCASE
    """, (alias.strip(),))
    result = cursor.fetchone()
    if result:
        return {
            'found': True,
            'ensembl_id': result['gene_id'],
            'official_symbol': result['symbol'],
            'lookup_tier': 'alias'
        }
    return {'found': False}


def _lookup_by_ensembl_id(cursor, ensembl_id: str) -> Dict[str, Any]:
    """Tier 3: Ensembl ID direct lookup."""
    cursor.execute("""
        SELECT gene_id, gene_symbol AS symbol
        FROM genes
        WHERE gene_id = ? COLLATE NOCASE
    """, (ensembl_id.strip(),))
    result = cursor.fetchone()
    if result:
        return {
            'found': True,
            'ensembl_id': result['gene_id'],
            'official_symbol': result['symbol'],
            'lookup_tier': 'ensembl_id'
        }
    return {'found': False}


def _lookup_by_description(cursor, description: str) -> Dict[str, Any]:
    """Tier 4: Description partial match in gene_function."""
    like_value = f"%{description.strip()}%"
    cursor.execute("""
        SELECT g.gene_id, g.gene_symbol AS symbol
        FROM gene_function gf
        JOIN genes g ON gf.gene_id = g.gene_id
        WHERE gf.description LIKE ? COLLATE NOCASE
        ORDER BY LENGTH(gf.description)
        LIMIT 1
    """, (like_value,))
    result = cursor.fetchone()
    if result:
        return {
            'found': True,
            'ensembl_id': result['gene_id'],
            'official_symbol': result['symbol'],
            'lookup_tier': 'description'
        }
    return {'found': False}


# ============================================================================
# MAIN RESOLUTION FUNCTION
# ============================================================================

def resolve_all_gene_identifiers(
    extraction_result: "GeneExtractionResult",
    db_path: str
) -> Tuple[List[ResolvedGene], List[UnresolvedIdentifier]]:
    """
    Resolve all gene identifiers from extraction to database records.

    Resolution order:
    1. Ensembl IDs → Direct lookup (highest confidence)
    2. Gene symbols → Direct symbol match, then alias fallback
    3. Aliases → Alias table lookup, then symbol fallback
    4. Full names → Description LIKE match (lowest confidence)

    Args:
        extraction_result: The enhanced gene extraction result
        db_path: Path to the SQLite database

    Returns:
        Tuple of (resolved_genes, unresolved_identifiers)
        - resolved_genes: List of ResolvedGene objects (deduplicated by ensembl_id)
        - unresolved_identifiers: List of UnresolvedIdentifier objects
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    resolved: Dict[str, ResolvedGene] = {}  # keyed by ensembl_id for deduplication
    unresolved: List[UnresolvedIdentifier] = []
    seen_identifiers: set = set()  # Track processed identifiers (case-insensitive)

    def add_resolved(result: Dict, identifier: str, id_type: str):
        """Helper to add a resolved gene, handling deduplication."""
        ensembl_id = result['ensembl_id']
        if ensembl_id not in resolved:
            resolved[ensembl_id] = ResolvedGene(
                ensembl_id=ensembl_id,
                official_symbol=result['official_symbol'],
                original_identifier=identifier,
                identifier_type=id_type,
                lookup_tier=result['lookup_tier']
            )
            print(f"    ✓ Resolved: {identifier} → {result['official_symbol']} (via {result['lookup_tier']})")
        else:
            # Already resolved by a previous identifier
            print(f"    ↪ Duplicate: {identifier} → {resolved[ensembl_id].official_symbol} (already resolved)")

    # Phase 1: Ensembl IDs (highest confidence, direct lookup)
    for ensembl_id in extraction_result.gene_ensembl_ids:
        if not ensembl_id or ensembl_id.upper() in seen_identifiers:
            continue
        seen_identifiers.add(ensembl_id.upper())

        result = _lookup_by_ensembl_id(cursor, ensembl_id)
        if result['found']:
            add_resolved(result, ensembl_id, 'ensembl_id')
        else:
            unresolved.append(UnresolvedIdentifier(
                identifier=ensembl_id,
                identifier_type='ensembl_id',
                reason='Ensembl ID not found in genes table'
            ))
            print(f"    ✗ Unresolved: {ensembl_id} (Ensembl ID not found)")

    # Phase 2: Gene symbols (try direct, then alias)
    for symbol in extraction_result.gene_symbols:
        if not symbol or symbol.upper() in seen_identifiers:
            continue
        seen_identifiers.add(symbol.upper())

        result = _lookup_by_direct_symbol(cursor, symbol)
        if result['found']:
            add_resolved(result, symbol, 'symbol')
            continue

        result = _lookup_by_alias(cursor, symbol)
        if result['found']:
            add_resolved(result, symbol, 'symbol')
            continue

        unresolved.append(UnresolvedIdentifier(
            identifier=symbol,
            identifier_type='symbol',
            reason='Symbol not found in genes or gene_alias tables'
        ))
        print(f"    ✗ Unresolved: {symbol} (symbol not found)")

    # Phase 3: Aliases (alias table, then symbol table)
    for alias in extraction_result.gene_aliases:
        if not alias or alias.upper() in seen_identifiers:
            continue
        seen_identifiers.add(alias.upper())

        result = _lookup_by_alias(cursor, alias)
        if result['found']:
            add_resolved(result, alias, 'alias')
            continue

        result = _lookup_by_direct_symbol(cursor, alias)
        if result['found']:
            add_resolved(result, alias, 'alias')
            continue

        unresolved.append(UnresolvedIdentifier(
            identifier=alias,
            identifier_type='alias',
            reason='Alias not found in gene_alias or genes tables'
        ))
        print(f"    ✗ Unresolved: {alias} (alias not found)")

    # Phase 4: Full names (description partial match - lowest confidence)
    for full_name in extraction_result.gene_full_names:
        if not full_name or full_name.upper() in seen_identifiers:
            continue
        seen_identifiers.add(full_name.upper())

        result = _lookup_by_description(cursor, full_name)
        if result['found']:
            add_resolved(result, full_name, 'full_name')
        else:
            unresolved.append(UnresolvedIdentifier(
                identifier=full_name,
                identifier_type='full_name',
                reason='Full name not found in gene_function descriptions'
            ))
            print(f"    ✗ Unresolved: {full_name} (full name not found)")

    conn.close()

    return list(resolved.values()), unresolved


def _flatten_profile(raw_profile):
    """Flatten nested profile structure into single dict"""
    if not raw_profile:
        return {}

    # If already flat, return as-is
    if not any(isinstance(v, dict) for v in raw_profile.values()):
        return raw_profile

    # Flatten nested dicts
    flattened = {}
    for key, value in raw_profile.items():
        if isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value

    return flattened


def _verify_profile_data(profile, gene_symbol):
    """Verify that profile has minimum required data"""
    if not profile:
        print(f"  ⚠️  Empty profile for {gene_symbol}")
        return False

    # Check for basic fields
    has_id = 'gene_id' in profile or 'ensembl_id' in profile
    has_symbol = 'symbol' in profile or 'gene_symbol' in profile

    if not has_id:
        print(f"  ⚠️  Missing gene_id for {gene_symbol}")
    if not has_symbol:
        print(f"  ⚠️  Missing symbol for {gene_symbol}")

    return has_id and has_symbol

@dataclass
class FetchAllGeneData(BaseNode[GeneState]):
    """Fetch factual data for each gene before any interpretation."""

    async def run(
        self,
        ctx: GraphRunContext[GeneState]
    ) -> FetchAllGeneData | AnalyzeNetworkOverlap:
        from src.nodes.analyze_network_overlap import AnalyzeNetworkOverlap

        print(f"\n{'='*70}")
        print("NODE: Fetch All Gene Data")
        print(f"{'='*70}")

        if not ctx.state.genes_to_process:
            print("✓ All genes fetched. Proceeding to cross-gene analyses.")
            ctx.state.factual_data_complete = True
            ctx.state.log_node_execution('fetch_gene_data')
            return AnalyzeNetworkOverlap()

        # Pop next gene from queue
        gene = ctx.state.genes_to_process.pop(0)
        ctx.state.current_gene = gene

        print(f"Processing: {gene}")
        print(f"Remaining in queue: {ctx.state.genes_to_process}")

        start_time = time.time()

        try:
            # Connect to database
            conn = sqlite3.connect(ctx.state.db_path)
            conn.row_factory = sqlite3.Row

            # Lookup gene in database
            lookup_result = _lookup_gene_by_tiers(conn, gene)

            if not lookup_result['found']:
                print(f"⚠ Gene {gene} not found in database: {lookup_result.get('notes', 'Unknown reason')}")
                # Store the lookup failure
                ctx.state.gene_mapping[gene] = lookup_result
                # Skip to next gene
                conn.close()
                return FetchAllGeneData()

            # Store lookup result
            ctx.state.gene_mapping[gene] = lookup_result
            ensembl_id = lookup_result['ensembl_id']
            official_symbol = lookup_result['official_symbol']

            print(f"  Found: {gene} → {official_symbol} ({ensembl_id})")

            # Fetch complete profile for this gene
            raw_profile = fetch_complete_profile(ensembl_id, ctx.state.db_path)

            # Flatten the profile for easier access
            profile = _flatten_profile(raw_profile)

            # Ensure correct gene_id and symbol
            if profile.get('gene_id') != ensembl_id:
                profile['gene_id'] = ensembl_id

            if official_symbol and profile.get('symbol') != official_symbol:
                profile['symbol'] = official_symbol
            elif official_symbol and not profile.get('symbol'):
                profile['symbol'] = official_symbol

            # Normalize chromosome naming for downstream logic
            if 'chromosome' not in profile and profile.get('chrom'):
                profile['chromosome'] = profile['chrom']

            # Track original query name (for alias handling)
            profile['query_name'] = gene

            # Verify data completeness
            _verify_profile_data(profile, gene)

            # Add provenance tracking
            provenance_tracker = ProvenanceTracker(ctx.state.db_path)
            profile['provenance'] = []

            # Track core fields
            core_fields = {
                'chromosome': (('chromosome', 'chrom'), 'genes', 'chromosome'),
                'start_position': (('start',), 'genes', 'start'),
                'end_position': (('end',), 'genes', 'end'),
                'strand': (('strand',), 'genes', 'strand'),
                'gene_type': (('gene_type',), 'genes', 'gene_type'),
                'source': (('source',), 'genes', 'data_source')
            }

            for field_name, (profile_keys, table, column_name) in core_fields.items():
                if isinstance(profile_keys, str):
                    profile_keys = (profile_keys,)

                value = None
                for key in profile_keys:
                    if key in profile and profile[key] is not None:
                        value = profile[key]
                        break

                if value is None:
                    continue

                prov = provenance_tracker.track_gene_field(
                    gene_id=ensembl_id,
                    field_name=field_name,
                    value=value,
                    table_name=table,
                    column_name=column_name
                )
                profile['provenance'].append(prov)

            # Track function/description
            if profile.get('description'):
                prov = provenance_tracker.track_gene_field(
                    gene_id=ensembl_id,
                    field_name='function_description',
                    value=profile['description'],
                    table_name='gene_function',
                    column_name='description'
                )
                profile['provenance'].append(prov)

            # Track GO terms
            for go_term in profile.get('go_terms', []):
                prov = provenance_tracker.track_go_term(ensembl_id, go_term)
                if 'provenance' not in go_term:
                    go_term['provenance'] = prov

            # Track interactions
            for interaction in profile.get('interactions', []):
                prov = provenance_tracker.track_interaction(ensembl_id, interaction)
                if 'provenance' not in interaction:
                    interaction['provenance'] = prov

            # Store in working memory
            ctx.state.current_gene_data = profile

            # Also add to accumulated data immediately using OFFICIAL SYMBOL as key
            ctx.state.all_gene_data[official_symbol] = profile

            # Track alias mapping if input differs from official symbol
            if gene != official_symbol:
                ctx.state.gene_alias_map[gene] = official_symbol

            # Track data source
            go_domains = []
            for term in profile.get('go_terms', []):
                domain = term.get('aspect') or term.get('namespace')
                if domain and domain not in go_domains:
                    go_domains.append(domain)

            ctx.state.add_data_source(
                gene=gene,
                source_type='gene_profile',
                source_info={
                    'ensembl_id': ensembl_id,
                    'database': ctx.state.db_path,
                    'table': 'genes',
                    'has_description': bool(profile.get('description')),
                    'has_function': bool(profile.get('description')),
                    'has_go_terms': len(profile.get('go_terms', [])) > 0,
                    'go_term_domains': go_domains,
                    'has_expression': len(profile.get('expression', [])) > 0,
                    'has_interactions': len(profile.get('interactions', [])) > 0
                }
            )

            # ========================================================================
            # CREATE GENEPROFILE (NEW)
            # ========================================================================
            # Extract GO terms by category (deduplicate by name, keep GO ID)
            go_terms = profile.get('go_terms', [])

            # Helper function to deduplicate GO terms by name but keep first GO ID
            def dedupe_go_terms(terms_list):
                seen_names = {}
                result = []
                for term in terms_list:
                    name = term['name']
                    if name not in seen_names:
                        seen_names[name] = True
                        result.append(term)
                return result

            # Extract molecular functions with GO IDs
            mf_list = [
                {'name': g.get('name', ''), 'go_id': g.get('go_id', '')}
                for g in go_terms
                if g.get('namespace') == 'molecular_function' or g.get('aspect') == 'F'
            ]
            molecular_functions = dedupe_go_terms(mf_list)

            # Extract biological processes with GO IDs
            bp_list = [
                {'name': g.get('name', ''), 'go_id': g.get('go_id', '')}
                for g in go_terms
                if g.get('namespace') == 'biological_process' or g.get('aspect') == 'P'
            ]
            biological_processes = dedupe_go_terms(bp_list)

            # Extract cellular components with GO IDs
            cc_list = [
                {'name': g.get('name', ''), 'go_id': g.get('go_id', '')}
                for g in go_terms
                if g.get('namespace') == 'cellular_component' or g.get('aspect') == 'C'
            ]
            cellular_components = dedupe_go_terms(cc_list)

            # Extract expression data
            expression_data = [
                {
                    "tissue": expr.get('tissue', 'Unknown'),
                    "tpm": float(
                        expr.get('tpm_value', expr.get('level', expr.get('tpm', 0.0)))
                    )
                }
                for expr in profile.get('expression', [])
            ]

            # Extract interactions
            interactions = [
                {
                    "partner": inter.get('partner_symbol') or inter.get('partner', 'Unknown'),
                    "score": float(inter.get('score', inter.get('combined_score', 0))) / 1000.0,
                    "description": inter.get('partner_annotation', inter.get('description', ''))
                }
                for inter in profile.get('interactions', [])
            ]

            # Determine mapping type
            mapping_type = "direct"
            mapping_note = None
            if gene != official_symbol:
                if lookup_result.get('lookup_tier') == 'alias':
                    mapping_type = "alias"
                    mapping_note = f"original: {gene}"
                else:
                    mapping_type = "corrected"
                    mapping_note = f"from {gene} to {official_symbol}"

            # Create GeneProfile (factual data only at this stage)
            gene_profile = GeneProfile(
                gene_symbol=official_symbol,
                gene_id=ensembl_id,
                location=f"{profile.get('chromosome', profile.get('chrom', 'Unknown'))}:"
                        f"{profile.get('start', '?')}-{profile.get('end', '?')}",
                gene_type=profile.get('gene_type', 'Unknown'),
                full_name=clean_gene_name(profile.get('description')) or official_symbol,
                long_description=profile.get('long_description'),
                search_term=gene,
                mapping_type=mapping_type,
                mapping_note=mapping_note,
                molecular_functions=molecular_functions,
                biological_processes=biological_processes,
                cellular_components=cellular_components,
                expression_data=expression_data,
                interactions=interactions,
                # Interpreted fields remain None - filled later by InterpretAllGenes
            )

            # Store in state
            ctx.state.gene_profiles[official_symbol] = gene_profile

            execution_time = time.time() - start_time

            print(f"✓ Fetched data for {gene}")
            print(f"  Total genes accumulated: {len(ctx.state.all_gene_data)}")
            print(f"  Execution time: {execution_time:.2f}s")

            ctx.state.log_node_execution('fetch_gene_data', execution_time)

            conn.close()

            return FetchAllGeneData()

        except Exception as e:
            ctx.state.error = f"Failed to fetch {gene}: {str(e)}"
            print(f"✗ ERROR: {ctx.state.error}")
            import traceback
            traceback.print_exc()
            # Skip this gene and try next one
            return FetchAllGeneData()
