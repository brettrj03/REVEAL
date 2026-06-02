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
import asyncio

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

async def resolve_all_gene_identifiers(
    extraction_result: "GeneExtractionResult",
    db_path: str
) -> Tuple[List[ResolvedGene], List[UnresolvedIdentifier]]:
    """
    Resolve all gene identifiers from extraction to database records (parallelized).

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

    @dataclass
    class LookupTask:
        """Represents a single identifier lookup task."""
        identifier: str
        identifier_type: str  # 'ensembl_id', 'symbol', 'alias', 'full_name'

    async def resolve_single_identifier(task: LookupTask) -> Tuple[str, str, Dict[str, Any]]:
        """
        Resolve a single identifier using its own database connection.

        Returns:
            Tuple of (identifier, identifier_type, lookup_result)
        """
        # Create dedicated connection for this task (thread safety)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # Perform lookup based on identifier type
            if task.identifier_type == 'ensembl_id':
                result = _lookup_by_ensembl_id(cursor, task.identifier)
                if not result['found']:
                    result['reason'] = 'Ensembl ID not found in genes table'
                return (task.identifier, task.identifier_type, result)

            elif task.identifier_type == 'symbol':
                # Try direct match first
                result = _lookup_by_direct_symbol(cursor, task.identifier)
                if result['found']:
                    return (task.identifier, task.identifier_type, result)

                # Try alias match as fallback
                result = _lookup_by_alias(cursor, task.identifier)
                if not result['found']:
                    result['reason'] = 'Symbol not found in genes or gene_alias tables'
                return (task.identifier, task.identifier_type, result)

            elif task.identifier_type == 'alias':
                # Try alias table first
                result = _lookup_by_alias(cursor, task.identifier)
                if result['found']:
                    return (task.identifier, task.identifier_type, result)

                # Try direct symbol as fallback
                result = _lookup_by_direct_symbol(cursor, task.identifier)
                if not result['found']:
                    result['reason'] = 'Alias not found in gene_alias or genes tables'
                return (task.identifier, task.identifier_type, result)

            elif task.identifier_type == 'full_name':
                result = _lookup_by_description(cursor, task.identifier)
                if not result['found']:
                    result['reason'] = 'Full name not found in gene_function descriptions'
                return (task.identifier, task.identifier_type, result)

            else:
                return (task.identifier, task.identifier_type, {
                    'found': False,
                    'reason': f'Unknown identifier type: {task.identifier_type}'
                })

        finally:
            conn.close()

    # Collect all identifiers into tasks
    tasks: List[LookupTask] = []
    seen_identifiers: set = set()

    # Phase 1: Ensembl IDs
    for ensembl_id in extraction_result.gene_ensembl_ids:
        if ensembl_id and ensembl_id.upper() not in seen_identifiers:
            seen_identifiers.add(ensembl_id.upper())
            tasks.append(LookupTask(ensembl_id, 'ensembl_id'))

    # Phase 2: Gene symbols
    for symbol in extraction_result.gene_symbols:
        if symbol and symbol.upper() not in seen_identifiers:
            seen_identifiers.add(symbol.upper())
            tasks.append(LookupTask(symbol, 'symbol'))

    # Phase 3: Aliases
    for alias in extraction_result.gene_aliases:
        if alias and alias.upper() not in seen_identifiers:
            seen_identifiers.add(alias.upper())
            tasks.append(LookupTask(alias, 'alias'))

    # Phase 4: Full names
    for full_name in extraction_result.gene_full_names:
        if full_name and full_name.upper() not in seen_identifiers:
            seen_identifiers.add(full_name.upper())
            tasks.append(LookupTask(full_name, 'full_name'))

    # Run all lookups concurrently
    results = await asyncio.gather(*[resolve_single_identifier(task) for task in tasks])

    # Process results - deduplicate by ensembl_id
    resolved: Dict[str, ResolvedGene] = {}  # keyed by ensembl_id for deduplication
    unresolved: List[UnresolvedIdentifier] = []

    for identifier, identifier_type, lookup_result in results:
        if lookup_result['found']:
            ensembl_id = lookup_result['ensembl_id']
            if ensembl_id not in resolved:
                resolved[ensembl_id] = ResolvedGene(
                    ensembl_id=ensembl_id,
                    official_symbol=lookup_result['official_symbol'],
                    original_identifier=identifier,
                    identifier_type=identifier_type,
                    lookup_tier=lookup_result['lookup_tier']
                )
                print(f"    ✓ Resolved: {identifier} → {lookup_result['official_symbol']} (via {lookup_result['lookup_tier']})")
            else:
                # Already resolved by a previous identifier
                print(f"    ↪ Duplicate: {identifier} → {resolved[ensembl_id].official_symbol} (already resolved)")
        else:
            unresolved.append(UnresolvedIdentifier(
                identifier=identifier,
                identifier_type=identifier_type,
                reason=lookup_result.get('reason', 'Unknown reason')
            ))
            print(f"    ✗ Unresolved: {identifier} ({lookup_result.get('reason', 'Unknown reason')})")

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
    ) -> AnalyzeNetworkOverlap:
        from src.nodes.analyze_network_overlap import AnalyzeNetworkOverlap

        print(f"\n{'='*70}")
        print("NODE: Fetch All Gene Data")
        print(f"{'='*70}")

        if not ctx.state.genes_to_process:
            print("✓ All genes fetched. Proceeding to cross-gene analyses.")
            ctx.state.factual_data_complete = True
            ctx.state.log_node_execution('fetch_gene_data')
            return AnalyzeNetworkOverlap()

        # Take all genes from queue
        genes_to_fetch = ctx.state.genes_to_process[:]
        print(f"Processing {len(genes_to_fetch)} genes concurrently...")

        start_time = time.time()

        async def fetch_single_gene(gene: str, db_path: str) -> Dict[str, Any] | None:
            """
            Fetch data for a single gene using its own database connection.

            Returns:
                Dict containing gene data with 'official_symbol' key for matching,
                or None if gene fetch failed.
            """
            try:
                # Create dedicated connection for this task (thread safety)
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row

                # Lookup gene in database
                lookup_result = _lookup_gene_by_tiers(conn, gene)

                if not lookup_result['found']:
                    print(f"⚠ Gene {gene} not found in database: {lookup_result.get('notes', 'Unknown reason')}")
                    conn.close()
                    return {
                        'gene': gene,
                        'official_symbol': None,
                        'lookup_result': lookup_result,
                        'error': 'not_found'
                    }

                ensembl_id = lookup_result['ensembl_id']
                official_symbol = lookup_result['official_symbol']

                print(f"  Found: {gene} → {official_symbol} ({ensembl_id})")

                # Fetch complete profile for this gene
                raw_profile = fetch_complete_profile(ensembl_id, db_path)

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
                provenance_tracker = ProvenanceTracker(db_path)
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

                # Track data source
                go_domains = []
                for term in profile.get('go_terms', []):
                    domain = term.get('aspect') or term.get('namespace')
                    if domain and domain not in go_domains:
                        go_domains.append(domain)

                data_source_info = {
                    'ensembl_id': ensembl_id,
                    'database': db_path,
                    'table': 'genes',
                    'has_description': bool(profile.get('description')),
                    'has_function': bool(profile.get('description')),
                    'has_go_terms': len(profile.get('go_terms', [])) > 0,
                    'go_term_domains': go_domains,
                    'has_expression': len(profile.get('expression', [])) > 0,
                    'has_interactions': len(profile.get('interactions', [])) > 0
                }

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

                conn.close()

                print(f"✓ Fetched data for {gene}")

                # Return all data needed to update state
                return {
                    'gene': gene,
                    'official_symbol': official_symbol,
                    'ensembl_id': ensembl_id,
                    'lookup_result': lookup_result,
                    'profile': profile,
                    'gene_profile': gene_profile,
                    'data_source_info': data_source_info,
                    'error': None
                }

            except Exception as e:
                print(f"✗ ERROR fetching {gene}: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    'gene': gene,
                    'official_symbol': None,
                    'error': str(e)
                }

        # Fetch all genes concurrently
        results = await asyncio.gather(*[
            fetch_single_gene(gene, ctx.state.db_path)
            for gene in genes_to_fetch
        ])

        # Process results - match by official_symbol
        successful_genes = 0
        failed_genes = 0

        for result in results:
            if result is None:
                failed_genes += 1
                continue

            gene = result['gene']
            official_symbol = result.get('official_symbol')

            # Store lookup result
            if 'lookup_result' in result:
                ctx.state.gene_mapping[gene] = result['lookup_result']

            # Handle errors
            if result.get('error'):
                failed_genes += 1
                continue

            # Store successful results using official_symbol as key
            if official_symbol:
                ctx.state.all_gene_data[official_symbol] = result['profile']
                ctx.state.gene_profiles[official_symbol] = result['gene_profile']

                # Track alias mapping if input differs from official symbol
                if gene != official_symbol:
                    ctx.state.gene_alias_map[gene] = official_symbol

                # Track data source
                ctx.state.add_data_source(
                    gene=gene,
                    source_type='gene_profile',
                    source_info=result['data_source_info']
                )

                successful_genes += 1

        # Clear the processing queue
        ctx.state.genes_to_process = []

        execution_time = time.time() - start_time

        print(f"\n✓ Fetched data for all genes concurrently")
        print(f"  Successful: {successful_genes}")
        print(f"  Failed: {failed_genes}")
        print(f"  Total genes accumulated: {len(ctx.state.all_gene_data)}")
        print(f"  Execution time: {execution_time:.2f}s")

        ctx.state.log_node_execution('fetch_gene_data', execution_time)
        ctx.state.factual_data_complete = True

        return AnalyzeNetworkOverlap()
