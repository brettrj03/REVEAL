"""
Agent 1: Context-Aware Data Fetcher with Deterministic Fallback
Intelligently fetches relevant gene data based on experiment type
Can run with or without LLM agent involvement
ENHANCED: Now includes STRING protein interaction analysis + Multi-ID format support
"""
from pydantic_ai import Agent, RunContext
from typing import List, Dict, Any, Optional
import sqlite3
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from src.models.models import ExperimentContext, GeneDataCollection

@dataclass
class FetcherDeps:
    """Dependencies for the data fetcher agent"""
    db_path: str
    experiment_context: ExperimentContext

# ============================================================================
# DETERMINISTIC DATABASE FUNCTIONS (always work, no LLM needed)
# ============================================================================

def get_conn(db_path: str) -> sqlite3.Connection:
    """Get database connection with Row factory"""
    # Ensure the parent directory exists
    import os
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def convert_ids_to_ensembl(identifiers: List[str], db_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Convert various gene identifiers to Ensembl IDs
    Handles: Gene symbols, Ensembl IDs, NCBI Gene IDs (GeneID:), HGNC IDs (HGNC:)

    Returns: {original_identifier: {'ensembl_id': str, 'official_symbol': str, 'found': bool}}
    """
    conn = get_conn(db_path)
    cursor = conn.cursor()
    results = {}

    for identifier in identifiers:
        identifier = identifier.strip()
        if not identifier:
            continue

        gene_id = None
        symbol = None

        # Case 1: Already an Ensembl ID (ENSG00000...)
        if identifier.startswith('ENSG'):
            cursor.execute("""
                SELECT gene_id, gene_symbol AS symbol
                FROM genes
                WHERE gene_id = ?
                LIMIT 1
            """, (identifier,))
            gene = cursor.fetchone()

            if gene:
                gene_id = gene['gene_id']
                symbol = gene['symbol']

        # Case 2: NCBI Gene ID (GeneID:12345)
        elif identifier.startswith('GeneID:'):
            ncbi_id = identifier.replace('GeneID:', '')
            # Try to find by NCBI gene ID in aliases or gene_id field
            cursor.execute("""
                SELECT g.gene_id, g.gene_symbol AS symbol
                FROM genes g
                JOIN gene_alias ga ON g.gene_id = ga.gene_id
                WHERE ga.symbol_alias = ? OR ga.data_source LIKE ?
                LIMIT 1
            """, (ncbi_id, f'%{ncbi_id}%'))
            gene = cursor.fetchone()

            if gene:
                gene_id = gene['gene_id']
                symbol = gene['symbol']

        # Case 3: HGNC ID (HGNC:12345)
        elif identifier.startswith('HGNC:'):
            hgnc_id = identifier.replace('HGNC:', '')
            # Check aliases table for HGNC IDs
            cursor.execute("""
                SELECT g.gene_id, g.gene_symbol AS symbol
                FROM genes g
                JOIN gene_alias ga ON g.gene_id = ga.gene_id
                WHERE ga.symbol_alias LIKE ? OR ga.data_source LIKE ?
                LIMIT 1
            """, (f'HGNC:{hgnc_id}', f'%HGNC:{hgnc_id}%'))
            gene = cursor.fetchone()

            if gene:
                gene_id = gene['gene_id']
                symbol = gene['symbol']

        # Case 4: Gene symbol (TP53, BRCA1, etc.)
        else:
            # Try direct match
            cursor.execute("""
                SELECT gene_id, gene_symbol AS symbol
                FROM genes
                WHERE LOWER(gene_symbol) = LOWER(?)
                LIMIT 1
            """, (identifier,))
            gene = cursor.fetchone()

            if not gene:
                # Try alias match
                cursor.execute("""
                    SELECT g.gene_id, g.gene_symbol AS symbol
                    FROM genes g
                    JOIN gene_alias ga ON g.gene_id = ga.gene_id
                    WHERE LOWER(ga.symbol_alias) = LOWER(?)
                    LIMIT 1
                """, (identifier,))
                gene = cursor.fetchone()

            if gene:
                gene_id = gene['gene_id']
                symbol = gene['symbol']

        # Store result
        if gene_id:
            results[identifier] = {
                'ensembl_id': gene_id,
                'official_symbol': symbol,
                'found': True,
                'original_format': 'ensembl' if identifier.startswith('ENSG') else 'other'
            }
        else:
            results[identifier] = {
                'ensembl_id': None,
                'official_symbol': None,
                'found': False,
                'error': f'Identifier {identifier} not found in database'
            }

    conn.close()
    return results

def convert_to_ensembl_ids(gene_symbols: List[str], db_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Convert gene symbols or IDs to Ensembl IDs (deterministic, no LLM)
    Now handles: Gene symbols, Ensembl IDs, GeneID:X, HGNC:X

    This is the main entry point - it calls convert_ids_to_ensembl
    Returns: {original_symbol: {'ensembl_id': str, 'official_symbol': str, 'found': bool}}
    """
    return convert_ids_to_ensembl(gene_symbols, db_path)

def get_string_protein_info(gene_symbol: str, gene_id: str, db_path: str) -> Dict[str, Any]:
    """
    Check if a gene has STRING protein data
    Returns: {'has_protein': bool, 'protein_id': str, 'annotation': str}
    """
    conn = get_conn(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT protein_id, symbol, annotation
        FROM string_proteins
        WHERE UPPER(symbol) = UPPER(?) OR gene_id = ?
        LIMIT 1
    """, (gene_symbol, gene_id))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'has_protein': True,
            'protein_id': row['protein_id'],
            'symbol': row['symbol'],
            'annotation': row['annotation']
        }
    return {'has_protein': False, 'protein_id': None, 'symbol': None, 'annotation': None}

def get_top_interactions(protein_id: str, db_path: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get top N protein interactions for a given protein
    Returns: List of top interactions sorted by score
    """
    conn = get_conn(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            CASE
                WHEN i.protein_id_1 = ? THEN i.protein_id_2
                ELSE i.protein_id_1
            END as partner_protein_id,
            i.combined_score,
            p.symbol as partner_symbol,
            p.annotation as partner_annotation
        FROM string_interactions i
        JOIN string_proteins p ON (
            CASE
                WHEN i.protein_id_1 = ? THEN i.protein_id_2 = p.protein_id
                ELSE i.protein_id_1 = p.protein_id
            END
        )
        WHERE (i.protein_id_1 = ? OR i.protein_id_2 = ?)
        ORDER BY i.combined_score DESC
        LIMIT ?
    """, (protein_id, protein_id, protein_id, protein_id, limit))

    results = []
    for row in cursor.fetchall():
        results.append({
            'partner_protein_id': row['partner_protein_id'],
            'partner_symbol': row['partner_symbol'] or 'Unknown',
            'partner_annotation': row['partner_annotation'],
            'combined_score': row['combined_score'],
            'confidence': get_confidence_label(row['combined_score'])
        })

    conn.close()
    return results

def check_direct_interaction(protein_id_1: str, protein_id_2: str, db_path: str) -> Optional[Dict[str, Any]]:
    """
    Check if two proteins directly interact
    Returns: {'score': int, 'confidence': str} or None
    """
    conn = get_conn(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT combined_score
        FROM string_interactions
        WHERE (protein_id_1 = ? AND protein_id_2 = ?)
           OR (protein_id_1 = ? AND protein_id_2 = ?)
        LIMIT 1
    """, (protein_id_1, protein_id_2, protein_id_2, protein_id_1))

    row = cursor.fetchone()
    conn.close()

    if row:
        score = row['combined_score']
        return {
            'score': score,
            'confidence': get_confidence_label(score)
        }
    return None

def get_confidence_label(score: int) -> str:
    """Convert STRING score to confidence label"""
    if score >= 900:
        return "HIGHEST"
    elif score >= 700:
        return "HIGH"
    elif score >= 400:
        return "MEDIUM"
    else:
        return "LOW"

def analyze_protein_network(gene_symbols: List[str], ensembl_mapping: Dict[str, Dict], db_path: str) -> Dict[str, Any]:
    """
    Analyze protein interaction network for a list of genes
    Returns comprehensive network analysis including:
    - Which genes have protein data
    - Direct interactions between query genes
    - Top 5 interactions for each protein-coding gene
    """
    conn = get_conn(db_path)
    cursor = conn.cursor()

    # Step 1: Identify which genes are protein-coding and have STRING data
    protein_status = {}
    protein_map = {}  # symbol -> protein_id

    for symbol in gene_symbols:
        if symbol not in ensembl_mapping or not ensembl_mapping[symbol]['found']:
            protein_status[symbol] = {
                'has_protein': False,
                'reason': 'Gene not found in database'
            }
            continue

        gene_id = ensembl_mapping[symbol]['ensembl_id']
        official_symbol = ensembl_mapping[symbol]['official_symbol']

        # Check if protein-coding
        cursor.execute("""
            SELECT gene_type
            FROM genes
            WHERE gene_id = ?
        """, (gene_id,))

        row = cursor.fetchone()
        if not row or row['gene_type'] != 'protein_coding':
            protein_status[symbol] = {
                'has_protein': False,
                'gene_type': row['gene_type'] if row else 'Unknown',
                'reason': f"Not protein-coding ({row['gene_type'] if row else 'Unknown'})"
            }
            continue

        # Check if has STRING data
        protein_info = get_string_protein_info(official_symbol, gene_id, db_path)

        if protein_info['has_protein']:
            protein_status[symbol] = {
                'has_protein': True,
                'protein_id': protein_info['protein_id'],
                'symbol': protein_info['symbol'],
                'annotation': protein_info['annotation'],
                'gene_type': 'protein_coding'
            }
            protein_map[symbol] = protein_info['protein_id']
        else:
            protein_status[symbol] = {
                'has_protein': False,
                'gene_type': 'protein_coding',
                'reason': 'No STRING protein data available'
            }

    # Count proteins found
    proteins_with_data = [s for s, info in protein_status.items() if info['has_protein']]

    # Step 2: Check for direct interactions between query proteins
    direct_interactions = []

    if len(proteins_with_data) >= 2:
        for i, symbol1 in enumerate(proteins_with_data):
            for symbol2 in proteins_with_data[i+1:]:
                protein_id_1 = protein_map[symbol1]
                protein_id_2 = protein_map[symbol2]

                interaction = check_direct_interaction(protein_id_1, protein_id_2, db_path)

                if interaction:
                    direct_interactions.append({
                        'gene1': symbol1,
                        'gene2': symbol2,
                        'protein1': protein_id_1,
                        'protein2': protein_id_2,
                        'score': interaction['score'],
                        'confidence': interaction['confidence']
                    })

    # Step 3: Get top 5 interactions for each protein
    top_interactions = {}

    if proteins_with_data:
        for symbol in proteins_with_data:
            protein_id = protein_map[symbol]
            interactions = get_top_interactions(protein_id, db_path, limit=5)
            top_interactions[symbol] = interactions

    conn.close()

    return {
        'protein_status': protein_status,
        'proteins_with_data': proteins_with_data,
        'direct_interactions': direct_interactions,
        'top_interactions': top_interactions,
        'summary': {
            'total_genes': len(gene_symbols),
            'protein_coding_with_data': len(proteins_with_data),
            'direct_interactions_found': len(direct_interactions),
            'genes_analyzed': len(top_interactions)
        }
    }

def fetch_complete_profile(ensembl_id: str, db_path: str) -> Dict[str, Any]:
    """
    Fetch complete gene profile (deterministic, no LLM)
    Returns all available data for a gene
    """
    conn = get_conn(db_path)
    cursor = conn.cursor()

    profile = {'gene_id': ensembl_id}

    # Core info (use aliases for compatibility with legacy keys)
    cursor.execute("""
        SELECT
            gene_id,
            gene_symbol AS symbol,
            gene_type,
            chromosome AS chrom,
            start,
            "end" AS end,
            strand,
            data_source AS source,
            NULL AS level,
            NULL AS source_release
        FROM genes WHERE gene_id = ?
    """, (ensembl_id,))
    row = cursor.fetchone()
    profile['core'] = dict(row) if row else {}

    # Function
    cursor.execute("""
        SELECT
            description,
            NULL AS long_description,
            NULL AS long_description_source,
            NULL AS evidence_tier,
            data_source AS source,
            NULL AS updated_at
        FROM gene_function WHERE gene_id = ?
    """, (ensembl_id,))
    row = cursor.fetchone()
    profile['function'] = dict(row) if row else {}

    # Aliases
    cursor.execute("""
        SELECT symbol_alias, data_source AS source
        FROM gene_alias WHERE gene_id = ?
    """, (ensembl_id,))
    profile['aliases'] = [dict(r) for r in cursor.fetchall()]

    # Tags table removed in new schema; keep empty list for compatibility.
    profile['tags'] = []

    # GO terms
    cursor.execute("""
        SELECT
            gg.go_id,
            gt.name,
            gt.namespace AS namespace,
            gt.definition AS definition,
            gg.evidence,
            NULL AS ref,
            gt.namespace AS aspect
        FROM gene_go gg
        JOIN go_terms gt ON gg.go_id = gt.go_id
        WHERE gg.gene_id = ? AND gt.is_obsolete = 0
    """, (ensembl_id,))
    profile['go_terms'] = [dict(r) for r in cursor.fetchall()]

    # Expression
    cursor.execute("""
        SELECT tissue, tpm_value AS level, unit AS units, data_source AS dataset
        FROM expression WHERE gene_id = ?
    """, (ensembl_id,))
    profile['expression'] = [dict(r) for r in cursor.fetchall()]

    # STRING interactions
    cursor.execute("""
        SELECT protein_id
        FROM string_proteins
        WHERE gene_id = ?
    """, (ensembl_id,))
    target_proteins = [row['protein_id'] for row in cursor.fetchall()]

    interactions: List[Dict[str, Any]] = []
    total_interaction_count = 0

    if target_proteins:
        placeholders = ','.join('?' * len(target_proteins))
        query = f"""
            SELECT partner_symbol, partner_function, combined_score, data_source
            FROM (
                SELECT
                    COALESCE(g_partner.gene_symbol, sp_partner.symbol, sp_partner.protein_id) AS partner_symbol,
                    gf_partner.description AS partner_function,
                    si.combined_score,
                    COALESCE(si.data_source, 'STRING') AS data_source
                FROM string_interactions si
                JOIN string_proteins sp_partner ON sp_partner.protein_id = si.protein_id_2
                LEFT JOIN genes g_partner ON g_partner.gene_id = sp_partner.gene_id
                LEFT JOIN gene_function gf_partner ON gf_partner.gene_id = g_partner.gene_id
                WHERE si.protein_id_1 IN ({placeholders})

                UNION ALL

                SELECT
                    COALESCE(g_partner.gene_symbol, sp_partner.symbol, sp_partner.protein_id) AS partner_symbol,
                    gf_partner.description AS partner_function,
                    si.combined_score,
                    COALESCE(si.data_source, 'STRING') AS data_source
                FROM string_interactions si
                JOIN string_proteins sp_partner ON sp_partner.protein_id = si.protein_id_1
                LEFT JOIN genes g_partner ON g_partner.gene_id = sp_partner.gene_id
                LEFT JOIN gene_function gf_partner ON gf_partner.gene_id = g_partner.gene_id
                WHERE si.protein_id_2 IN ({placeholders})
            )
            ORDER BY combined_score DESC
        """
        params = target_proteins + target_proteins
        cursor.execute(query, params)
        rows = cursor.fetchall()
        total_interaction_count = len(rows)
        interactions = [
            {
                'partner_symbol': row['partner_symbol'],
                'interaction_type': 'STRING',
                'pubmed_id': None,
                'source': row['data_source'],
                'evidence_code': None,
                'score': row['combined_score'],
                'partner_function': row['partner_function']
            }
            for row in rows
        ]

    profile['interactions'] = interactions
    profile['total_interaction_count'] = total_interaction_count

    # STRING Protein Info (for protein-coding genes)
    # Get protein description from STRING database if available
    try:
        # First check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='string_proteins'
        """)
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Try to get the gene symbol first
            cursor.execute("SELECT gene_symbol AS symbol FROM genes WHERE gene_id = ?", (ensembl_id,))
            symbol_row = cursor.fetchone()
            gene_symbol = symbol_row['symbol'] if symbol_row else None

            if gene_symbol:
                # Query by gene symbol (column is called 'symbol' in string_proteins)
                cursor.execute("""
                    SELECT protein_id, symbol as gene_name, annotation
                    FROM string_proteins
                    WHERE UPPER(symbol) = UPPER(?)
                    LIMIT 1
                """, (gene_symbol,))
                protein_row = cursor.fetchone()

                if protein_row:
                    profile['string_protein'] = dict(protein_row)
                else:
                    profile['string_protein'] = {}
            else:
                profile['string_protein'] = {}
        else:
            profile['string_protein'] = {}
    except Exception as e:
        # If any error occurs (table doesn't exist, wrong columns, etc.), just skip
        profile['string_protein'] = {}

    conn.close()
    return profile


def fetch_gene_profile(gene_symbol: str, db_path: str) -> Dict[str, Any]:
    """
    Fetch complete profile for a SINGLE gene (stateful pipeline helper).

    This is a wrapper around fetch_complete_profile that handles lookup.
    Used by the stateful pipeline to fetch one gene at a time.

    Args:
        gene_symbol: Gene symbol to fetch
        db_path: Path to the database

    Returns:
        Dict with gene profile data, or empty dict if not found
    """
    conn = get_conn(db_path)

    # Look up the gene to get Ensembl ID
    from src.nodes.retrieve_data import _lookup_gene_by_tiers
    lookup_result = _lookup_gene_by_tiers(conn, gene_symbol)

    if not lookup_result['found']:
        conn.close()
        return {}

    ensembl_id = lookup_result['ensembl_id']
    conn.close()

    # Fetch the complete profile
    return fetch_complete_profile(ensembl_id, db_path)


# ============================================================================
# AGENT-BASED TOOLS (for smart context-aware fetching)
# ============================================================================

# Lazy agent initialization to avoid requiring API key at import
_data_fetcher = None
_data_fetcher_model: str = ""

def get_data_fetcher():
    """Get or create the data fetcher agent, rebuilding if the active model changed."""
    global _data_fetcher, _data_fetcher_model
    from src.config import get_active_model
    current_model = get_active_model()
    if _data_fetcher is None or _data_fetcher_model != current_model:
        _data_fetcher_model = current_model
        _data_fetcher = Agent(
            f'openai:{current_model}',
            deps_type=FetcherDeps,
            system_prompt="""You are an intelligent data collection agent for gene analysis.

    Your job is to:
    1. Convert gene symbols to Ensembl IDs
    2. Analyze the experiment context to determine which data is most relevant
    3. Selectively use tools based on the experiment type
    4. Collect comprehensive but focused data

    IMPORTANT: Be strategic about which tools you use based on the experiment context:

    ALWAYS DO:
    - Convert gene symbols to Ensembl IDs (convert_to_ensembl)
    - Get gene information AND function for ALL genes (get_gene_info) - this is mandatory
    - Check for protein interactions (get_protein_interactions_network) - important for understanding gene relationships

    THEN CONDITIONALLY ADD:
    - If the experiment involves EXPRESSION (RNA-seq, microarray, transcriptomics):
      → Use get_gene_expression, get_tissue_expression
      → Focus on GO biological processes

    - If the experiment involves PROTEIN INTERACTIONS or SIGNALING:
      → Use get_protein_interactions, find_interacting_partners
      → Look at literature-based interactions

    - If the experiment involves DNA DAMAGE, CELL CYCLE, or PATHWAY analysis:
      → Use get_gene_go_terms for biological processes
      → Use pathway-related tools

    - If the experiment involves SPECIFIC TISSUES or CELL TYPES:
      → Use get_tissue_expression to check tissue specificity

    Always start by converting gene symbols to Ensembl IDs, then proceed strategically.
    Don't collect irrelevant data - be smart about what the researcher needs."""
        )
    return _data_fetcher

def register_tools():
    """Register tools with the agent - call this when agent is needed"""
    agent = get_data_fetcher()

    @agent.tool
    def convert_to_ensembl(ctx: RunContext[FetcherDeps], gene_symbols: List[str]) -> dict:
        """Convert gene symbols to Ensembl IDs. Also handles gene aliases."""
        return {'mapping': convert_to_ensembl_ids(gene_symbols, ctx.deps.db_path)}

    @agent.tool
    def get_gene_info(ctx: RunContext[FetcherDeps], ensembl_ids: List[str]) -> List[dict]:
        """Get comprehensive information about genes using Ensembl IDs"""
        conn = get_conn(ctx.deps.db_path)
        cursor = conn.cursor()

        results = []
        placeholders = ','.join('?' * len(ensembl_ids))

        cursor.execute(f"""
            SELECT g.gene_id, g.gene_symbol AS symbol, g.gene_type, g.chromosome AS chrom,
                   g.start, g."end" AS end, g.strand, g.data_source AS source
            FROM genes g
            WHERE g.gene_id IN ({placeholders})
        """, ensembl_ids)

        for gene_info in cursor.fetchall():
            result = dict(gene_info)
            gene_id = result['gene_id']

            cursor.execute("""
                SELECT description, NULL AS evidence_tier, data_source AS source
                FROM gene_function WHERE gene_id = ?
            """, (gene_id,))

            function_info = cursor.fetchone()
            if function_info:
                result['function'] = dict(function_info)

            cursor.execute("""
                SELECT symbol_alias, data_source AS source
                FROM gene_alias WHERE gene_id = ? LIMIT 5
            """, (gene_id,))

            result['aliases'] = [dict(row) for row in cursor.fetchall()]
            results.append(result)

        conn.close()
        return results

    @agent.tool
    def get_protein_interactions_network(ctx: RunContext[FetcherDeps], gene_symbols: List[str]) -> dict:
        """
        Analyze protein interaction network for genes.
        Returns which genes have protein data, direct interactions between them,
        and top 5 interactions for each protein-coding gene.
        """
        mapping = convert_to_ensembl_ids(gene_symbols, ctx.deps.db_path)
        return analyze_protein_network(gene_symbols, mapping, ctx.deps.db_path)

    @agent.tool
    def get_gene_go_terms(ctx: RunContext[FetcherDeps], ensembl_ids: List[str],
                          namespace: str = None) -> List[dict]:
        """
        Get Gene Ontology terms for genes.
        namespace: Optional filter - 'biological_process', 'molecular_function', or 'cellular_component'
        """
        conn = get_conn(ctx.deps.db_path)
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(ensembl_ids))
        namespace_filter = ""
        params = list(ensembl_ids)
        if namespace:
            namespace_filter = "AND COALESCE(gg.go_namespace, gt.namespace) = ?"
            params.append(namespace)

        cursor.execute(f"""
            SELECT
                gt.go_id,
                gt.name,
                COALESCE(gg.go_namespace, gt.namespace) AS namespace,
                COALESCE(gg.go_namespace, gt.namespace) AS aspect,
                COUNT(DISTINCT g.gene_id) as gene_count,
                GROUP_CONCAT(DISTINCT g.gene_symbol) as genes
            FROM go_terms gt
            JOIN gene_go gg ON gt.go_id = gg.go_id
            JOIN genes g ON gg.gene_id = g.gene_id
            WHERE g.gene_id IN ({placeholders})
            AND gt.is_obsolete = 0
            {namespace_filter}
            GROUP BY
                gt.go_id,
                gt.name,
                COALESCE(gg.go_namespace, gt.namespace),
                COALESCE(gg.go_namespace, gt.namespace)
            HAVING gene_count >= 2
            ORDER BY gene_count DESC, namespace
            LIMIT 30
        """, params)

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    @agent.tool
    def get_protein_interactions(ctx: RunContext[FetcherDeps], gene_symbols: List[str],
                                 min_score: int = 400) -> List[dict]:
        """Get protein-protein interactions from STRING database"""
        conn = get_conn(ctx.deps.db_path)
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(gene_symbols))

        cursor.execute(f"""
            SELECT DISTINCT protein_id, symbol
            FROM string_proteins
            WHERE symbol IN ({placeholders})
        """, gene_symbols)

        protein_map = {row['symbol']: row['protein_id'] for row in cursor.fetchall()}

        if not protein_map:
            conn.close()
            return []

        protein_ids = list(protein_map.values())
        placeholders2 = ','.join('?' * len(protein_ids))

        cursor.execute(f"""
            SELECT
                si.protein_id_1, si.protein_id_2, si.combined_score,
                sp1.symbol as gene1, sp2.symbol as gene2
            FROM string_interactions si
            JOIN string_proteins sp1 ON si.protein_id_1 = sp1.protein_id
            JOIN string_proteins sp2 ON si.protein_id_2 = sp2.protein_id
            WHERE si.protein_id_1 IN ({placeholders2})
            AND si.protein_id_2 IN ({placeholders2})
            AND si.combined_score >= ?
            ORDER BY si.combined_score DESC
            LIMIT 50
        """, protein_ids + protein_ids + [min_score])

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    @agent.tool
    def get_gene_expression(ctx: RunContext[FetcherDeps], ensembl_ids: List[str]) -> List[dict]:
        """Get tissue expression data for genes"""
        conn = get_conn(ctx.deps.db_path)
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(ensembl_ids))

        cursor.execute(f"""
            SELECT g.gene_symbol AS symbol, g.gene_id, e.tissue,
                   e.tpm_value AS level, e.unit AS units, e.data_source AS dataset
            FROM expression e
            JOIN genes g ON e.gene_id = g.gene_id
            WHERE g.gene_id IN ({placeholders})
            ORDER BY g.gene_symbol, e.tpm_value DESC
        """, ensembl_ids)

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    return agent

# ============================================================================
# MAIN INTERFACE FUNCTIONS
# ============================================================================

def fetch_gene_data_deterministic(
    gene_list: List[str],
    experiment_context: ExperimentContext,
    db_path: str = "gene_database.sqlite",
    output_file: str = "gene_data.json"
) -> str:
    """
    Fast deterministic fetch (NO LLM) - fetches ALL data for all genes
    NOW INCLUDES: Protein interaction network analysis + Multi-ID format support
    Returns: path to saved JSON file
    """
    print("\n" + "="*70)
    print("DETERMINISTIC DATA FETCH (No LLM)")
    print("="*70)

    # Convert symbols to Ensembl IDs
    print(f"\n🔍 Converting {len(gene_list)} gene identifiers to Ensembl IDs...")
    mapping = convert_to_ensembl_ids(gene_list, db_path)

    found_genes = [k for k, v in mapping.items() if v['found']]
    missing_genes = [k for k, v in mapping.items() if not v['found']]

    print(f"   ✓ Found: {len(found_genes)} genes")
    if missing_genes:
        print(f"   ⚠ Missing: {len(missing_genes)} genes - {', '.join(missing_genes)}")

    # Fetch complete profiles
    print(f"\n📊 Fetching gene data...")
    profiles = []
    for symbol, info in mapping.items():
        if info['found']:
            profile = fetch_complete_profile(info['ensembl_id'], db_path)
            profiles.append(profile)
    print(f"   ✓ Gene information and functions")

    # Analyze protein interaction network
    print(f"\n🧬 Fetching protein interaction network...")
    protein_network = analyze_protein_network(gene_list, mapping, db_path)
    print(f"   ✓ Protein interaction network analysis")
    print(f"   ✓ Direct interactions between your genes")
    print(f"   ✓ Top 5 protein partners for each gene")
    print(f"   ✓ GO terms, expression data, and more")

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'experiment_context': {
            'organism': experiment_context.organism,
            'cell_type': experiment_context.cell_type,
            'treatment': experiment_context.treatment,
            'timepoint': experiment_context.timepoint,
            'comparison': experiment_context.comparison,
            'hypothesis': experiment_context.hypothesis
        },
        'gene_list': gene_list,
        'mapping': mapping,
        'protein_network': protein_network,
        'profiles': profiles,
        'metadata': {
            'fetch_type': 'deterministic',
            'timestamp': datetime.now().isoformat(),
            'genes_requested': len(gene_list),
            'genes_found': len(found_genes),
            'profiles_fetched': len(profiles),
            'proteins_analyzed': protein_network['summary']['protein_coding_with_data'],
            'direct_interactions_found': protein_network['summary']['direct_interactions_found']
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n✅ Saved to: {output_file}")

    return str(output_path)

async def fetch_gene_data_smart(
    gene_list: List[str],
    experiment_context: ExperimentContext,
    db_path: str = "gene_database.sqlite",
    output_file: str = "gene_data.json"
) -> str:
    """
    Smart context-aware fetch (uses LLM agent) - fetches only relevant data
    NOW INCLUDES: Protein interaction network analysis + Multi-ID format support
    Returns: path to saved JSON file
    """
    print("\n" + "="*70)
    print("SMART CONTEXT-AWARE DATA FETCH (LLM Agent)")
    print("="*70)

    deps = FetcherDeps(
        db_path=db_path,
        experiment_context=experiment_context
    )

    # Register tools and get agent
    agent = register_tools()

    context_desc = f"""
Organism: {experiment_context.organism}
Cell type: {experiment_context.cell_type or 'Not specified'}
Treatment: {experiment_context.treatment or 'Not specified'}
Timepoint: {experiment_context.timepoint or 'Not specified'}
Comparison: {experiment_context.comparison}
Hypothesis: {experiment_context.hypothesis or 'Not specified'}
"""

    prompt = f"""You need to collect data for the following genes in the context of this experiment:

GENES: {', '.join(gene_list)}

EXPERIMENT CONTEXT:
{context_desc}

INSTRUCTIONS:
1. First, convert all gene symbols to Ensembl IDs using convert_to_ensembl
2. ALWAYS get basic information AND function for all genes using get_gene_info (this is mandatory)
3. ALWAYS analyse the protein interaction network using get_protein_interactions_network (mandatory)
4. Based on the experiment context, intelligently choose which additional tools to use:

   - Is this an EXPRESSION study? → Use get_gene_expression
   - Is this about PROTEIN INTERACTIONS or SIGNALING? → Use get_protein_interactions
   - Is this about PATHWAYS or BIOLOGICAL PROCESSES? → Use get_gene_go_terms

5. Collect focused, relevant data - but ALWAYS include gene function and protein interactions
6. Organise the data systematically

IMPORTANT: Gene function descriptions and protein interactions are critical context - always fetch them."""

    print("\n🤖 Running LLM agent to collect relevant data...")
    result = await agent.run(prompt, deps=deps)

    print(f"\n✓ Data collection complete")
    print(f"   ✓ Gene information and functions")
    print(f"   ✓ Protein interaction network analysis")
    print(f"   ✓ Context-specific data based on experiment type")

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract the actual result data
    result_data = {}
    if hasattr(result, 'data') and result.data is not None:
        if hasattr(result.data, 'model_dump'):
            result_data = result.data.model_dump()
        else:
            result_data = result.data
    elif hasattr(result, 'output'):
        result_data = {'output': result.output}
    else:
        result_data = {'raw_result': str(result)}

    # Capture message history with tool calls
    messages_data = []
    if hasattr(result, 'all_messages'):
        for msg in result.all_messages():
            msg_dict = {}
            if hasattr(msg, 'model_dump'):
                msg_dict = msg.model_dump()
            else:
                msg_dict = {'content': str(msg)}
            messages_data.append(msg_dict)

    with open(output_path, 'w') as f:
        json.dump({
            'experiment_context': experiment_context.model_dump(),
            'gene_list': gene_list,
            'result_data': result_data,
            'messages': messages_data,
            'metadata': {
                'fetch_type': 'smart_agent',
                'timestamp': datetime.now().isoformat(),
                'genes_requested': len(gene_list)
            }
        }, f, indent=2, default=str)

    print(f"\n✅ Saved to: {output_file}")

    return str(output_path)

# Convenience wrapper
async def fetch_gene_data(
    gene_list: List[str],
    experiment_context: ExperimentContext,
    db_path: str = "gene_database.sqlite",
    output_file: str = "gene_data.json",
    use_agent: bool = False
) -> str:
    """
    Fetch gene data - choose between deterministic (fast) or smart (LLM-based)

    Args:
        use_agent: If True, use LLM agent for context-aware fetching.
                   If False, fetch all data deterministically (faster, no LLM needed)
    """
    if use_agent:
        return await fetch_gene_data_smart(gene_list, experiment_context, db_path, output_file)
    else:
        return fetch_gene_data_deterministic(gene_list, experiment_context, db_path, output_file)
