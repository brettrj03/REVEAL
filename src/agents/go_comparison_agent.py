"""
GO Term Comparison Analysis Agent

Pure data analysis - no LLM calls.
Analyzes GO term overlap across genes using only state.all_gene_data.
"""

from typing import Dict, Any
from collections import defaultdict
from datetime import datetime


def _normalize_category(namespace: str, aspect: str) -> str:
    """
    Normalize GO term category from various formats.

    Args:
        namespace: GO namespace (e.g., 'molecular_function')
        aspect: GO aspect (e.g., 'F', 'P', 'C')

    Returns:
        Normalized category name
    """
    # Map aspect codes to full names
    aspect_map = {
        'F': 'molecular_function',
        'P': 'biological_process',
        'C': 'cellular_component'
    }

    # Try namespace first
    if namespace:
        return namespace.lower()

    # Fall back to aspect
    if aspect in aspect_map:
        return aspect_map[aspect]

    return 'unknown'


def analyze_go_term_overlap(all_gene_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze GO term overlap across all genes.

    Args:
        all_gene_data: Dictionary of gene data from state

    Returns:
        Structured analysis results
    """

    # Extract genes with GO terms
    genes_analyzed = []
    gene_go_terms = {}  # gene -> list of GO terms

    for gene_symbol, gene_data in all_gene_data.items():
        go_terms = gene_data.get('go_terms', [])
        if go_terms:
            genes_analyzed.append(gene_symbol)
            gene_go_terms[gene_symbol] = go_terms

    if len(genes_analyzed) < 2:
        return {
            'genes_analyzed': genes_analyzed,
            'shared_go_terms': [],
            'by_category': {
                'molecular_function': [],
                'biological_process': [],
                'cellular_component': []
            },
            'gene_overlap_matrix': [],
            'statistics': {
                'total_genes': len(genes_analyzed),
                'total_go_terms_per_gene': {g: len(gene_go_terms.get(g, [])) for g in genes_analyzed},
                'shared_terms_count': 0,
                'shared_terms_by_category': {
                    'molecular_function': 0,
                    'biological_process': 0,
                    'cellular_component': 0
                },
                'most_common_shared_term': None,
                'average_go_terms_per_gene': sum(len(terms) for terms in gene_go_terms.values()) / len(genes_analyzed) if genes_analyzed else 0
            },
            'note': 'Need at least 2 genes with GO terms for comparison'
        }

    # ========================================================================
    # 1. Find shared GO terms (appearing in 2+ genes)
    # ========================================================================
    go_term_index = defaultdict(lambda: {
        'go_id': None,
        'term': None,
        'category': None,
        'genes': set()
    })

    for gene_symbol, go_terms in gene_go_terms.items():
        for term in go_terms:
            go_id = term.get('go_id', '')
            term_name = term.get('name', term.get('term', ''))
            namespace = term.get('namespace', '')
            aspect = term.get('aspect', '')

            if go_id:
                go_term_index[go_id]['go_id'] = go_id
                go_term_index[go_id]['term'] = term_name
                go_term_index[go_id]['category'] = _normalize_category(namespace, aspect)
                go_term_index[go_id]['genes'].add(gene_symbol)

    # Filter to terms appearing in 2+ genes
    shared_terms_list = []
    for go_id, info in go_term_index.items():
        if len(info['genes']) >= 2:
            shared_terms_list.append({
                'go_id': info['go_id'],
                'term': info['term'],
                'category': info['category'],
                'shared_by': sorted(list(info['genes'])),
                'gene_count': len(info['genes'])
            })

    # Sort by gene_count descending
    shared_terms_list.sort(key=lambda x: x['gene_count'], reverse=True)

    # ========================================================================
    # 2. Categorize by GO type
    # ========================================================================
    by_category = {
        'molecular_function': [],
        'biological_process': [],
        'cellular_component': []
    }

    for term in shared_terms_list:
        category = term['category']
        if category in by_category:
            by_category[category].append(term)

    # ========================================================================
    # 3. Gene pair overlap matrix
    # ========================================================================
    gene_list = sorted(genes_analyzed)
    gene_overlap_matrix = []

    for i, gene_a in enumerate(gene_list):
        for gene_b in gene_list[i+1:]:
            # Get GO IDs for each gene
            go_ids_a = {term.get('go_id') for term in gene_go_terms[gene_a] if term.get('go_id')}
            go_ids_b = {term.get('go_id') for term in gene_go_terms[gene_b] if term.get('go_id')}

            # Calculate overlap
            shared_go_ids = go_ids_a & go_ids_b
            shared_count = len(shared_go_ids)

            if shared_count > 0:
                total_unique = len(go_ids_a | go_ids_b)
                overlap_percentage = (shared_count / total_unique * 100) if total_unique > 0 else 0

                gene_overlap_matrix.append({
                    'gene_a': gene_a,
                    'gene_b': gene_b,
                    'shared_go_count': shared_count,
                    'gene_a_total': len(go_ids_a),
                    'gene_b_total': len(go_ids_b),
                    'overlap_percentage': round(overlap_percentage, 1)
                })

    # Sort by shared count descending
    gene_overlap_matrix.sort(key=lambda x: x['shared_go_count'], reverse=True)

    # ========================================================================
    # 4. Statistics
    # ========================================================================
    total_go_terms_per_gene = {
        gene: len(gene_go_terms[gene]) for gene in genes_analyzed
    }

    shared_terms_by_category = {
        'molecular_function': len(by_category['molecular_function']),
        'biological_process': len(by_category['biological_process']),
        'cellular_component': len(by_category['cellular_component'])
    }

    most_common_shared_term = None
    if shared_terms_list:
        most_common = shared_terms_list[0]
        most_common_shared_term = {
            'go_id': most_common['go_id'],
            'term': most_common['term'],
            'gene_count': most_common['gene_count']
        }

    average_go_terms = sum(total_go_terms_per_gene.values()) / len(genes_analyzed)

    statistics = {
        'total_genes': len(genes_analyzed),
        'total_go_terms_per_gene': total_go_terms_per_gene,
        'shared_terms_count': len(shared_terms_list),
        'shared_terms_by_category': shared_terms_by_category,
        'most_common_shared_term': most_common_shared_term,
        'average_go_terms_per_gene': round(average_go_terms, 2)
    }

    # ========================================================================
    # Return complete analysis
    # ========================================================================
    return {
        'genes_analyzed': genes_analyzed,
        'shared_go_terms': shared_terms_list,
        'by_category': by_category,
        'gene_overlap_matrix': gene_overlap_matrix,
        'statistics': statistics,
        'analysis_timestamp': datetime.now().isoformat()
    }
