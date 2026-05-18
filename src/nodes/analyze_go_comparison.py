"""Pipeline node for cross-gene GO-term overlap analysis."""

from __future__ import annotations
from typing import TYPE_CHECKING

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
import time

from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.agents.go_comparison_agent import analyze_go_term_overlap
from src.graph.state import GeneState

if TYPE_CHECKING:
    from src.nodes.populate_report_metadata import PopulateReportMetadata


@dataclass
class AnalyzeGoComparison(BaseNode[GeneState]):
    """
    Analyse GO term overlap across all genes in the dataset.

    Reads from:
        state.all_gene_data - GO terms for each gene

    Writes to:
        state.go_comparison_analysis - complete GO comparison results

    Returns:
        PopulateReportMetadata - continues Phase 2
    """

    async def run(
        self,
        ctx: GraphRunContext[GeneState]
    ) -> PopulateReportMetadata:
        """
        Run GO term comparison analysis.

        Args:
            ctx: Graph context with current state

        Returns:
            End node to terminate pipeline
        """

        print("\n" + "="*80)
        print("GO TERM COMPARISON ANALYSIS")
        print("="*80)

        # Import here to avoid circular dependencies
        from src.nodes.populate_report_metadata import PopulateReportMetadata

        _t0 = time.perf_counter()

        try:
            # Check if we have gene data
            if not ctx.state.all_gene_data:
                print("⚠️  No gene data available for GO term analysis")
                ctx.state.go_comparison_analysis = {
                    'genes_analyzed': [],
                    'error': 'No gene data available',
                    'analysis_timestamp': None
                }
                return PopulateReportMetadata()

            # Check if genes have GO term data
            genes_with_go_terms = {
                gene: data for gene, data in ctx.state.all_gene_data.items()
                if data.get('go_terms')
            }

            if not genes_with_go_terms:
                print("⚠️  No GO term data available for any genes")
                ctx.state.go_comparison_analysis = {
                    'genes_analyzed': list(ctx.state.all_gene_data.keys()),
                    'error': 'No GO term data available',
                    'analysis_timestamp': None
                }
                return PopulateReportMetadata()

            print(f"Analysing GO terms for {len(genes_with_go_terms)} genes...")

            try:
                analysis = analyze_go_term_overlap(genes_with_go_terms)

                shared_terms = [
                    {
                        'go_id': term.get('go_id'),
                        'name': term.get('term') or term.get('name'),
                        'category': term.get('category'),
                        'genes': term.get('shared_by', []),
                    }
                    for term in analysis.get('shared_go_terms', [])
                ]

                similarity_matrix = _build_similarity_matrix(
                    analysis.get('gene_overlap_matrix', []),
                    list(genes_with_go_terms.keys())
                )

                overlap_stats = _summarize_overlap_stats(genes_with_go_terms, shared_terms)

                legacy_terms = [
                    {
                        'go_id': term.get('go_id'),
                        'go_term': term.get('name'),
                        'aspect': term.get('category'),
                        'gene_count': len(term.get('genes', [])),
                    }
                    for term in shared_terms
                ]

                total_go_terms = sum(
                    len(data.get('go_terms', []))
                    for data in genes_with_go_terms.values()
                )
                gene_count = len(genes_with_go_terms)

                legacy_stats = {
                    'total_go_terms': total_go_terms,
                    'shared_go_terms': len(shared_terms),
                    'avg_go_terms_per_gene': (total_go_terms / gene_count) if gene_count else 0,
                    'shared_total_ratio': (len(shared_terms) / total_go_terms) if total_go_terms else 0,
                }

                timestamp = datetime.now().isoformat()
                ctx.state.go_comparison_analysis = {
                    'shared_terms': shared_terms,
                    'shared_go_terms': legacy_terms,
                    'overlap_stats': overlap_stats,
                    'similarity_matrix': similarity_matrix,
                    'overlap_matrix': similarity_matrix,
                    'statistics': legacy_stats,
                    'genes_analyzed': list(genes_with_go_terms.keys()),
                    'timestamp': timestamp,
                    'analysis_timestamp': timestamp,
                }

                print(f"\n✅ GO comparison complete: {len(shared_terms)} shared terms")

            except Exception as e:
                print(f"❌ GO comparison analysis failed: {e}")
                import traceback

                traceback.print_exc()
                timestamp = datetime.now().isoformat()
                ctx.state.go_comparison_analysis = {
                    'error': str(e),
                    'timestamp': timestamp,
                    'analysis_timestamp': timestamp,
                    'genes_analyzed': list(ctx.state.all_gene_data.keys()),
                }

            print("="*80 + "\n")

            return PopulateReportMetadata()
        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )


def _build_similarity_matrix(
    gene_overlap_matrix: list,
    gene_names: list
) -> dict:
    """
    Convert a flat list of gene-pair overlap records into a nested dict matrix.

    Returns a dict like:
    {
        'MED12': {'MED12': 1.0, 'F8A3': 0.12},
        'F8A3':  {'MED12': 0.12, 'F8A3': 1.0},
        ...
    }
    Values are Jaccard-like overlap (shared / union) ranging 0–1.
    Diagonal is always 1.0.
    """
    # Build a lookup from (gene_a, gene_b) → overlap_percentage
    pair_scores: dict = {}
    for entry in gene_overlap_matrix:
        ga = entry.get('gene_a', '')
        gb = entry.get('gene_b', '')
        score = (entry.get('overlap_percentage') or 0) / 100.0  # convert % to 0-1
        pair_scores[(ga, gb)] = score
        pair_scores[(gb, ga)] = score  # symmetric

    matrix: dict = {}
    for gene_a in gene_names:
        matrix[gene_a] = {}
        for gene_b in gene_names:
            if gene_a == gene_b:
                matrix[gene_a][gene_b] = 1.0
            else:
                matrix[gene_a][gene_b] = pair_scores.get((gene_a, gene_b), 0.0)
    return matrix


def _summarize_overlap_stats(
    genes_with_go_terms: dict,
    shared_terms: list
) -> dict:
    """
    Summarise GO term overlap statistics across genes.

    Returns a dict with counts and percentages useful for the report.
    """
    gene_count = len(genes_with_go_terms)
    total_go_terms = sum(
        len(data.get('go_terms', []))
        for data in genes_with_go_terms.values()
    )
    shared_count = len(shared_terms)

    # Count how many genes share each term
    sharing_counts = [len(t.get('genes', [])) for t in shared_terms]
    fully_shared = sum(1 for c in sharing_counts if c == gene_count)

    return {
        'total_go_terms': total_go_terms,
        'shared_go_terms': shared_count,
        'fully_shared_terms': fully_shared,  # shared by ALL genes
        'avg_go_terms_per_gene': round(total_go_terms / gene_count, 1) if gene_count else 0,
        'overlap_ratio': round(shared_count / total_go_terms, 3) if total_go_terms else 0,
    }
