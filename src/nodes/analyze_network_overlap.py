"""Pipeline node for cross-gene network overlap analysis."""

from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
import time

from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.agents.network_overlap_agent import analyze_network_overlap
from src.graph.state import GeneState

if TYPE_CHECKING:
    from src.nodes.analyze_go_comparison import AnalyzeGoComparison


@dataclass
class AnalyzeNetworkOverlap(BaseNode[GeneState]):
    """
    Analyse network overlap across all genes in the dataset.

    Reads from:
        state.all_gene_data - protein interactions for each gene

    Writes to:
        state.network_overlap_analysis - complete network analysis results

    Returns:
        AnalyzeGoComparison - continues to GO term analysis
    """

    async def run(
        self,
        ctx: GraphRunContext[GeneState]
    ) -> AnalyzeGoComparison:
        """
        Run network overlap analysis.

        Args:
            ctx: Graph context with current state

        Returns:
            AnalyzeGoComparison node
        """
        # Import here to avoid circular dependencies
        from src.nodes.analyze_go_comparison import AnalyzeGoComparison

        _t0 = time.perf_counter()

        print("\n" + "="*80)
        print("NETWORK OVERLAP ANALYSIS")
        print("="*80)

        try:
            # Check if we have gene data
            if not ctx.state.all_gene_data:
                print("⚠️  No gene data available for network analysis")
                ctx.state.network_overlap_analysis = {
                    'genes_analyzed': [],
                    'error': 'No gene data available',
                    'analysis_timestamp': datetime.now().isoformat()
                }
                return AnalyzeGoComparison()

            # Check if genes have interaction data
            genes_with_interactions = {
                gene: data for gene, data in ctx.state.all_gene_data.items()
                if data.get('interactions')
            }


            if not genes_with_interactions:
                print("⚠️  No interaction data available for any genes")
                ctx.state.network_overlap_analysis = {
                    'genes_analyzed': list(ctx.state.all_gene_data.keys()),
                    'error': 'No interaction data available',
                    'analysis_timestamp': datetime.now().isoformat()
                }
                return AnalyzeGoComparison()

            print(f"Analysing network overlap for {len(genes_with_interactions)} genes...")

            try:
                raw_analysis = await analyze_network_overlap(
                    genes_with_interactions,
                    state=ctx.state,
                    node_name=self.__class__.__name__
                )

                hub_proteins = [
                    {
                        'protein': partner.get('partner'),
                        'genes': partner.get('shared_by', []),
                        'avg_confidence': round(sum(partner.get('confidence_scores', {}).values()) / len(partner.get('confidence_scores', {})), 3) if partner.get('confidence_scores') else None,
                    }
                    for partner in raw_analysis.get('shared_partners', [])
                ]

                direct_interactions = [
                    {
                        'gene_a': interaction.get('gene_a'),
                        'gene_b': interaction.get('gene_b'),
                        'score': interaction.get('confidence', 0.0),
                        'description': interaction.get('description'),
                    }
                    for interaction in raw_analysis.get('direct_interactions', [])
                ]

                stats = raw_analysis.get('network_statistics', {})
                network_stats = {
                    'total_interactions': stats.get('total_interactions', 0),
                    'shared_partners': stats.get('shared_partner_count', stats.get('shared_partners_count', 0)),
                    'avg_confidence': stats.get('avg_confidence'),
                    'unique_partners': stats.get('unique_partners'),
                    'shared_partners_percentage': stats.get('shared_partners_percentage'),
                    'hub_genes_count': stats.get('hub_genes_count'),
                }

                legacy_stats = {
                    'total_unique_partners': stats.get('unique_partners', 0),
                    'avg_partners_per_gene': (
                        stats.get('total_interactions', 0)
                        / max(stats.get('genes_analyzed', 1), 1)
                    ) if stats.get('genes_analyzed') else 0,
                    'network_density': (stats.get('shared_partners_percentage', 0) or 0) / 100.0,
                    'hub_genes_count': stats.get('hub_genes_count', 0),
                    'shared_partners': stats.get('shared_partner_count', stats.get('shared_partners_count', 0)),
                }

                timestamp = datetime.now().isoformat()
                ctx.state.network_overlap_analysis = {
                    'hub_proteins': hub_proteins,
                    'direct_interactions': direct_interactions,
                    'network_stats': network_stats,
                    'network_statistics': legacy_stats,
                    'shared_partners': hub_proteins,
                    'genes_analyzed': list(genes_with_interactions.keys()),
                    'network_modules': raw_analysis.get('network_modules', []),
                    'timestamp': timestamp,
                    'analysis_timestamp': timestamp,
                }

                print(f"\n✅ Network analysis complete: {len(hub_proteins)} shared partners identified")

            except Exception as e:
                print(f"❌ Network analysis failed: {e}")
                import traceback

                traceback.print_exc()
                timestamp = datetime.now().isoformat()
                ctx.state.network_overlap_analysis = {
                    'error': str(e),
                    'timestamp': timestamp,
                    'analysis_timestamp': timestamp,
                    'genes_analyzed': list(ctx.state.all_gene_data.keys()),
                }

            print("="*80 + "\n")

            return AnalyzeGoComparison()
        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )
