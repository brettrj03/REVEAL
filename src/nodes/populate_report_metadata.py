"""
Node: Populate Report Metadata

Populates ReportMetadata and BiologicalSynthesis components after all genes
have been processed. This happens right before FinalSummary.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from datetime import datetime
from pydantic_graph.nodes import BaseNode, GraphRunContext
from src.graph.state import GeneState
from src.models.report_components import ReportMetadata
import time

if TYPE_CHECKING:
    from src.nodes.build_literature_query_plan import BuildLiteratureQueryPlan


@dataclass
class PopulateReportMetadata(BaseNode[GeneState]):
    """
    Populates report metadata and cross-gene synthesis.

    This node:
    1. Creates ReportMetadata from gene mapping statistics
    2. Generates BiologicalSynthesis using LLM (if in interpreted mode)
    3. Routes to AnalyzeNetworkOverlap (analysis runs before report generation)
    """

    async def run(self, ctx: GraphRunContext[GeneState]) -> BuildLiteratureQueryPlan:
        from src.nodes.build_literature_query_plan import BuildLiteratureQueryPlan

        print(f"\n{'='*70}")
        print(f"NODE: Populate Report Metadata")
        print(f"{'='*70}")

        start_time = time.time()

        # ========================================================================
        # 1. CREATE REPORT METADATA
        # ========================================================================
        print("Creating report metadata...")

        # Collect gene mapping statistics
        direct_matches = []
        mapped_via_correction = []
        not_found = []

        for gene, mapping_info in ctx.state.gene_mapping.items():
            if mapping_info.get('found', False):
                official_symbol = mapping_info.get('official_symbol', gene)
                if gene == official_symbol:
                    direct_matches.append(gene)
                else:
                    mapped_via_correction.append({
                        'from': gene,
                        'to': official_symbol,
                        'note': mapping_info.get('lookup_tier', 'unknown')
                    })
            else:
                not_found_entry = {'gene': gene}
                if 'suggestion' in mapping_info:
                    not_found_entry['suggestion'] = mapping_info['suggestion']
                not_found.append(not_found_entry)

        # Get total genes
        input_genes = len(ctx.state.genes_to_process) + len(ctx.state.gene_mapping)
        unique_genes = len(ctx.state.gene_mapping)
        successfully_mapped = len(direct_matches) + len(mapped_via_correction)

        # Create metadata
        ctx.state.report_metadata = ReportMetadata(
            query=ctx.state.user_query,
            date=datetime.now(),
            organism=ctx.state.experiment_context.organism if ctx.state.experiment_context else "human",
            comparison=ctx.state.experiment_context.comparison if ctx.state.experiment_context else None,
            cell_type=ctx.state.experiment_context.cell_type if ctx.state.experiment_context else None,
            treatment=ctx.state.experiment_context.treatment if ctx.state.experiment_context else None,
            hypothesis=ctx.state.experiment_context.hypothesis if ctx.state.experiment_context else None,
            input_genes=input_genes,
            unique_genes=unique_genes,
            successfully_mapped=successfully_mapped,
            direct_matches=direct_matches,
            mapped_via_correction=mapped_via_correction,
            not_found=not_found
        )

        print(f"  ✓ Metadata created: {successfully_mapped}/{unique_genes} genes mapped")

        execution_time = time.time() - start_time
        print(f"\n✓ Report metadata populated")
        print(f"  Execution time: {execution_time:.2f}s")

        ctx.state.log_node_execution('populate_report_metadata', execution_time)
        ctx.state.cross_gene_analysis_complete = True

        return BuildLiteratureQueryPlan()
