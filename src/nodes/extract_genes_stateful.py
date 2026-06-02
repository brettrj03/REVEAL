"""
Node: Extract Genes (Stateful)

Extracts ALL genes and puts them in the processing queue.
BaseNode version for stateful pipeline.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
from src.graph.state import GeneState, _accumulate_tokens
from src.agents.gene_extractor import extract_genes_with_context
from src.utils.gene_normalizer import normalize_gene_list
import time

if TYPE_CHECKING:
    from src.nodes.fetch_gene_data import FetchAllGeneData

@dataclass
class ExtractGenesStateful(BaseNode[GeneState]):
    """
    Extracts genes from user query and populates the processing queue.

    Returns FetchGeneData to start processing first gene.
    """

    async def run(self, ctx: GraphRunContext[GeneState]) -> FetchAllGeneData:
        # Import here to avoid circular dependencies
        from src.nodes.fetch_gene_data import FetchAllGeneData, resolve_all_gene_identifiers

        print(f"\n{'='*70}")
        print(f"NODE: Extract Genes")
        print(f"{'='*70}")
        print(f"Query: {ctx.state.user_query}")

        start_time = time.time()

        # ================================================================
        # BYPASS: Skip extraction if genes already provided (CSV/manual input)
        # ================================================================
        if ctx.state.genes_to_process:
            print("\n  ✓ Genes already populated (CSV or manual input) — skipping LLM extraction")
            print(f"  Genes provided: {len(ctx.state.genes_to_process)}")
            for gene in ctx.state.genes_to_process:
                print(f"    - {gene}")
            print(f"  Output mode: {ctx.state.output_mode}")
            print(f"  Execution time: {time.time() - start_time:.2f}s")
            return FetchAllGeneData()

        try:
            # ================================================================
            # STAGE 1: Enhanced LLM Extraction
            # ================================================================
            print("\n  Stage 1: LLM Extraction...")
            result = await extract_genes_with_context(
                ctx.state.user_query,
                state=ctx.state,
                node_name=self.__class__.__name__
            )
            ctx.state.extraction_confidence = result.confidence

            # Log what was extracted by category
            if result.gene_symbols:
                print(f"    Symbols: {', '.join(result.gene_symbols)}")
            if result.gene_ensembl_ids:
                print(f"    Ensembl IDs: {', '.join(result.gene_ensembl_ids)}")
            if result.gene_full_names:
                print(f"    Full names: {', '.join(result.gene_full_names)}")
            if result.gene_aliases:
                print(f"    Aliases: {', '.join(result.gene_aliases)}")

            # ================================================================
            # STAGE 2: Database Resolution
            # ================================================================
            print("\n  Stage 2: Database Resolution...")
            resolved_genes, unresolved = await resolve_all_gene_identifiers(
                result, ctx.state.db_path
            )

            # Store unresolved identifiers for user feedback
            ctx.state.unresolved_identifiers = [
                {'identifier': u.identifier, 'type': u.identifier_type, 'reason': u.reason}
                for u in unresolved
            ]

            # Populate queue with resolved official symbols only
            resolved_symbols = [g.official_symbol for g in resolved_genes]

            # Normalize gene list (for duplicate tracking)
            gene_mapping = normalize_gene_list(resolved_symbols)
            ctx.state.genes_to_process = gene_mapping.unique_genes
            ctx.state.gene_name_mapping = gene_mapping

            # Update experiment context
            if not ctx.state.experiment_context.organism and result.organism:
                ctx.state.experiment_context.organism = result.organism

            if not ctx.state.experiment_context.cell_type and result.cell_type:
                ctx.state.experiment_context.cell_type = result.cell_type

            if not ctx.state.experiment_context.treatment and result.treatment:
                ctx.state.experiment_context.treatment = result.treatment

            if not ctx.state.experiment_context.timepoint and result.timepoint:
                ctx.state.experiment_context.timepoint = result.timepoint

            if not ctx.state.experiment_context.comparison and result.comparison:
                ctx.state.experiment_context.comparison = result.comparison

            if not ctx.state.experiment_context.hypothesis and result.hypothesis:
                ctx.state.experiment_context.hypothesis = result.hypothesis

            # Log execution
            execution_time = time.time() - start_time
            ctx.state.log_node_execution('extract_genes', execution_time)

            # Report results
            print(f"\n✓ Extraction & Resolution complete!")
            print(f"  Genes resolved: {len(ctx.state.genes_to_process)}")
            for gene in ctx.state.genes_to_process:
                print(f"    - {gene}")

            if unresolved:
                print(f"\n  ⚠ Unresolved identifiers: {len(unresolved)}")
                for u in unresolved:
                    print(f"    • {u.identifier} ({u.identifier_type}): {u.reason}")

            if ctx.state.gene_name_mapping and ctx.state.gene_name_mapping.duplicates:
                print("\n  ⚠ Duplicates removed before analysis:")
                for gene, count in ctx.state.gene_name_mapping.duplicates.items():
                    print(f"    • {gene} (appeared {count} times)")

            print(f"\n  Experimental Context:")
            print(f"    Organism: {ctx.state.experiment_context.organism}")
            if ctx.state.experiment_context.cell_type:
                print(f"    Cell type: {ctx.state.experiment_context.cell_type}")
            if ctx.state.experiment_context.treatment:
                print(f"    Treatment: {ctx.state.experiment_context.treatment}")
            if ctx.state.experiment_context.timepoint:
                print(f"    Timepoint: {ctx.state.experiment_context.timepoint}")
            if ctx.state.experiment_context.comparison:
                print(f"    Comparison: {ctx.state.experiment_context.comparison}")
            if ctx.state.experiment_context.hypothesis:
                print(f"    Hypothesis: {ctx.state.experiment_context.hypothesis}")

            print(f"\n  Confidence: {ctx.state.extraction_confidence:.2f}")
            print(f"  Execution time: {execution_time:.2f}s")

            if not ctx.state.genes_to_process:
                ctx.state.error = "No genes found in query"
                print(f"\n⚠ WARNING: {ctx.state.error}")

        except Exception as e:
            ctx.state.error = f"Gene extraction failed: {str(e)}"
            print(f"\n✗ ERROR: {ctx.state.error}")
            import traceback
            traceback.print_exc()

        # Always go to FetchAllGeneData next
        return FetchAllGeneData()
