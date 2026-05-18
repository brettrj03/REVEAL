"""Phase 3 interpretation of network overlap results."""

from __future__ import annotations

from dataclasses import dataclass
import time

from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.agents.network_interpretation_agent import (
    NetworkInterpretationContext,
    generate_network_interpretation,
)
from src.agents.gene_summarizer import (
    generate_network_interpretation as generate_single_gene_network_interpretation,
)
from src.graph.state import GeneState
from src.utils.tracing import trace_event


@dataclass
class InterpretNetworkOverlap(BaseNode[GeneState]):
    """Generate narrative for protein interaction overlap in interpreted mode."""

    async def run(self, ctx: GraphRunContext[GeneState]) -> "ValidateNetworkInterpretation":
        _t0 = time.perf_counter()
        try:
            from src.nodes.validate_network_interpretation import ValidateNetworkInterpretation

            print(f"\n{'='*70}")
            print("NODE: Interpret Network Overlap")
            print(f"{'='*70}")

            analysis = ctx.state.network_overlap_analysis or {}
            hub_proteins = analysis.get('hub_proteins', [])
            direct_interactions = analysis.get('direct_interactions', [])
            shared_partners = analysis.get('shared_partners', [])

            # Get query genes
            query_genes = sorted(ctx.state.gene_profiles.keys()) if ctx.state.gene_profiles else []
            is_single_gene = len(query_genes) == 1

            if ctx.state.output_mode == "factual":
                print("Skipping network interpretation (factual mode)")
                message = (
                    "Network interpretation skipped in factual mode."
                )
                ctx.state.network_overlap_analysis['interpretation'] = message
                return ValidateNetworkInterpretation()

            # For single-gene queries, use the individual gene network interpretation
            if is_single_gene:
                print("Single-gene query: using individual gene network interpretation")
                gene_symbol = query_genes[0]
                gene_data = ctx.state.all_gene_data.get(gene_symbol, {})

                if not gene_data:
                    # Try case-insensitive lookup
                    for k, v in ctx.state.all_gene_data.items():
                        if k.upper() == gene_symbol.upper():
                            gene_data = v
                            break

                # Build gene_data dict for single-gene interpreter
                single_gene_data = {
                    "gene_symbol": gene_symbol,
                    "interactions": gene_data.get("interactions", []),
                }

                try:
                    description = await generate_single_gene_network_interpretation(
                        single_gene_data,
                        state=ctx.state,
                        node_name="InterpretNetworkOverlap",
                    )
                    ctx.state.network_overlap_analysis['interpretation'] = description
                    print("✓ Single-gene network interpretation generated")
                except Exception as exc:
                    print(f"⚠️  Failed to interpret single-gene network: {exc}")
                    ctx.state.network_overlap_analysis['interpretation'] = (
                        f"Network interpretation could not be generated for {gene_symbol}."
                    )

                return ValidateNetworkInterpretation()

            # Multi-gene case: check for sufficient data
            if not hub_proteins and not shared_partners and len(direct_interactions) < 2:
                print("Skipping network interpretation (insufficient network overlap)")
                message = (
                    "No significant protein interaction overlap was detected between the "
                    "query genes."
                )
                ctx.state.network_overlap_analysis['interpretation'] = message
                return ValidateNetworkInterpretation()

            try:
                context = NetworkInterpretationContext(
                    genes=query_genes,
                    hub_proteins=hub_proteins,
                    direct_interactions=direct_interactions,
                    network_stats=analysis.get('network_stats'),
                    experimental_context=
                        ctx.state.experiment_context.model_dump()
                        if ctx.state.experiment_context
                        else None,
                )
                description = await generate_network_interpretation(
                    context,
                    state=ctx.state,
                    node_name="InterpretNetworkOverlap",
                )
                ctx.state.network_overlap_analysis['interpretation'] = description
                print("✓ Network interpretation generated")
                trace_event(
                    "interpretation.network_overlap",
                    genes=context.genes,
                    analysis_keys=sorted(analysis.keys()),
                    state_inputs=['network_overlap_analysis']
                )
            except Exception as exc:  # pragma: no cover - best effort
                print(f"⚠️  Failed to interpret network overlap: {exc}")

            return ValidateNetworkInterpretation()
        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )
