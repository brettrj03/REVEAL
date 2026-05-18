"""
Validate the Network Interpretation.

Runs after InterpretNetworkOverlap to validate the AI-generated network interpretation
against the same source data that was available to the interpreter.

The interpreter had access to:
- Query genes being analysed
- Hub proteins (shared partners) with protein name, genes connected, avg_confidence
- Direct interactions between query genes with gene_a, gene_b, score
- Network statistics: total_interactions, shared_partners count, avg_confidence
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.graph.state import GeneState
from src.validation.unified_validator import validate_section
from src.validation_config.validation_config import VALIDATION_ENABLED


def _build_network_source_context(
    hub_proteins: List[Dict[str, Any]],
    direct_interactions: List[Dict[str, Any]],
    network_stats: Dict[str, Any],
    genes: List[str],
    experimental_context: Optional[Dict[str, Any]],
) -> str:
    """Build the source context string for network interpretation validation.

    Contains ONLY what InterpretNetworkOverlap had access to.
    """
    parts: List[str] = []

    # Query genes
    parts.append("=== QUERY GENES ===")
    parts.append(f"Genes analysed: {', '.join(genes)}")

    # Network statistics
    parts.append("\n=== NETWORK STATISTICS ===")
    total_int = network_stats.get("total_interactions")
    if total_int is not None:
        parts.append(f"Total interactions: {total_int}")
    shared_count = network_stats.get("shared_partners", network_stats.get("shared_partner_count"))
    if shared_count is not None:
        parts.append(f"Shared partners count: {shared_count}")
    avg_conf = network_stats.get("avg_confidence")
    if avg_conf is not None:
        parts.append(f"Average confidence: {avg_conf:.3f}")

    # Direct gene-gene interactions
    if direct_interactions:
        parts.append(f"\n=== DIRECT GENE-GENE INTERACTIONS ({len(direct_interactions)} pairs) ===")
        for inter in direct_interactions:
            if isinstance(inter, dict):
                gene_a = inter.get("gene_a", "?")
                gene_b = inter.get("gene_b", "?")
                score = inter.get("score", 0)
                # Normalize score if needed
                if score > 1:
                    score = score / 1000.0
                parts.append(f"  {gene_a} ↔ {gene_b}: score {score:.3f}")

    # Hub proteins / shared partners
    if hub_proteins:
        parts.append(f"\n=== SHARED INTERACTION PARTNERS ({len(hub_proteins)} partners) ===")
        # Show all for validation (up to 50 for reasonable context size)
        for hp in hub_proteins[:50]:
            if isinstance(hp, dict):
                protein = hp.get("protein", "Unknown")
                genes_connected = hp.get("genes", [])
                avg_conf = hp.get("avg_confidence")
                conf_str = f" (avg_confidence: {avg_conf:.3f})" if avg_conf is not None else ""
                parts.append(f"  - {protein}: connects {', '.join(genes_connected)}{conf_str}")
        if len(hub_proteins) > 50:
            parts.append(f"  ... and {len(hub_proteins) - 50} more partners")

    # Experimental context (if available)
    if experimental_context:
        import json
        parts.append(f"\n=== EXPERIMENTAL CONTEXT ===")
        parts.append(json.dumps(experimental_context, indent=2))

    return "\n".join(parts)


@dataclass
class ValidateNetworkInterpretation(BaseNode[GeneState]):
    """Validate the network interpretation against source data."""

    async def run(self, ctx: GraphRunContext[GeneState]) -> "InterpretGoPatterns":
        from src.nodes.interpret_go_patterns import InterpretGoPatterns

        _t0 = time.perf_counter()
        try:
            print(f"\n{'='*70}")
            print("NODE: Validate Network Interpretation")
            print(f"{'='*70}")

            state = ctx.state

            # Skip conditions
            if not VALIDATION_ENABLED:
                print("  Validation disabled in config, skipping")
                return InterpretGoPatterns()

            if state.output_mode == "factual":
                print("  Skipping validation in factual mode")
                return InterpretGoPatterns()

            analysis = state.network_overlap_analysis or {}
            interpretation = analysis.get("interpretation")

            if not interpretation or not interpretation.strip():
                print("  No network interpretation to validate")
                return InterpretGoPatterns()

            # Get the data that the interpreter had access to
            hub_proteins = analysis.get("hub_proteins", analysis.get("shared_partners", []))
            direct_interactions = analysis.get("direct_interactions", [])
            network_stats = analysis.get("network_stats", analysis.get("network_statistics", {}))

            if not hub_proteins and not direct_interactions:
                print("  No network data to validate against")
                return InterpretGoPatterns()

            # Get genes
            genes = analysis.get("genes_analyzed", [])
            if not genes:
                genes = sorted(state.gene_profiles.keys()) if state.gene_profiles else []

            # Get experimental context
            exp_context = (
                state.experiment_context.model_dump()
                if state.experiment_context and hasattr(state.experiment_context, "model_dump")
                else None
            )

            # Build source context string
            source_context = _build_network_source_context(
                hub_proteins,
                direct_interactions,
                network_stats,
                genes,
                exp_context,
            )

            # Build gene list string for multi-gene validation
            gene_list = ", ".join(genes)

            # Validate the interpretation
            result = await validate_section(
                section_name="network_interpretation",
                section_text=interpretation,
                gene_symbol=gene_list,  # Comma-separated list of genes being validated
                state_data={},  # Not used since we provide source_context_override
                state_ref=state,
                node_name="ValidateNetworkInterpretation",
                source_context_override=source_context,
                lite_mode=False,
            )

            # Store results in cross-gene validation fields
            state.cross_gene_validation_results["network_interpretation"] = result
            accuracy = result.get("final_accuracy") or 0.0
            state.cross_gene_accuracy_scores["network_interpretation"] = accuracy

            # Update interpretation if corrected
            final_text = result.get("final_summary", "")
            if final_text and final_text.strip() != interpretation.strip():
                state.network_overlap_analysis["interpretation"] = final_text
                print(f"  ✓ Network interpretation corrected")

            status = result.get("status", "unknown")
            print(f"✓ Network interpretation validation complete — accuracy: {accuracy:.1f}%, status: {status}")

            return InterpretGoPatterns()

        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )
