"""
Validate the GO Term Overlap Interpretation.

Runs after InterpretGoPatterns to validate the AI-generated GO interpretation
against the same source data that was available to the interpreter.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.graph.state import GeneState
from src.nodes.interpret_go_patterns import _enrich_shared_terms
from src.validation.unified_validator import validate_section
from src.validation_config.validation_config import VALIDATION_ENABLED


def _build_go_source_context(
    enriched_terms: List[Dict[str, Any]],
    overlap_stats: Dict[str, Any],
    genes: List[str],
    experimental_context: Optional[Dict[str, Any]],
) -> str:
    """Build the source context string for GO interpretation validation.

    Contains ONLY what InterpretGoPatterns had access to.
    """
    parts: List[str] = []

    # Overlap statistics
    parts.append("=== OVERLAP STATISTICS ===")
    parts.append(f"Total shared terms: {len(enriched_terms)}")
    total_terms = overlap_stats.get("total_terms") or overlap_stats.get("total_go_terms") or 0
    parts.append(f"Total GO terms across gene set: {total_terms}")
    ratio = overlap_stats.get("overlap_ratio") or overlap_stats.get("shared_total_ratio") or 0
    try:
        ratio_display = f"{float(ratio):.3f}"
    except (TypeError, ValueError):
        ratio_display = str(ratio)
    parts.append(f"Overlap ratio: {ratio_display}")

    # Genes in the analysis
    parts.append(f"\n=== GENES ANALYSED ===")
    parts.append(f"Genes: {', '.join(genes)}")

    # Experimental context
    if experimental_context:
        parts.append(f"\n=== EXPERIMENTAL CONTEXT ===")
        parts.append(json.dumps(experimental_context, indent=2))

    # Enriched shared terms (sorted by depth)
    parts.append(f"\n=== SHARED GO TERMS (sorted by depth, most specific first) ===")
    for term in enriched_terms:
        term_name = term.get("term_name", "Unknown")
        go_id = term.get("go_id", "N/A")
        namespace = term.get("namespace", "unknown")
        depth = term.get("depth", 0)
        definition = term.get("definition", "")
        genes_list = term.get("genes", [])

        parts.append(f"\n  {term_name} ({go_id})")
        parts.append(f"    Namespace: {namespace}")
        parts.append(f"    Depth: {depth}")
        parts.append(f"    Genes: {', '.join(genes_list) if genes_list else 'None'}")
        if definition:
            parts.append(f"    Definition: {definition}")

    return "\n".join(parts)


@dataclass
class ValidateGoInterpretation(BaseNode[GeneState]):
    """Validate the GO term overlap interpretation against source data."""

    async def run(self, ctx: GraphRunContext[GeneState]) -> "GenerateCrossGeneSynthesis":
        from src.nodes.generate_cross_gene_synthesis import GenerateCrossGeneSynthesis

        _t0 = time.perf_counter()
        try:
            print(f"\n{'='*70}")
            print("NODE: Validate GO Interpretation")
            print(f"{'='*70}")

            state = ctx.state

            # Skip conditions
            if not VALIDATION_ENABLED:
                print("  Validation disabled in config, skipping")
                return GenerateCrossGeneSynthesis()

            if state.output_mode == "factual":
                print("  Skipping validation in factual mode")
                return GenerateCrossGeneSynthesis()

            analysis = state.go_comparison_analysis or {}
            interpretation = analysis.get("interpretation")

            if not interpretation or not interpretation.strip():
                print("  No GO interpretation to validate")
                return GenerateCrossGeneSynthesis()

            # Get shared terms and enrich them (same as InterpretGoPatterns)
            shared_terms_raw = (analysis.get("shared_terms") or [])[:100]
            if not shared_terms_raw:
                print("  No shared terms to build source context")
                return GenerateCrossGeneSynthesis()

            enriched_terms = _enrich_shared_terms(shared_terms_raw, state.db_path)

            # Build overlap stats
            overlap_stats = analysis.get("overlap_stats", {})
            if not overlap_stats:
                # Reconstruct minimal stats if missing
                overlap_stats = {
                    "total_shared": len(shared_terms_raw),
                    "total_terms": analysis.get("total_terms", 0),
                    "overlap_ratio": analysis.get("overlap_ratio", 0),
                }

            # Get genes
            genes = sorted(state.gene_profiles.keys()) if state.gene_profiles else []

            # Get experimental context
            exp_context = (
                state.experiment_context.model_dump()
                if state.experiment_context and hasattr(state.experiment_context, "model_dump")
                else None
            )

            # Build source context string
            source_context = _build_go_source_context(
                enriched_terms,
                overlap_stats,
                genes,
                exp_context,
            )

            # Build gene list string for multi-gene validation
            gene_list = ", ".join(genes)

            # Validate the interpretation
            result = await validate_section(
                section_name="go_interpretation",
                section_text=interpretation,
                gene_symbol=gene_list,  # Comma-separated list of genes being validated
                state_data={},  # Not used since we provide source_context_override
                state_ref=state,
                node_name="ValidateGoInterpretation",
                source_context_override=source_context,
                lite_mode=False,
            )

            # Store results in cross-gene validation fields
            state.cross_gene_validation_results["go_interpretation"] = result
            accuracy = result.get("final_accuracy") or 0.0
            state.cross_gene_accuracy_scores["go_interpretation"] = accuracy

            # Update interpretation if corrected
            final_text = result.get("final_summary", "")
            if final_text and final_text.strip() != interpretation.strip():
                state.go_comparison_analysis["interpretation"] = final_text
                print(f"  ✓ GO interpretation corrected")

            status = result.get("status", "unknown")
            print(f"✓ GO interpretation validation complete — accuracy: {accuracy:.1f}%, status: {status}")

            return GenerateCrossGeneSynthesis()

        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )
