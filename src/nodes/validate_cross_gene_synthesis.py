"""
Validate the Cross-Gene Analysis Synthesis.

Runs after GenerateCrossGeneSynthesis to validate the AI-generated key findings
against the same source data that was available to the synthesis generator.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.graph.state import GeneState
from src.validation.unified_validator import validate_section
from src.validation_config.validation_config import VALIDATION_ENABLED


def _build_synthesis_source_context(
    gene_profiles: Dict[str, Any],
    gene_interpretations: Dict[str, Any],
    network_overlap: Optional[Dict[str, Any]],
    go_comparison: Optional[Dict[str, Any]],
    experimental_context: Optional[Dict[str, Any]],
) -> str:
    """Build the source context string for cross-gene synthesis validation.

    Contains ONLY what GenerateCrossGeneSynthesis had access to.
    """
    parts: List[str] = []

    # Experimental context
    if experimental_context:
        parts.append("=== EXPERIMENTAL CONTEXT ===")
        parts.append(json.dumps(experimental_context, indent=2))

    # Gene profiles summary
    parts.append(f"\n=== GENE PROFILES ({len(gene_profiles)} genes) ===")
    for gene_symbol, profile in sorted(gene_profiles.items()):
        parts.append(f"\n  {gene_symbol}:")
        if isinstance(profile, dict):
            func_summary = profile.get("functional_summary", "")
            if func_summary:
                parts.append(f"    Functional summary: {func_summary[:300]}...")
            mol_funcs = profile.get("molecular_functions", [])
            if mol_funcs:
                func_names = []
                for func in mol_funcs[:3]:
                    if isinstance(func, dict):
                        func_names.append(func.get("name", "Unknown"))
                    else:
                        func_names.append(str(func))
                parts.append(f"    Molecular functions: {', '.join(func_names)}")
        else:
            # Pydantic model
            func_summary = getattr(profile, "functional_summary", "")
            if func_summary:
                parts.append(f"    Functional summary: {func_summary[:300]}...")
            mol_funcs = getattr(profile, "molecular_functions", [])
            if mol_funcs:
                func_names = []
                for func in mol_funcs[:3]:
                    if isinstance(func, dict):
                        func_names.append(func.get("name", "Unknown"))
                    else:
                        func_names.append(str(func))
                parts.append(f"    Molecular functions: {', '.join(func_names)}")

    # Gene interpretations (expression data)
    parts.append(f"\n=== GENE INTERPRETATIONS ===")
    for gene_symbol, interp in sorted(gene_interpretations.items()):
        if isinstance(interp, dict):
            expr = interp.get("expression_interpretation", "")
            if expr:
                parts.append(f"  {gene_symbol} expression: {expr[:200]}...")

    # Network overlap analysis
    if network_overlap:
        parts.append(f"\n=== NETWORK OVERLAP ANALYSIS ===")
        hub_proteins = network_overlap.get("hub_proteins", [])
        if hub_proteins:
            parts.append(f"  Shared interaction partners: {len(hub_proteins)} proteins")
            for hub in hub_proteins[:10]:
                partner = hub.get("protein") or hub.get("partner", "Unknown")
                genes = ", ".join(hub.get("genes", []))
                parts.append(f"    - {partner} interacts with: {genes}")

        direct_interactions = network_overlap.get("direct_interactions", [])
        if direct_interactions:
            parts.append(f"  Direct gene-gene interactions: {len(direct_interactions)} pairs")
            for interaction in direct_interactions[:5]:
                if isinstance(interaction, dict):
                    gene1 = interaction.get("gene1", "")
                    gene2 = interaction.get("gene2", "")
                    parts.append(f"    - {gene1} <-> {gene2}")
                elif isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                    parts.append(f"    - {interaction[0]} <-> {interaction[1]}")

    # GO comparison analysis (excluding interpretation)
    if go_comparison:
        parts.append(f"\n=== GO COMPARISON ANALYSIS ===")
        shared_terms = go_comparison.get("shared_terms", [])
        if shared_terms:
            parts.append(f"  Shared GO terms: {len(shared_terms)} terms present in 2+ genes")
            for term in shared_terms[:10]:
                term_name = term.get("name", term.get("term", "Unknown"))
                genes = ", ".join(term.get("genes", []))
                parts.append(f"    - {term_name}: {genes}")

    return "\n".join(parts)


def _parse_corrected_findings(corrected_text: str) -> List[str]:
    """Parse corrected bullet point text back into a list of findings."""
    findings = []
    for line in corrected_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove bullet prefixes
        if line.startswith("•"):
            line = line[1:].strip()
        elif line.startswith("-"):
            line = line[1:].strip()
        elif line.startswith("*"):
            line = line[1:].strip()
        # Remove numbered prefixes like "1." or "1)"
        if line and line[0].isdigit():
            for i, char in enumerate(line):
                if char in ".)" and i < 3:
                    line = line[i + 1:].strip()
                    break
                elif not char.isdigit():
                    break
        if line:
            findings.append(line)
    return findings


@dataclass
class ValidateCrossGeneSynthesis(BaseNode[GeneState]):
    """Validate the cross-gene synthesis key findings against source data."""

    async def run(self, ctx: GraphRunContext[GeneState]) -> "ValidateLiteratureFindings":
        from src.nodes.validate_literature_findings import ValidateLiteratureFindings

        _t0 = time.perf_counter()
        try:
            print(f"\n{'='*70}")
            print("NODE: Validate Cross-Gene Synthesis")
            print(f"{'='*70}")

            state = ctx.state

            # Skip conditions
            if not VALIDATION_ENABLED:
                print("  Validation disabled in config, skipping")
                return ValidateLiteratureFindings()

            if state.output_mode == "factual":
                print("  Skipping validation in factual mode")
                return ValidateLiteratureFindings()

            if state.synthesis is None:
                print("  No synthesis to validate")
                return ValidateLiteratureFindings()

            # Get key findings
            if hasattr(state.synthesis, "key_findings"):
                key_findings = state.synthesis.key_findings
            elif isinstance(state.synthesis, dict):
                key_findings = state.synthesis.get("key_findings", [])
            else:
                key_findings = []

            if not key_findings:
                print("  No key findings to validate")
                return ValidateLiteratureFindings()

            # Build text to validate (bullet points)
            text_to_validate = "\n".join(f"• {finding}" for finding in key_findings)

            # Build source context with only what GenerateCrossGeneSynthesis had access to
            gene_profiles = {
                g: (p.model_dump() if hasattr(p, "model_dump") else p)
                for g, p in state.gene_profiles.items()
            }

            # Exclude the GO interpretation from the GO comparison analysis
            go_comparison_filtered = None
            if state.go_comparison_analysis:
                go_comparison_filtered = {
                    k: v for k, v in state.go_comparison_analysis.items()
                    if k != "interpretation"
                }

            exp_context = (
                state.experiment_context.model_dump()
                if state.experiment_context and hasattr(state.experiment_context, "model_dump")
                else None
            )

            source_context = _build_synthesis_source_context(
                gene_profiles=gene_profiles,
                gene_interpretations=state.gene_interpretations,
                network_overlap=state.network_overlap_analysis,
                go_comparison=go_comparison_filtered,
                experimental_context=exp_context,
            )

            # Build gene list string for multi-gene validation
            gene_list = ", ".join(sorted(state.gene_profiles.keys()))

            # Validate the synthesis
            result = await validate_section(
                section_name="cross_gene_synthesis",
                section_text=text_to_validate,
                gene_symbol=gene_list,  # Comma-separated list of genes being validated
                state_data={},  # Not used since we provide source_context_override
                state_ref=state,
                node_name="ValidateCrossGeneSynthesis",
                source_context_override=source_context,
                lite_mode=False,
            )

            # Store results in cross-gene validation fields
            state.cross_gene_validation_results["cross_gene_synthesis"] = result
            accuracy = result.get("final_accuracy") or 0.0
            state.cross_gene_accuracy_scores["cross_gene_synthesis"] = accuracy

            # Update key findings if corrected
            final_text = result.get("final_summary", "")
            if final_text and final_text.strip() != text_to_validate.strip():
                corrected_findings = _parse_corrected_findings(final_text)
                if corrected_findings:
                    if hasattr(state.synthesis, "key_findings"):
                        state.synthesis.key_findings = corrected_findings
                    elif isinstance(state.synthesis, dict):
                        state.synthesis["key_findings"] = corrected_findings
                    print(f"  ✓ Cross-gene synthesis corrected ({len(corrected_findings)} findings)")

            status = result.get("status", "unknown")
            print(f"✓ Cross-gene synthesis validation complete — accuracy: {accuracy:.1f}%, status: {status}")

            return ValidateLiteratureFindings()

        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )
