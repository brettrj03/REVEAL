"""Phase 3: interpret all genes after factual analysis is complete."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import time

from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.agents.gene_summarizer import summarize_single_gene
from src.graph.state import GeneState
from src.utils.tracing import trace_event


@dataclass
class InterpretAllGenes(BaseNode[GeneState]):
    """Generate per-gene LLM interpretations once all facts are collected."""

    async def run(self, ctx: GraphRunContext[GeneState]) -> "ValidateInterpretations":
        _t0 = time.perf_counter()
        try:
            from src.nodes.validate_interpretations import ValidateInterpretations

            print(f"\n{'='*70}")
            print("NODE: Interpret All Genes")
            print(f"{'='*70}")

            if ctx.state.output_mode == "factual":
                print("Skipping gene-level interpretations (factual mode)")
                ctx.state.interpretations_complete = True
                return ValidateInterpretations()

            if not ctx.state.gene_profiles:
                print("⚠️  No gene profiles available for interpretation")
                print("    Check that FetchAllGeneData ran successfully")
                ctx.state.interpretations_complete = True
                return ValidateInterpretations()

            print(f"Output mode: {ctx.state.output_mode}")
            print(f"Generating interpretations for {len(ctx.state.gene_profiles)} genes...")
            print()

            success_count = 0
            error_count = 0

            for gene_symbol in sorted(ctx.state.gene_profiles.keys()):
                profile = ctx.state.gene_profiles[gene_symbol]
                gene_data = ctx.state.all_gene_data.get(gene_symbol, {})

                summarizer_payload: Dict[str, object] = {
                    'gene_symbol': gene_symbol,
                    'full_name': gene_data.get('full_name', profile.full_name),
                    'go_terms': gene_data.get('go_terms', []),
                    'expression_data': profile.expression_data,
                    'interactions': profile.interactions,
                }

                try:
                    summaries = await summarize_single_gene(
                        summarizer_payload,
                        state=ctx.state,
                        node_name="InterpretAllGenes",
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    print(f"⚠️  Summarizer failed for {gene_symbol}: {exc}")
                    import traceback
                    traceback.print_exc()
                    # Store error in state for debugging
                    ctx.state.gene_interpretations[gene_symbol] = {
                        'gene_symbol': gene_symbol,
                        'error': str(exc),
                        'functional_summary': None,
                        'expression_interpretation': None,
                        'interaction_interpretation': None,
                        'experimental_relevance': None,
                    }
                    error_count += 1
                    continue

                ctx.state.gene_summaries[gene_symbol] = summaries
                success_count += 1

                # Log success with summary lengths
                print(f"  ✓ {gene_symbol}: Generated {len([v for v in summaries.values() if v])}/4 interpretations")

                # Prefer functional_interpretation (richer 3-4 sentence LLM text with GO terms,
                # interactions and cellular localisation) over the shorter gene_description.
                # gene_description is a 2-3 sentence academic stub that rarely contains enough
                # discrete claims for the validator, producing "no claims extracted" on most runs.
                # Falls back to gene_description so nothing breaks if the interpretation is absent.
                functional_summary = (
                    summaries.get('functional_interpretation')
                    or summaries.get('gene_description')
                )
                if functional_summary:
                    profile.functional_summary = functional_summary

                expression_text = summaries.get('expression_interpretation')
                if expression_text:
                    profile.expression_interpretation = expression_text

                interaction_text = summaries.get('network_interpretation')
                if interaction_text:
                    profile.interaction_interpretation = interaction_text

                # NOTE: experimental_relevance is intentionally not mapped here.
                # The summarizer payload does not include experiment context, so the field
                # would always be empty. It has been removed from SECTIONS_TO_VALIDATE.

                ctx.state.gene_interpretations[gene_symbol] = {
                    'gene_symbol': gene_symbol,
                    'functional_summary': profile.functional_summary,
                    'expression_interpretation': profile.expression_interpretation,
                    'interaction_interpretation': profile.interaction_interpretation,
                    # experimental_relevance intentionally omitted — never generated
                    # (removed from SECTIONS_TO_VALIDATE in validation_config.py)
                }

                trace_event(
                    "interpretation.gene_sections",
                    gene=gene_symbol,
                    sections=sorted(summaries.keys()),
                    payload_keys=sorted(summarizer_payload.keys()),
                    state_inputs=['gene_profiles', 'all_gene_data']
                )

            # Print summary
            print()
            print(f"Interpretation complete: {success_count} succeeded, {error_count} failed")
            if error_count > 0:
                print("⚠️  Some interpretations failed - check logs above for details")

            ctx.state.interpretations_complete = True
            return ValidateInterpretations()
        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )
