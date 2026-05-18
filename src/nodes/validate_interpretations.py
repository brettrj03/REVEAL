"""
Validate Interpretations

Runs immediately after InterpretAllGenes so every downstream node sees
validated, grounded summaries.  Uses asyncio.gather for parallel gene
validation — each gene's sections also run in parallel internally.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.graph.state import ClaimObject, GeneState
from src.validation.simple_validation_runner import validate_gene_sections
from src.validation_config.validation_config import (
    ACCURACY_THRESHOLD,
    SECTIONS_TO_VALIDATE,
    SKIP_IN_FACTUAL_MODE,
    VALIDATION_ENABLED,
)


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _state_to_dict(state: GeneState) -> Dict[str, Any]:
    """Serialize GeneState to a plain dict for the validator."""
    d: Dict[str, Any] = {
        "all_gene_data": state.all_gene_data,
        "gene_interpretations": state.gene_interpretations,
        "gene_summaries": state.gene_summaries,
        "gene_profiles": {
            k: v.model_dump() if hasattr(v, "model_dump") else v
            for k, v in state.gene_profiles.items()
        },
        "gene_top_papers": state.gene_top_papers,
        "literature_findings_summary": state.literature_findings_summary,
        "network_overlap_analysis": state.network_overlap_analysis,
        "go_comparison_analysis": state.go_comparison_analysis,
    }
    if state.synthesis:
        d["synthesis"] = (
            state.synthesis.model_dump()
            if hasattr(state.synthesis, "model_dump")
            else state.synthesis
        )
    return d


def flatten_claims_from_snapshot(snapshot: Dict[str, Any]) -> List[ClaimObject]:
    """Convert final-iteration claims from a snapshot into ClaimObjects."""
    objects: List[ClaimObject] = []
    tier1_types = {
        "database_fact",
        "expression_pattern",
        "quantitative_statement",
        "go_annotation",
        "protein_interaction",
    }
    for sec_name, sec in snapshot.get("sections", {}).items():
        iters = sec.get("iterations", [])
        if not iters:
            continue
        for claim in iters[-1].get("all_claims", []):
            cid = claim.get("claim_id")
            if not cid:
                continue
            verdict = claim.get("verdict", "accurate")
            objects.append(
                ClaimObject(
                    claim_id=cid,
                    section=sec_name,
                    claim_text=claim.get("claim_text", ""),
                    category=claim.get("claim_type", "other"),
                    tier=1 if claim.get("claim_type") in tier1_types else 2,
                    verdict=verdict,
                    final_text=claim.get("claim_text"),
                )
            )
    return objects


def _store_gene_result(
    state: GeneState,
    gene: str,
    result: Dict[str, Any],
) -> None:
    """Write a single gene's validation result into all relevant state fields."""
    state.validation_results[gene] = result

    accuracy = result.get("_accuracy_score")
    if accuracy is not None:
        state.accuracy_scores[gene] = accuracy
    state.low_confidence_flags[gene] = (
        accuracy is not None and accuracy < ACCURACY_THRESHOLD
    )

    claim_objects = flatten_claims_from_snapshot(result)
    state.validated_claims[gene] = claim_objects
    state.claim_verdicts[gene] = {
        obj.claim_id: {
            "section": obj.section,
            "verdict": obj.verdict,
            "clarify_decision": obj.clarify_decision,
            "final_text": obj.final_text,
            "tier": obj.tier,
            "category": obj.category,
        }
        for obj in claim_objects
    }

    # Write corrected text back to state.gene_interpretations for functional_summary
    # (so UI displays the validated version, not the original uncorrected text)
    fs_section = result.get("sections", {}).get("functional_summary")
    if fs_section:
        final_text = fs_section.get("final_summary", fs_section.get("original_summary", ""))
        interp = state.gene_interpretations.get(gene)
        if isinstance(interp, dict) and final_text:
            interp["functional_summary"] = final_text
            state.gene_interpretations[gene] = interp

    # Write corrected text back for expression_interpretation
    ei_section = result.get("sections", {}).get("expression_interpretation")
    if ei_section:
        final_text = ei_section.get("final_summary", ei_section.get("original_summary", ""))
        interp = state.gene_interpretations.get(gene)
        if isinstance(interp, dict) and final_text:
            interp["expression_interpretation"] = final_text
            state.gene_interpretations[gene] = interp

    # Write corrected text back for interaction_interpretation
    ii_section = result.get("sections", {}).get("interaction_interpretation")
    if ii_section:
        final_text = ii_section.get("final_summary", ii_section.get("original_summary", ""))
        interp = state.gene_interpretations.get(gene)
        if isinstance(interp, dict) and final_text:
            interp["interaction_interpretation"] = final_text
            state.gene_interpretations[gene] = interp

    # Write corrected text back for experimental_relevance
    er_section = result.get("sections", {}).get("experimental_relevance")
    if er_section:
        final_text = er_section.get("final_summary", er_section.get("original_summary", ""))
        interp = state.gene_interpretations.get(gene)
        if isinstance(interp, dict) and final_text:
            interp["experimental_relevance"] = final_text
            state.gene_interpretations[gene] = interp

    if state.validation_logs is None:
        state.validation_logs = {}
    state.validation_logs[gene] = {
        "accuracy": accuracy,
        "improvement": result.get("overall_accuracy"),
        "hallucinations": result.get("total_hallucinations_detected", 0),
        "corrections": result.get("total_corrections_made", 0),
        "low_confidence": state.low_confidence_flags.get(gene, False),
        "passes": max(
            (
                len(s.get("iterations", []))
                for s in result.get("sections", {}).values()
            ),
            default=0,
        ),
    }


# ───────────────────────────────────────────────────────────────────────────
# Node
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class ValidateInterpretations(BaseNode[GeneState]):
    """Validates AI-generated interpretation sections and stores results in state."""

    async def run(self, ctx: GraphRunContext[GeneState]) -> "InterpretNetworkOverlap":
        from src.nodes.interpret_network_overlap import InterpretNetworkOverlap

        print(f"\n{'=' * 70}")
        print("NODE: Validate Interpretations")
        print(f"{'=' * 70}")

        state = ctx.state
        state.log_node_execution(self.__class__.__name__)
        t0 = time.time()

        if not VALIDATION_ENABLED:
            print("  Validation disabled in config, skipping")
            return InterpretNetworkOverlap()

        if SKIP_IN_FACTUAL_MODE and state.output_mode == "factual":
            print("  Skipping validation in factual mode")
            return InterpretNetworkOverlap()

        genes = state.get_genes_found()
        if not genes:
            print("  No genes to validate")
            return InterpretNetworkOverlap()

        state_dict = _state_to_dict(state)

        # Build section texts per gene
        gene_sections: Dict[str, Dict[str, str]] = {}
        for gene in genes:
            interp = state.gene_interpretations.get(gene, {})
            sections: Dict[str, str] = {}
            if isinstance(interp, dict):
                for sec in SECTIONS_TO_VALIDATE:
                    sections[sec] = interp.get(sec) or ""
            else:
                sections["full_interpretation"] = str(interp or "")
            gene_sections[gene] = sections

        print(f"  Running parallel validation for {len(gene_sections)} gene(s)...")

        # Validate all genes in parallel
        tasks = [
            validate_gene_sections(
                gene_symbol=gene,
                sections=gene_sections[gene],
                state_data=state_dict,
                sections_to_validate=SECTIONS_TO_VALIDATE,
                state_ref=state,
                node_name="ValidateInterpretations",
            )
            for gene in gene_sections
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_corrections = 0
        genes_validated = 0

        for gene, result in zip(gene_sections.keys(), results):
            if isinstance(result, Exception):
                print(f"  ⚠️ Validation failed for {gene}: {result}")
                continue

            _store_gene_result(state, gene, result)
            total_corrections += result.get("total_corrections_made", 0)
            genes_validated += 1

            overall = result.get("overall_accuracy")
            if overall is None:
                print(f"  {gene}: No verifiable claims")
            else:
                print(f"  {gene}: Accuracy {overall:.1f}%")

        elapsed = time.time() - t0
        print(f"{'=' * 70}")
        print("VALIDATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Genes validated: {genes_validated}")
        print(f"  Total corrections: {total_corrections}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"{'=' * 70}")

        return InterpretNetworkOverlap()
