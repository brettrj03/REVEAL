"""
Validate literature key findings against their source abstracts.

Each paper's key_finding is checked against its own abstract as the source
of truth. Papers marked as false positives or missing key findings are skipped.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.graph.state import GeneState
from src.nodes.validate_interpretations import (
    _state_to_dict,
    _store_gene_result,
)
from src.validation.simple_validation_runner import (
    merge_gene_summary_into_snapshot,
    _refresh_aggregates,
)
from src.validation.unified_validator import validate_section
from src.validation_config.validation_config import (
    LITE_MODE_LITERATURE_FINDINGS,
    VALIDATE_LITERATURE_FINDINGS,
    VALIDATION_ENABLED,
)


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _extract_papers(gene: str, state: GeneState) -> List[Dict[str, Any]]:
    """
    Extract validatable papers from state.gene_top_papers[gene].

    Skips papers where:
    - is_false_positive is True
    - key_finding is missing or equals 'No key finding extracted'
    - abstract is missing or empty
    """
    top_papers_data = state.gene_top_papers.get(gene, {})
    papers = top_papers_data.get("top_papers", [])

    validatable = []
    for paper in papers:
        # Skip false positives
        if paper.get("is_false_positive", False):
            continue

        # Skip papers without meaningful key findings
        key_finding = paper.get("key_finding", "")
        if not key_finding or key_finding == "No key finding extracted":
            continue

        # Skip papers without abstracts
        abstract = paper.get("abstract", "")
        if not abstract or abstract == "No abstract available":
            continue

        validatable.append(paper)

    return validatable


def _build_abstract_context(paper: Dict[str, Any], gene: str) -> str:
    """Build source context from a paper's abstract for validation."""
    pmid = paper.get("pmid", "Unknown")
    title = paper.get("title", "Unknown title")
    year = paper.get("year", "N/A")
    journal = paper.get("journal", "Unknown journal")
    abstract = paper.get("abstract", "")

    return f"""=== PAPER METADATA ===
PMID: {pmid}
Title: {title}
Journal: {journal} ({year})
Gene being discussed: {gene}

=== ABSTRACT (source of truth) ===
{abstract}

=== VALIDATION INSTRUCTIONS ===
The key finding claim MUST be supported by the abstract text above.
- Mark as ACCURATE if the abstract supports the claim
- Mark as INACCURATE if the abstract contradicts the claim, the claim overstates findings, or the abstract contains insufficient information to verify the claim
"""


async def _validate_one_paper(
    paper: Dict[str, Any],
    gene: str,
    state: GeneState,
    state_dict: Dict[str, Any],
    lite_mode: bool = True,
) -> Dict[str, Any]:
    """
    Validate a single paper's key_finding against its abstract.

    Args:
        lite_mode: If True (default), run Pass 1 only — skip rewrite/Pass 2.
            Literature key findings are short claims where full remediation is overkill.

    Returns the validation result dict.
    """
    key_finding = paper.get("key_finding", "")
    pmid = paper.get("pmid", "Unknown")

    # Build abstract-based source context
    source_context = _build_abstract_context(paper, gene)

    result = await validate_section(
        section_name="literature_finding",
        section_text=key_finding,
        gene_symbol=gene,
        state_data=state_dict,
        state_ref=state,
        node_name="ValidateLiteratureFindings",
        source_context_override=source_context,
        lite_mode=lite_mode,
    )

    # Attach PMID for tracking
    result["pmid"] = pmid

    return result


async def _validate_one_gene(
    gene: str,
    state: GeneState,
    state_dict: Dict[str, Any],
) -> int:
    """
    Validate all papers for one gene and write results back to state.

    Returns count of papers validated.
    """
    papers = _extract_papers(gene, state)

    if not papers:
        print(f"    {gene}: No validatable papers (skipped false positives / missing findings)")
        return 0

    # Validate all papers in parallel (lite mode: Pass 1 only, no rewrite)
    tasks = [
        _validate_one_paper(paper, gene, state, state_dict, lite_mode=LITE_MODE_LITERATURE_FINDINGS)
        for paper in papers
    ]
    results = await asyncio.gather(*tasks)

    # Write validated findings back to paper objects
    validated_count = 0
    top_papers = state.gene_top_papers.get(gene, {}).get("top_papers", [])

    for paper, result in zip(papers, results):
        pmid = paper.get("pmid")

        # Find the paper in top_papers and update it
        for tp in top_papers:
            if tp.get("pmid") == pmid:
                # Get the final (possibly rewritten) key finding
                final_finding = result.get("final_summary", paper.get("key_finding", ""))
                tp["key_finding"] = final_finding
                tp["key_finding_validated"] = True

                # Store validation metadata
                tp["key_finding_accuracy"] = result.get("final_accuracy")
                tp["key_finding_status"] = result.get("status", "unknown")

                validated_count += 1
                break

    # Merge literature finding results into existing validation snapshot
    existing = state.validation_results.get(gene)

    # Build a combined literature section result
    total_accuracy = 0.0
    valid_count = 0
    for r in results:
        acc = r.get("final_accuracy")
        if acc is not None:
            total_accuracy += acc
            valid_count += 1

    avg_accuracy = total_accuracy / valid_count if valid_count > 0 else 0.0

    paper_claims = []
    accurate_claims: List[Dict[str, Any]] = []
    inaccurate_claims: List[Dict[str, Any]] = []
    partial_claims: List[Dict[str, Any]] = []

    verdict_map = {
        "passed": "accurate",
        "corrected": "accurate",
        "partial": "partially_accurate",
        "lite_partial": "partially_accurate",
        "failed": "inaccurate",
    }

    for paper, result in zip(papers, results):
        pmid = paper.get("pmid", "unknown")
        verdict = verdict_map.get(result.get("status"), "accurate")

        # Extract reasoning from validation result's iterations/all_claims
        # Use final iteration (iters[-1]) to match the iteration used for final counts
        reasoning = ""
        iters = result.get("iterations", [])
        if iters and isinstance(iters, list) and len(iters) > 0:
            all_claims = iters[-1].get("all_claims", [])
            if all_claims and isinstance(all_claims, list) and len(all_claims) > 0:
                reasoning = all_claims[0].get("reasoning", "")

        claim = {
            "claim_id": f"literature_finding_{pmid or 'unknown'}",
            "claim_text": paper.get("key_finding", ""),
            "claim_type": "literature_finding",
            "verdict": verdict,
            "reasoning": reasoning,
            "correction": "",
            "flagged": False,
        }
        paper_claims.append(claim)
        if verdict == "accurate":
            accurate_claims.append(claim)
        elif verdict == "inaccurate":
            inaccurate_claims.append(claim)
        elif verdict == "partially_accurate":
            partial_claims.append(claim)

    literature_section = {
        "display_name": "Literature Findings",
        "status": "passed" if avg_accuracy >= 100.0 else "partial",
        "error_message": None,
        "initial_accuracy": avg_accuracy,
        "final_accuracy": avg_accuracy,
        "improvement": 0.0,
        "original_summary": f"{len(papers)} paper key findings",
        "final_summary": f"{validated_count} papers validated",
        "claims_summary": {
            "initial_claim_count": len(papers),
            "final_claim_count": validated_count,
            "accurate_count": sum(1 for r in results if (r.get("final_accuracy") or 0) >= 100),
            "inaccurate_count": sum(1 for r in results if r.get("status") == "failed"),
            "partial_count": sum(1 for r in results if r.get("status") == "partial"),
        },
        "iterations": [
            {
                "pass_number": 1,
                "all_claims": paper_claims,
                "accurate_claims": accurate_claims,
                "inaccurate_claims": inaccurate_claims,
                "partially_accurate_claims": partial_claims,
                "accuracy_percentage": avg_accuracy,
                "refinement_changes": [],
            }
        ],
        "paper_results": results,  # Store individual paper results
    }

    if existing:
        # Merge into existing snapshot
        merged = existing.copy()
        if "sections" not in merged:
            merged["sections"] = {}
        merged["sections"]["literature_finding"] = literature_section

        # Refresh aggregates to include literature findings in overall accuracy
        _refresh_aggregates(merged, merged["sections"])
    else:
        # Create new snapshot
        merged = {
            "gene": gene,
            "sections": {"literature_finding": literature_section},
            "timestamp": None,
        }

    _store_gene_result(state, gene, merged)

    print(f"    {gene}: {validated_count} papers validated — {avg_accuracy:.1f}% average accuracy")
    return validated_count


# ───────────────────────────────────────────────────────────────────────────
# Node
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class ValidateLiteratureFindings(BaseNode[GeneState]):
    """Validate key findings from literature papers against their abstracts."""

    async def run(self, ctx: GraphRunContext[GeneState]) -> "GenerateGeneSummaries":
        from src.nodes.generate_gene_summaries import GenerateGeneSummaries

        t0 = time.perf_counter()
        try:
            print(f"\n{'=' * 70}")
            print("NODE: Validate Literature Findings")
            print(f"{'=' * 70}")

            state = ctx.state
            if not VALIDATION_ENABLED or not VALIDATE_LITERATURE_FINDINGS:
                print("  Literature findings validation disabled")
                return GenerateGeneSummaries()

            if not state.gene_top_papers:
                print("  No literature papers to validate")
                return GenerateGeneSummaries()

            state_dict = _state_to_dict(state)
            genes = list(state.gene_top_papers.keys())

            tasks = [
                _validate_one_gene(gene, state, state_dict)
                for gene in genes
            ]

            results = await asyncio.gather(*tasks)
            total_validated = sum(results)

            if total_validated == 0:
                print("  No literature findings were validated")
            else:
                print(f"  Validated {total_validated} paper key findings across {len(genes)} genes")

            return GenerateGeneSummaries()
        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - t0, 3),
            )
