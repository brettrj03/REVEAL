"""
Validate comprehensive gene summaries after they are generated.

Uses the same validation runner as ValidateInterpretations, but targets
only the gene_summary section and merges results into the existing
per-gene snapshot so Streamlit sees everything in one place.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.graph.state import GeneState
from src.nodes.validate_interpretations import (
    _state_to_dict,
    _store_gene_result,
    flatten_claims_from_snapshot,
)
from src.validation.simple_validation_runner import (
    merge_gene_summary_into_snapshot,
    validate_gene_sections,
)
from src.validation_config.validation_config import (
    ACCURACY_THRESHOLD,
    VALIDATE_GENE_SUMMARY,
    VALIDATION_ENABLED,
)


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _extract_text(entry: Any) -> Optional[str]:
    """Pull plain text from a gene summary entry (str or dict)."""
    if isinstance(entry, str):
        return entry.strip() or None
    if isinstance(entry, dict):
        for key in ("gene_summary", "summary", "functional_interpretation"):
            val = entry.get(key)
            if val:
                text = str(val).strip()
                if text:
                    return text
    return None


async def _validate_one(
    gene: str,
    summary_text: str,
    state: GeneState,
    state_dict: Dict[str, Any],
) -> int:
    """Validate one gene's summary and merge into existing snapshot. Returns 1 on success."""
    result = await validate_gene_sections(
        gene_symbol=gene,
        sections={"gene_summary": summary_text},
        state_data=state_dict,
        sections_to_validate=["gene_summary"],
        state_ref=state,
        node_name="ValidateGeneSummaries",
    )

    gs_section = result.get("sections", {}).get("gene_summary")
    if not gs_section:
        print(f"    ⚠️ Gene summary validation failed for {gene}")
        return 0

    # Update the summary text in state if rewritten
    final_text = gs_section.get("final_summary", summary_text)
    state.gene_summaries[gene] = final_text
    interp = state.gene_interpretations.get(gene)
    if isinstance(interp, dict):
        interp["gene_summary"] = final_text
        state.gene_interpretations[gene] = interp

    # Merge into existing snapshot (from ValidateInterpretations) or create new
    existing = state.validation_results.get(gene)
    if existing:
        merged = merge_gene_summary_into_snapshot(existing, gs_section)
    else:
        merged = result

    # Store everything via the shared helper
    _store_gene_result(state, gene, merged)

    acc = gs_section.get("final_accuracy") or 0.0
    print(f"    {gene} / gene_summary — {acc:.1f}% accuracy")
    return 1


# ───────────────────────────────────────────────────────────────────────────
# Node
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class ValidateGeneSummaries(BaseNode[GeneState]):
    """Runs a targeted validation pass for comprehensive gene summaries."""

    async def run(self, ctx: GraphRunContext[GeneState]) -> "FinalSummary":
        from src.nodes.final_summary import FinalSummary

        t0 = time.perf_counter()
        try:
            print(f"\n{'=' * 70}")
            print("NODE: Validate Gene Summaries")
            print(f"{'=' * 70}")

            state = ctx.state
            if not VALIDATION_ENABLED or not VALIDATE_GENE_SUMMARY:
                print("  Gene summary validation disabled")
                return FinalSummary()

            if not state.gene_summaries:
                print("  No gene summaries to validate")
                return FinalSummary()

            state_dict = _state_to_dict(state)

            tasks = []
            for gene, entry in state.gene_summaries.items():
                text = _extract_text(entry)
                if text:
                    tasks.append(_validate_one(gene, text, state, state_dict))
                else:
                    if entry is None:
                        reason = "entry was None"
                    elif isinstance(entry, str):
                        reason = "string was empty after stripping"
                    elif isinstance(entry, dict):
                        keys = ", ".join(sorted(entry.keys())) or "<no keys>"
                        reason = f"dict keys were: {keys}"
                    else:
                        reason = f"unsupported type: {type(entry).__name__}"
                    print(
                        f"  ⚠️ gene_summaries for {gene} had no extractable text — {reason}"
                    )

            if not tasks:
                print("  No summaries with extractable text — skipping gene summary validation")
                return FinalSummary()

            results = await asyncio.gather(*tasks)
            count = sum(results)

            if count == 0:
                print("  No summaries were validated")
            else:
                print(f"  Validated {count} gene summary/summaries")

            return FinalSummary()
        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - t0, 3),
            )
