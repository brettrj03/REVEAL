"""
Orchestrate validation across multiple sections and genes.

Uses asyncio.gather for parallelism — no shared mutable state.
Produces the exact snapshot schema that Streamlit reads from
state.validation_results[gene].
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.validation.unified_validator import SECTION_DISPLAY_NAMES, validate_section


# ───────────────────────────────────────────────────────────────────────────
# Validate all sections for a single gene (parallel)
# ───────────────────────────────────────────────────────────────────────────

async def validate_gene_sections(
    gene_symbol: str,
    sections: Dict[str, str],
    state_data: Dict[str, Any],
    sections_to_validate: Optional[List[str]] = None,
    state_ref=None,
    node_name: Optional[str] = None,
    source_context_override: str = "",
    lite_mode: bool = False,
) -> Dict[str, Any]:
    """
    Validate multiple sections for one gene in parallel.

    Returns a snapshot dict that plugs directly into
    state.validation_results[gene_symbol].

    Args:
        source_context_override: If provided, use this as the source context
            for all sections instead of building from state_data.
        lite_mode: If True, skip Pass 2 (rewrite + re-validate) for all sections.
    """

    # Filter to requested sections
    if sections_to_validate:
        sections = {
            k: v for k, v in sections.items() if k in sections_to_validate
        }

    if not sections:
        return _empty_snapshot(gene_symbol)

    names = list(sections.keys())
    tasks = [
        validate_section(
            name,
            sections[name],
            gene_symbol,
            state_data,
            state_ref=state_ref,
            node_name=node_name,
            source_context_override=source_context_override,
            lite_mode=lite_mode,
        )
        for name in names
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    section_map: Dict[str, Dict[str, Any]] = dict(zip(names, results))
    return _assemble_snapshot(gene_symbol, section_map)


# ───────────────────────────────────────────────────────────────────────────
# Merge a gene_summary section into an existing snapshot
# ───────────────────────────────────────────────────────────────────────────

def merge_gene_summary_into_snapshot(
    existing: Dict[str, Any],
    gene_summary_section: Dict[str, Any],
) -> Dict[str, Any]:
    """Add/replace gene_summary in an existing snapshot, recomputing aggregates."""
    snapshot = dict(existing)
    section_map = dict(snapshot.get("sections", {}))
    section_map["gene_summary"] = gene_summary_section
    snapshot["sections"] = section_map
    snapshot["gene_summary_validation"] = gene_summary_section

    # Recompute aggregates
    _refresh_aggregates(snapshot, section_map)
    return snapshot


# ───────────────────────────────────────────────────────────────────────────
# Internals
# ───────────────────────────────────────────────────────────────────────────

def _empty_snapshot(gene_symbol: str) -> Dict[str, Any]:
    return {
        "gene_symbol": gene_symbol,
        "validated_at": datetime.now().isoformat(),
        "overall_accuracy": None,
        "total_hallucinations_detected": 0,
        "total_corrections_made": 0,
        "status": "no_claims",
        "sections": {},
        "gene_summary_validation": None,
        "_accuracy_score": None,
        "_all_claims": [],
    }


def _assemble_snapshot(
    gene_symbol: str,
    section_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a full gene snapshot from per-section results."""

    # Pool claims across all sections from final iterations
    total_accurate = 0
    total_partially_accurate = 0
    total_inaccurate = 0

    for s in section_map.values():
        if s.get("status") in ("no_claims", "failed"):
            continue

        iters = s.get("iterations", [])
        if not iters:
            continue

        final_iter = iters[-1]
        total_accurate += final_iter.get("claims_accurate", 0)
        total_partially_accurate += final_iter.get("claims_partially_accurate", 0)
        total_inaccurate += final_iter.get("claims_inaccurate", 0)

    # Calculate pooled accuracy using same formula as section accuracy
    total_claims = total_accurate + total_partially_accurate + total_inaccurate
    if total_claims > 0:
        overall_accuracy = ((total_accurate + 0.5 * total_partially_accurate) / total_claims) * 100
    else:
        overall_accuracy = None

    # Overall status
    statuses = [s.get("status", "") for s in section_map.values()]
    meaningful = [s for s in statuses if s not in ("no_claims", "")]
    if not meaningful:
        overall_status = "no_claims"
    elif "failed" in meaningful:
        overall_status = "failed"
    elif any(s == "partial" for s in meaningful):
        overall_status = "partial"
    elif any(s == "corrected" for s in meaningful):
        overall_status = "corrected"
    else:
        overall_status = "passed"

    # Totals
    total_hall = sum(
        s.get("claims_summary", {}).get("inaccurate_count", 0)
        for s in section_map.values()
    )
    total_corr = sum(
        sum(
            len(it.get("refinement_changes", []))
            for it in s.get("iterations", [])
        )
        for s in section_map.values()
    )

    # Flat claim list for validated_claims / claim_verdicts
    all_claims: List[Dict[str, Any]] = []
    for sec_name, sec in section_map.items():
        iters = sec.get("iterations", [])
        if not iters:
            continue
        final_iter = iters[-1]
        for claim in final_iter.get("all_claims", []):
            entry = dict(claim)
            entry["section"] = sec_name
            all_claims.append(entry)

    return {
        "gene_symbol": gene_symbol,
        "validated_at": datetime.now().isoformat(),
        "overall_accuracy": overall_accuracy,
        "total_hallucinations_detected": total_hall,
        "total_corrections_made": total_corr,
        "status": overall_status,
        "sections": section_map,
        "gene_summary_validation": section_map.get("gene_summary"),
        "_accuracy_score": overall_accuracy,
        "_all_claims": all_claims,
    }


def _refresh_aggregates(
    snapshot: Dict[str, Any],
    section_map: Dict[str, Dict[str, Any]],
) -> None:
    """Re-compute aggregates in-place (used after merge)."""
    # Pool claims across all sections from final iterations
    total_accurate = 0
    total_partially_accurate = 0
    total_inaccurate = 0

    for s in section_map.values():
        if s.get("status") in ("no_claims", "failed"):
            continue

        iters = s.get("iterations", [])
        if not iters:
            continue

        final_iter = iters[-1]
        total_accurate += final_iter.get("claims_accurate", 0)
        total_partially_accurate += final_iter.get("claims_partially_accurate", 0)
        total_inaccurate += final_iter.get("claims_inaccurate", 0)

    # Calculate pooled accuracy using same formula as section accuracy
    total_claims = total_accurate + total_partially_accurate + total_inaccurate
    if total_claims > 0:
        overall_accuracy = ((total_accurate + 0.5 * total_partially_accurate) / total_claims) * 100
    else:
        overall_accuracy = None

    snapshot["overall_accuracy"] = overall_accuracy
    snapshot["_accuracy_score"] = overall_accuracy

    statuses = [s.get("status", "") for s in section_map.values()]
    meaningful = [s for s in statuses if s not in ("no_claims", "")]
    if not meaningful:
        snapshot["status"] = "no_claims"
    elif "failed" in meaningful:
        snapshot["status"] = "failed"
    elif any(s == "partial" for s in meaningful):
        snapshot["status"] = "partial"
    elif any(s == "corrected" for s in meaningful):
        snapshot["status"] = "corrected"
    else:
        snapshot["status"] = "passed"

    snapshot["total_hallucinations_detected"] = sum(
        s.get("claims_summary", {}).get("inaccurate_count", 0)
        for s in section_map.values()
    )
    snapshot["total_corrections_made"] = sum(
        sum(
            len(it.get("refinement_changes", []))
            for it in s.get("iterations", [])
        )
        for s in section_map.values()
    )

    all_claims: List[Dict[str, Any]] = []
    for sec_name, sec in section_map.items():
        iters = sec.get("iterations", [])
        if not iters:
            continue
        for claim in iters[-1].get("all_claims", []):
            entry = dict(claim)
            entry["section"] = sec_name
            all_claims.append(entry)
    snapshot["_all_claims"] = all_claims
    snapshot["validated_at"] = datetime.now().isoformat()
