"""
Unified validation: extract claims, validate, optionally rewrite — all in one place.

Architecture
------------
- Pass 1: gpt-4.1-mini extracts + validates claims in a single structured-JSON call.
- If accuracy < threshold AND corrections exist:
    Pass 2a: gpt-4.1-mini rewrites the section text.
    Pass 2b: gpt-4.1-mini re-validates the rewritten text.
    Pass 3a/3b: If still below threshold, repeats rewrite + re-validate one more time.
- Maximum validation passes controlled by MAX_REFINEMENT_PASSES (default: 3).
- No shared mutable logger — every section is self-contained.
- Output is the exact dict schema Streamlit's render_validation_results() reads.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import openai
from openai import AsyncOpenAI

from src.graph.state import _accumulate_tokens
from src.validation_config.validation_config import (
    ACCURACY_THRESHOLD,
    CLAIM_VALIDATOR_TEMPERATURE,
    MAX_REFINEMENT_PASSES,
    MAX_REWRITE_ATTEMPTS,
    SUMMARY_REFINER_TEMPERATURE,
)
from src.config import get_active_model

# ───────────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────────

SECTION_DISPLAY_NAMES: Dict[str, str] = {
    "functional_summary": "Functional Summary",
    "expression_interpretation": "Expression Interpretation",
    "interaction_interpretation": "Interaction Interpretation",
    "experimental_relevance": "Experimental Relevance",
    "gene_summary": "Gene Summary",
    "literature_finding": "Literature Finding",
    "go_interpretation": "GO Term Overlap Interpretation",
    "cross_gene_synthesis": "Cross-Gene Analysis Synthesis",
    "network_interpretation": "Network Interpretation",
}

# ───────────────────────────────────────────────────────────────────────────
# OpenAI client (singleton)
# ───────────────────────────────────────────────────────────────────────────

_openai_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


# ───────────────────────────────────────────────────────────────────────────
# Token budget helper
# ───────────────────────────────────────────────────────────────────────────

def _max_tokens_for(section_text: str, section_name: str = "") -> int:
    """Scale completion budget to section length so large sections don't truncate."""
    n = len(section_text or "")
    if n > 2000:
        base = 10000
    elif n > 800:
        base = 8000
    else:
        base = 6000
    # gene_summary has ~3x the expected claims, needs proportionally more output tokens
    if section_name == "gene_summary":
        return int(base * 3.0)
    return base


# ───────────────────────────────────────────────────────────────────────────
# Safe JSON parsing
# ───────────────────────────────────────────────────────────────────────────

def _parse_json(raw: Optional[str], label: str) -> Dict[str, Any]:
    """Parse JSON from model output, raising a descriptive error on failure."""
    if not raw or not raw.strip():
        raise ValueError(f"Empty response from model for {label}. Got: {repr(raw)}")
    # Strip markdown fences if the model wraps its output
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.index("\n") if "\n" in text else 3
        text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON for {label}: {exc}. Preview: {repr(text[:300])}"
        ) from exc


def _first_sentence(text: str) -> str:
    """Return only the first sentence of text (prevents multi-sentence corrections fragmenting claims)."""
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+|\n", text.strip())
    first = parts[0].strip() if parts else text.strip()
    if first and not first[-1] in ".!?":
        first += "."
    return first


def _count_sentences(text: str) -> int:
    """Approximate sentence count for rewrite fragmentation guard."""
    if not text or not text.strip():
        return 0
    return len([s for s in re.split(r"(?<=[.!?])\s+|\n", text.strip()) if s.strip()])


# ───────────────────────────────────────────────────────────────────────────
# API call with single retry on empty response
# ───────────────────────────────────────────────────────────────────────────

async def _api_validate(
    client: AsyncOpenAI,
    prompt: str,
    max_tokens: int,
    label: str,
    *,
    state_ref=None,
    node_name: Optional[str] = None,
) -> str:
    """Call the validation model (json_object mode) with one retry on empty."""
    for attempt in range(2):
        try:
            resp = await client.chat.completions.create(
                model=get_active_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=CLAIM_VALIDATOR_TEMPERATURE,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                timeout=120.0,  # 2 minutes max per validation call
            )
        except openai.APITimeoutError:
            raise ValueError(f"Validation API call timed out after 120s for {label}")
        usage = getattr(resp, "usage", None)
        _accumulate_tokens(state_ref, node_name or "Validation", usage)
        content = resp.choices[0].message.content
        finish = resp.choices[0].finish_reason
        # Fail fast on truncation — incomplete JSON will break _parse_json()
        if finish == "length":
            raise ValueError(
                f"Response truncated (finish_reason='length') for {label} — increase max_tokens"
            )
        if content and content.strip():
            return content
        print(
            f"      ⚠️  [{label}] Empty response "
            f"(finish_reason={finish!r}), attempt {attempt + 1}/2"
        )
    raise ValueError(
        f"Model returned empty response for {label} after 2 attempts "
        f"(finish_reason={finish!r})"
    )


async def _api_rewrite(
    client: AsyncOpenAI,
    prompt: str,
    fallback: str,
    *,
    state_ref=None,
    node_name: Optional[str] = None,
) -> str:
    """Call the rewrite model (plain text). Returns fallback on empty."""
    try:
        resp = await client.chat.completions.create(
            model=get_active_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=SUMMARY_REFINER_TEMPERATURE,
            max_completion_tokens=1500,
            timeout=60.0,  # 1 minute max for rewrites
        )
    except openai.APITimeoutError:
        raise ValueError("Rewrite API call timed out after 60s")
    usage = getattr(resp, "usage", None)
    _accumulate_tokens(state_ref, node_name or "Validation", usage)
    text = (resp.choices[0].message.content or "").strip()
    return text if text else fallback


# ───────────────────────────────────────────────────────────────────────────
# Source-data builder (what the validator sees)
# ───────────────────────────────────────────────────────────────────────────

def _build_source_context(
    state_data: Dict[str, Any], gene_symbol: str, section_name: str = ""
) -> str:
    """Assemble the source data the validator compares claims against.

    When section_name is provided, returns only the data that interpreter
    had access to for that section (to prevent information advantage).
    """
    parts: List[str] = []
    gene_data = state_data.get("all_gene_data", {}).get(gene_symbol, {})
    top_papers = (
        state_data.get("gene_top_papers", {})
        .get(gene_symbol, {})
        .get("top_papers", [])
    )

    # ── Pre-computed statistics ──
    parts.append("=== PRE-COMPUTED STATISTICS ===")
    parts.append("(Use these to verify claims about averages, totals, and ranges)")

    # Interactions
    interactions = gene_data.get("interactions", [])
    if interactions:
        scores = []
        for inter in interactions:
            score = inter.get("score", inter.get("combined_score", 0)) or 0
            if score > 1:
                score /= 1000.0
            scores.append(
                (score, inter.get("partner_symbol", inter.get("partner", "Unknown")))
            )
        if scores:
            vals = [s[0] for s in scores]
            mean_s = sum(vals) / len(vals)
            max_s = max(scores, key=lambda x: x[0])
            min_s = min(scores, key=lambda x: x[0])
            parts.append("\nInteraction Statistics:")
            parts.append(f"  Total interaction partners: {len(interactions)}")
            parts.append(f"  Mean interaction confidence: {mean_s:.3f}")
            parts.append(f"  Max confidence: {max_s[0]:.3f} ({max_s[1]})")
            parts.append(f"  Min confidence: {min_s[0]:.3f} ({min_s[1]})")
    else:
        parts.append("\nInteraction Statistics:")
        parts.append("  Total interaction partners: 0")

    # Expression
    expression = gene_data.get("expression", [])
    if expression:
        tpm_pairs = []
        for expr in expression:
            tpm = expr.get("level", expr.get("tpm", 0)) or 0
            tpm_pairs.append((tpm, expr.get("tissue", "Unknown")))
        if tpm_pairs:
            vals = [t[0] for t in tpm_pairs]
            mean_t = sum(vals) / len(vals)
            max_t = max(tpm_pairs, key=lambda x: x[0])
            nonzero = [(t, ts) for t, ts in tpm_pairs if t >= 0.005]
            if nonzero:
                min_t = min(nonzero, key=lambda x: x[0])
            else:
                min_t = (0, "N/A")
            zero_count = sum(1 for v in vals if v == 0)
            parts.append("\nExpression Statistics:")
            parts.append(f"  Total tissues measured: {len(expression)}")
            parts.append(f"  Mean TPM across all tissues: {mean_t:.2f}")
            parts.append(f"  Highest expressed tissue: {max_t[1]} ({max_t[0]:.2f} TPM)")
            parts.append(
                f"  Lowest expressed tissue (non-zero): {min_t[1]} ({min_t[0]:.2f} TPM)"
            )
            parts.append(f"  Tissues with zero expression: {zero_count}")
    else:
        parts.append("\nExpression Statistics:")
        parts.append("  Total tissues measured: 0")

    # GO terms
    go_terms = gene_data.get("go_terms", [])
    mf = sum(1 for t in go_terms if t.get("namespace") == "molecular_function")
    bp = sum(1 for t in go_terms if t.get("namespace") == "biological_process")
    cc = sum(1 for t in go_terms if t.get("namespace") == "cellular_component")
    parts.append("\nGO Term Statistics:")
    parts.append(f"  Total GO terms: {len(go_terms)}")
    parts.append(f"  Molecular Function: {mf}")
    parts.append(f"  Biological Process: {bp}")
    parts.append(f"  Cellular Component: {cc}")

    # Literature
    if top_papers:
        years: List[int] = []
        for p in top_papers:
            y = p.get("year")
            if y:
                try:
                    years.append(int(y))
                except (ValueError, TypeError):
                    pass
        parts.append("\nLiterature Statistics:")
        parts.append(f"  Total papers: {len(top_papers)}")
        if years:
            parts.append(f"  Date range: {min(years)}-{max(years)}")
    else:
        parts.append("\nLiterature Statistics:")
        parts.append("  Total papers: 0")

    # ── Section-specific early return for functional_summary ──
    # The functional_summary interpreter only sees:
    # - 5 MF + 5 BP + 3 CC GO terms (with definitions, filtered to those that have them)
    # - Interaction count (no partner names)
    # - Literature count (no titles/abstracts)
    if section_name == "functional_summary":
        # Gene identity
        parts.append("\n=== GENE IDENTITY ===")
        parts.append(f"Symbol: {gene_data.get('symbol', gene_symbol)}")
        parts.append(f"Gene ID: {gene_data.get('gene_id', 'N/A')}")
        parts.append(
            f"Full Name: {gene_data.get('full_name', gene_data.get('name', 'N/A'))}"
        )

        # GO terms — match interpreter's _format_go_terms() exactly:
        # - Filter to terms WITH definitions
        # - 5 MF, 5 BP, 3 CC
        # - Format: "{go_id} | {name}: {definition}"
        go_terms = gene_data.get("go_terms", [])
        if go_terms:
            parts.append("\n=== GO TERMS (interpreter slice) ===")
            for ns, limit in [
                ("molecular_function", 5),
                ("biological_process", 5),
                ("cellular_component", 3),
            ]:
                ns_terms = [
                    t for t in go_terms
                    if t.get("namespace") == ns and t.get("definition")
                ]
                seen_ids: set = set()
                count = 0
                for t in ns_terms:
                    go_id = t.get("go_id", t.get("id", "N/A"))
                    if go_id in seen_ids:
                        continue
                    seen_ids.add(go_id)
                    name = t.get("name", "Unknown")
                    defn = t.get("definition", "")
                    parts.append(f"  [{ns}] {go_id} | {name}: {defn}")
                    count += 1
                    if count >= limit:
                        break

        # Interactions — count only (interpreter sees "X partners", no names)
        interactions = gene_data.get("interactions", [])
        parts.append("\n=== INTERACTION SUMMARY ===")
        parts.append(f"  Total interaction partners: {len(interactions)}")

        # Literature — count only (interpreter sees "X records", no details)
        literature = gene_data.get("literature", [])
        parts.append("\n=== LITERATURE SUMMARY ===")
        parts.append(f"  Total literature records: {len(literature)}")
        if top_papers:
            parts.append(f"  Top papers available: {len(top_papers)}")

        return "\n".join(parts)

    # ── Section-specific early return for expression_interpretation ──
    # The expression_interpretation interpreter only sees:
    # - Top 5 tissues sorted by TPM (descending)
    # - Statistics: mean_tpm, min_tpm, max_tpm, num_tissues
    # - No GO terms, no interactions, no literature
    if section_name == "expression_interpretation":
        # Gene identity
        parts.append("\n=== GENE IDENTITY ===")
        parts.append(f"Symbol: {gene_data.get('symbol', gene_symbol)}")
        parts.append(f"Gene ID: {gene_data.get('gene_id', 'N/A')}")
        parts.append(
            f"Full Name: {gene_data.get('full_name', gene_data.get('name', 'N/A'))}"
        )

        # Expression — top 5 tissues with TPM (matching interpreter exactly)
        expression = gene_data.get("expression", [])
        if expression:
            parts.append("\n=== EXPRESSION DATA (interpreter slice) ===")
            expr_sorted = sorted(
                [
                    (e.get("level", e.get("tpm", 0)) or 0, e.get("tissue", "Unknown"))
                    for e in expression
                ],
                key=lambda x: x[0],
                reverse=True,
            )
            # Top 5 only — matches interpreter's sorted_expr[:5]
            for tpm, tissue in expr_sorted[:5]:
                parts.append(f"  {tissue}: {tpm:.2f} TPM")

        return "\n".join(parts)

    # ── Section-specific early return for interaction_interpretation ──
    # The interaction_interpretation interpreter only sees:
    # - Total interaction count
    # - Top 5 partner names (NO confidence scores — just comma-joined names)
    # - avg_score used for qualitative guidance only
    # - No GO terms, no expression, no literature
    if section_name == "interaction_interpretation":
        # Gene identity
        parts.append("\n=== GENE IDENTITY ===")
        parts.append(f"Symbol: {gene_data.get('symbol', gene_symbol)}")
        parts.append(f"Gene ID: {gene_data.get('gene_id', 'N/A')}")
        parts.append(
            f"Full Name: {gene_data.get('full_name', gene_data.get('name', 'N/A'))}"
        )

        # Interactions — top 5 partner names only (matching interpreter exactly)
        interactions = gene_data.get("interactions", [])
        if interactions:
            parts.append("\n=== PROTEIN INTERACTIONS (interpreter slice) ===")

            def _score(entry: dict) -> float:
                score = entry.get("score")
                if score is not None:
                    return float(score)
                combined = entry.get("combined_score")
                if combined is not None:
                    combined = float(combined)
                    return combined / 1000.0 if combined > 1 else combined
                return 0.0

            def _partner(entry: dict) -> str:
                return entry.get("partner_symbol") or entry.get("partner") or "Unknown"

            sorted_int = sorted(interactions, key=_score, reverse=True)[:5]
            avg_score = sum(_score(i) for i in interactions) / len(interactions)
            parts.append(f"  Total interactions: {len(interactions)}")
            parts.append(f"  Average confidence (all interactions): {avg_score:.3f}")
            parts.append(f"  Confidence thresholds used by interpreter: >0.7 = high, 0.5–0.7 = moderate, <0.5 = lower")
            for entry in sorted_int:
                parts.append(f"  - {_partner(entry)}: confidence {_score(entry):.3f}")

        return "\n".join(parts)

    # ── Section-specific early return for gene_summary ──
    # The gene_summary interpreter sees a very specific slice:
    # - Function description (long_description or description, 800 chars max)
    # - GO BP: top 5 unique names only (no definitions, no IDs)
    # - GO CC: top 3 unique names only (no definitions)
    # - GO MF: NOT included
    # - Expression: top 1-2 tissue NAMES only (no TPM values!)
    # - Interactions: total count + top 5 partner names (no scores!)
    # - Literature: top 3 key_findings (300 chars each), no PMIDs/titles/abstracts
    # - Literature summary: themes, counts, known_findings
    # - Cross-gene synthesis (if multi-gene query)
    if section_name == "gene_summary":
        # Gene identity with description
        parts.append("\n=== GENE IDENTITY ===")
        parts.append(f"Symbol: {gene_data.get('symbol', gene_symbol)}")
        parts.append(f"Gene ID: {gene_data.get('gene_id', 'N/A')}")
        parts.append(
            f"Full Name: {gene_data.get('full_name', gene_data.get('name', 'N/A'))}"
        )
        # Function description (matching interpreter's 800 char limit)
        func_desc = gene_data.get("long_description") or gene_data.get("description", "")
        if func_desc:
            if len(func_desc) > 800:
                func_desc = func_desc[:800] + "..."
            parts.append(f"Function: {func_desc}")

        # GO terms — names only, no definitions (matching interpreter exactly)
        go_terms = gene_data.get("go_terms", [])
        if go_terms:
            parts.append("\n=== GO TERMS (interpreter slice — names only) ===")
            # BP: top 5 unique names
            bp_terms = [t for t in go_terms if t.get("namespace") == "biological_process"]
            seen_bp: set = set()
            unique_bp = []
            for t in bp_terms:
                name = t.get("name", "")
                if name and name not in seen_bp:
                    seen_bp.add(name)
                    unique_bp.append(name)
                if len(unique_bp) >= 5:
                    break
            if unique_bp:
                parts.append(f"  GO Biological Processes: {', '.join(unique_bp)}")

            # CC: top 3 unique names
            cc_terms = [t for t in go_terms if t.get("namespace") == "cellular_component"]
            seen_cc: set = set()
            unique_cc = []
            for t in cc_terms:
                name = t.get("name", "")
                if name and name not in seen_cc:
                    seen_cc.add(name)
                    unique_cc.append(name)
                if len(unique_cc) >= 3:
                    break
            if unique_cc:
                parts.append(f"  Cellular Localisation: {', '.join(unique_cc)}")

        # Expression — top 1-2 tissue names only (no TPM values!)
        expression = gene_data.get("expression", [])
        if expression:
            parts.append("\n=== EXPRESSION (interpreter slice — names only) ===")
            expr_sorted = sorted(
                expression,
                key=lambda x: x.get("level", x.get("tpm", 0)) or 0,
                reverse=True,
            )
            if expr_sorted:
                highest = expr_sorted[0]
                highest_name = highest.get("tissue", "Unknown").replace("_", " ")
                highest_level = highest.get("level", highest.get("tpm", 0)) or 0
                # Check if second tissue is within 20% of highest
                if len(expr_sorted) > 1:
                    second = expr_sorted[1]
                    second_level = second.get("level", second.get("tpm", 0)) or 0
                    if highest_level > 0 and second_level >= highest_level * 0.8:
                        second_name = second.get("tissue", "Unknown").replace("_", " ")
                        parts.append(f"  Highest-expressed tissues: {highest_name} and {second_name}")
                    else:
                        parts.append(f"  Highest-expressed tissue: {highest_name}")
                else:
                    parts.append(f"  Highest-expressed tissue: {highest_name}")

        # Interactions — count + top 5 partner names only (no scores!)
        interactions = gene_data.get("interactions", [])
        if interactions:
            parts.append("\n=== PROTEIN INTERACTIONS (interpreter slice — names only) ===")
            top_partners = [
                i.get("partner_symbol") or i.get("partner") or "Unknown"
                for i in interactions[:5]
            ]
            parts.append(f"  Total interactions: {len(interactions)}")
            parts.append(f"  Top partners: {', '.join(top_partners)}")

        # Literature — top 3 key_findings + per-paper relevance scores
        if top_papers:
            parts.append("\n=== LITERATURE (interpreter slice — findings only) ===")
            parts.append(f"  Papers available: {len(top_papers)}")

            # Per-paper relevance scores with titles
            parts.append("  Per-paper relevance (1=low, 5=high focus on this gene):")
            high_focus = 0  # score 4-5
            moderate = 0    # score 3
            low_relevance = 0  # score 1-2
            for paper in top_papers:
                score = paper.get("relevance_score", 0)
                title = paper.get("title", "Unknown title")
                if len(title) > 80:
                    title = title[:77] + "..."
                parts.append(f"    [{score}] {title}")
                if score >= 4:
                    high_focus += 1
                elif score == 3:
                    moderate += 1
                else:
                    low_relevance += 1
            parts.append(f"  Highly focused papers (score 4-5): {high_focus}")
            parts.append(f"  Moderately relevant papers (score 3): {moderate}")
            parts.append(f"  Low relevance papers (score 1-2): {low_relevance}")

            # Key findings (top 3)
            for i, paper in enumerate(top_papers[:3], 1):
                finding = paper.get("key_finding", "")
                if finding:
                    if len(finding) > 300:
                        finding = finding[:300] + "..."
                    parts.append(f"  Finding {i}: {finding}")

        # Literature findings summary (themes, counts, known findings)
        lit_summary = state_data.get("literature_findings_summary", {}).get(gene_symbol, {})
        if lit_summary:
            parts.append("\n=== LITERATURE ANALYSIS SUMMARY ===")
            themes = lit_summary.get("research_themes", [])[:5]
            if themes:
                parts.append(f"  Research themes: {', '.join(themes)}")
            total = lit_summary.get("total_papers", 0)
            recent = lit_summary.get("recent_papers_count", 0)
            if total:
                parts.append(f"  Total papers analysed: {total}")
            if recent:
                parts.append(f"  Recent papers (last 2 years): {recent}")
            known = lit_summary.get("known_findings", [])[:3]
            if known:
                parts.append(f"  Established findings: {'; '.join(known)}")

        # Cross-gene synthesis (important for multi-gene queries!)
        synthesis = state_data.get("synthesis", {})
        if isinstance(synthesis, dict):
            exec_summary = synthesis.get("executive_summary", "")
        elif isinstance(synthesis, str):
            exec_summary = synthesis
        else:
            exec_summary = ""
        if exec_summary:
            if len(exec_summary) > 500:
                exec_summary = exec_summary[:500] + "..."
            parts.append("\n=== CROSS-GENE SYNTHESIS ===")
            parts.append(f"  {exec_summary}")

        return "\n".join(parts)

    # ── Section-specific early return for network_interpretation ──
    # The network interpretation interpreter only sees:
    # - Query genes being analysed
    # - Hub proteins (shared partners) with protein name, genes connected, avg_confidence
    # - Direct interactions between query genes with gene_a, gene_b, score
    # - Network statistics: total_interactions, shared_partners count
    if section_name == "network_interpretation":
        network = state_data.get("network_overlap_analysis", {})

        # Query genes
        genes_analyzed = network.get("genes_analyzed", [])
        if genes_analyzed:
            parts.append("\n=== QUERY GENES ===")
            parts.append(f"  Genes: {', '.join(genes_analyzed)}")

        # Hub proteins / shared partners
        hub_proteins = network.get("hub_proteins", network.get("shared_partners", []))
        if hub_proteins:
            parts.append("\n=== SHARED INTERACTION PARTNERS (hub proteins) ===")
            parts.append(f"  Total shared partners: {len(hub_proteins)}")
            for hp in hub_proteins[:20]:  # Show top 20
                if isinstance(hp, dict):
                    protein = hp.get("protein", "Unknown")
                    genes = hp.get("genes", [])
                    avg_conf = hp.get("avg_confidence")
                    conf_str = f", avg_confidence: {avg_conf:.3f}" if avg_conf is not None else ""
                    parts.append(f"  - {protein}: connects {', '.join(genes)}{conf_str}")

        # Direct gene-gene interactions
        direct = network.get("direct_interactions", [])
        if direct:
            parts.append("\n=== DIRECT GENE-GENE INTERACTIONS ===")
            parts.append(f"  Total direct interactions: {len(direct)}")
            for inter in direct:
                if isinstance(inter, dict):
                    gene_a = inter.get("gene_a", "?")
                    gene_b = inter.get("gene_b", "?")
                    score = inter.get("score", 0)
                    # Normalize score if needed
                    if score > 1:
                        score = score / 1000.0
                    parts.append(f"  - {gene_a} ↔ {gene_b}: score {score:.3f}")

        # Network statistics
        stats = network.get("network_stats", network.get("network_statistics", {}))
        if stats:
            parts.append("\n=== NETWORK STATISTICS ===")
            total_int = stats.get("total_interactions")
            if total_int is not None:
                parts.append(f"  Total interactions: {total_int}")
            shared_count = stats.get("shared_partners", stats.get("shared_partner_count"))
            if shared_count is not None:
                parts.append(f"  Shared partners count: {shared_count}")
            avg_conf = stats.get("avg_confidence")
            if avg_conf is not None:
                parts.append(f"  Average confidence: {avg_conf:.3f}")

        return "\n".join(parts)

    # ── Gene identity ──
    parts.append("\n=== GENE IDENTITY ===")
    parts.append(f"Symbol: {gene_data.get('symbol', gene_symbol)}")
    parts.append(f"Gene ID: {gene_data.get('gene_id', 'N/A')}")
    parts.append(
        f"Full Name: {gene_data.get('full_name', gene_data.get('name', 'N/A'))}"
    )
    desc = gene_data.get("description") or gene_data.get("long_description", "")
    if desc:
        parts.append(f"Description: {desc}")

    # ── GO terms (detailed) ──
    if go_terms:
        parts.append("\n=== GO TERMS ===")
        for term in go_terms:
            ns = term.get("namespace")
            tid = term.get("go_id", term.get("id", "N/A"))
            name = term.get("name", "Unknown")
            parts.append(f"  [{ns}] {tid}: {name}")
            defn = term.get("definition", "")
            if defn:
                parts.append(f"     Definition: {defn}")

    # ── Expression (top 15 only) ──
    if expression:
        parts.append("\n=== EXPRESSION DATA (top 15 tissues) ===")
        expr_sorted = sorted(
            [
                (e.get("level", e.get("tpm", 0)) or 0, e.get("tissue", "Unknown"))
                for e in expression
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        nonzero_expr = [(t, ts) for t, ts in expr_sorted if t >= 0.005]
        zero_ct = len(expr_sorted) - len(nonzero_expr)
        for tpm, tissue in nonzero_expr[:15]:
            parts.append(f"  {tissue}: {tpm:.2f} TPM")
        remaining = len(nonzero_expr) - min(len(nonzero_expr), 15)
        if remaining > 0:
            parts.append(f"  ... ({remaining} additional tissues omitted)")
        if zero_ct > 0:
            parts.append(f"  ({zero_ct} tissues near zero expression)")

    # ── Protein interactions ──
    if interactions:
        parts.append("\n=== PROTEIN INTERACTIONS ===")
        for inter in interactions:
            partner = inter.get("partner_symbol", inter.get("partner", "Unknown"))
            score = inter.get("score", inter.get("combined_score", 0)) or 0
            if score > 1:
                score /= 1000.0
            parts.append(f"  {partner}: confidence {score:.3f}")

    # ── Top papers ──
    if top_papers:
        parts.append("\n=== TOP PAPERS ===")
        for paper in top_papers:
            title = paper.get("title", "Unknown")
            year = paper.get("year", "N/A")
            pmid = paper.get("pmid", "N/A")
            finding = paper.get("key_finding", "")
            parts.append(f"  PMID {pmid} ({year}): {title}")
            if finding:
                parts.append(f"    Finding: {finding}")
            abstract = paper.get("abstract") or ""
            if abstract:
                abstract_preview = (abstract[:300] + "...") if len(abstract) > 300 else abstract
                parts.append(f"    Abstract: {abstract_preview}")

    lit_summary = (
        state_data.get("literature_findings_summary", {}).get(gene_symbol, {})
    )
    if lit_summary:
        themes = lit_summary.get("research_themes", [])
        if themes:
            parts.append(f"  Research themes: {', '.join(themes)}")

    # ── Network overlap ──
    network = state_data.get("network_overlap_analysis", {})
    if network:
        parts.append("\n=== NETWORK OVERLAP ANALYSIS ===")
        shared = network.get("shared_partners", network.get("hub_proteins", []))
        if shared:
            parts.append(f"  Shared partners: {shared}")
        direct = network.get("direct_interactions", [])
        if direct:
            parts.append(f"  Direct interactions: {direct}")

    # ── GO comparison ──
    go_comp = state_data.get("go_comparison_analysis", {})
    if go_comp:
        parts.append("\n=== GO COMPARISON ANALYSIS ===")
        for term in go_comp.get("shared_terms", []):
            if isinstance(term, dict):
                parts.append(
                    f"  {term.get('name', 'Unknown')}: {term.get('genes', [])}"
                )

    return "\n".join(parts)


# ───────────────────────────────────────────────────────────────────────────
# Prompt builders
# ───────────────────────────────────────────────────────────────────────────

def _prompt_validate(
    section_name: str,
    section_text: str,
    source_data: str,
    gene_symbol: str,
) -> str:
    # Detect multi-gene input via comma presence
    is_multi_gene = "," in gene_symbol

    if is_multi_gene:
        intro_line = f"Validate the summary below about the following set of genes: {gene_symbol}"
        source_header = f"=== SOURCE DATA FOR GENE SET: {gene_symbol} ==="
    else:
        intro_line = f"Validate the summary below about gene {gene_symbol} against the source data."
        source_header = f"=== SOURCE DATA FOR {gene_symbol} ==="

    return f"""{intro_line}

Your task:
1. Identify every distinct factual or interpretive claim in the text.
2. Validate each claim against the source data using tiered criteria.
3. Return ALL claims in a single JSON response.

=== TIERED VALIDATION CRITERIA ===

TIER 1 — STRICT (database_fact, expression_pattern, quantitative_statement, go_annotation, protein_interaction):
- accurate: data directly supports it
- inaccurate: data explicitly contradicts it (wrong numbers, wrong names), OR the claim makes
  a specific factual assertion for which the source data is entirely silent. In both cases
  provide a correction:
  • Contradicted: provide the correct value from source data.
  • Data absent: soften the claim to remove the ungroundable assertion, e.g.
    "X has 5 partners with mean confidence 0.72" → "X has interaction partners in STRING" if
    the exact values aren't in the source.

TIER 2 — LENIENT (biological_interpretation, reasonable_inference, literature_finding, cross_gene_relationship, other):
- accurate: reasonable scientific extrapolation from available data
- accurate: even if not explicitly stated, as long as it logically follows
- accurate: a hedged inference ("suggests", "may indicate", "is consistent with", "could reflect")
  drawn from facts that ARE present in the source data — the underlying facts are confirmed,
  so the hedged inference from them is acceptable. Example that should pass:
  "the co-occurrence of F8A family members suggests a closely related interaction neighbourhood"
  — if the interaction partners are confirmed in source, this hedged inference is accurate.
- inaccurate: if data ACTIVELY contradicts it, OR if the claim makes a definitive (un-hedged)
  factual assertion about something completely absent from the source data. Example:
  "F8A2 is a member of the endosomal trafficking complex" — stated as fact, no source supports it.
  Provide a correction that removes the unsupported assertion or replaces it with a hedged inference.
  Do NOT mark as inaccurate a claim that is hedged and drawn from confirmed source facts — that
  is accurate.

ROUNDING RULE: A quantitative claim is accurate if rounded to 2 decimal places.
Only mark partially_accurate or inaccurate if the difference exceeds 0.05.

HEDGED CLAIMS (using "may", "suggests", "indicates", "consistent with"):
- accurate: if there is at least one piece of supporting evidence in the source context
  (a GO term, interaction partner, expression pattern, or literature finding) that
  could plausibly ground the claim, even loosely
- inaccurate: if there is zero supporting data in the source context that relates
  to the claim even loosely, OR if the source data actively contradicts it.
  Provide a correction that softens the claim to what IS supported, or removes the assertion.

LOW-RELEVANCE LITERATURE (when source context shows most papers scored 1-2):
The interpreter was instructed to be CAUTIOUS about low-relevance papers and avoid overstating findings.
Claims like "limited dedicated literature", "appeared as incidental finding", or "no papers focus specifically on [gene]"
are ACCURATE if relevance scores support this (mostly 1-2). Do NOT mark such claims as inaccurate for being conservative.

When in doubt about an interpretive claim and some relevant data exists, mark it
accurate. If NO relevant data exists at all, mark it inaccurate and provide a correction
that softens or removes the ungrounded assertion.

{source_header}

{source_data}

=== SUMMARY TO VALIDATE ({section_name}) ===

{section_text}

=== INSTRUCTIONS ===

Return a JSON object with a "claims" array. Each entry:
{{
  "claim_index": <int>,
  "claim_text": "exact text of the claim",
  "claim_type": "database_fact|expression_pattern|quantitative_statement|go_annotation|protein_interaction|biological_interpretation|reasonable_inference|literature_finding|cross_gene_relationship|other",
  "verdict": "accurate|inaccurate|partially_accurate",
  "confidence": <float 0.0-1.0>,
  "reasoning": "1-2 sentences citing the specific evidence",
  "correction": "corrected text if inaccurate or partially_accurate, else null"
}}

Extract every claim. A claim is one distinct factual or interpretive assertion —
typically one sentence or independent clause. Return ONLY valid JSON.
"""


def _prompt_rewrite(
    section_name: str,
    original_text: str,
    corrections: List[Dict[str, str]],
    source_context: str = "",
    locked_claims: List[Dict[str, str]] = None,
) -> str:
    if not corrections:
        return f"Return the following text unchanged:\n\n{original_text}"

    lines = []
    for i, c in enumerate(corrections, 1):
        lines.append(
            f"{i}. Claim type: {c.get('claim_type', 'unknown')}\n"
            f"   Verdict: {c.get('verdict', 'unknown')}\n"
            f"   Why it failed: {c.get('reasoning', 'No reasoning provided')}\n"
            f"   Replace: {c['original']}\n"
            f"   With: {c['corrected']}"
        )
    corrections_text = "\n\n".join(lines)

    # Build locked claims section
    locked_section = ""
    if locked_claims:
        locked_lines = []
        for i, lc in enumerate(locked_claims, 1):
            locked_lines.append(
                f"{i}. {lc.get('claim_text', '')}"
            )
        locked_text = "\n".join(locked_lines)
        locked_section = f"""
=== CLAIMS THAT PASSED VALIDATION (PRESERVE EXACTLY) ===

The following claims passed validation in a previous iteration and MUST be preserved
word-for-word in your rewrite. Do NOT rephrase, shorten, or modify these in any way:

{locked_text}

"""

    source_section = ""
    if source_context:
        source_section = f"""
=== SOURCE DATA (verify corrections against this) ===

{source_context}

"""

    prompt = f"""Rewrite the following {section_name} to fix the specific errors listed below.

Rules:
1. LOCKED CLAIMS — Any claim text listed under "CLAIMS THAT PASSED VALIDATION" MUST be preserved exactly as written. These claims are accurate and must not be changed in any way.
2. CORRECTIONS ARE MANDATORY — every listed correction MUST be applied, no exceptions.
3. For each correction, find the exact phrase under "Replace:" in the original text and substitute it with the text under "With:". If the phrase spans part of a sentence, rewrite only that portion and preserve the rest of the sentence. If the "With:" text contains multiple sentences, condense it to exactly one sentence before applying.
4. Do NOT merge, combine, or consolidate separate sentences into one — each original sentence must remain a distinct sentence in the output.
5. For each correction, verify the suggested replacement against the SOURCE DATA.
   - If it matches the source data → apply it exactly as written
   - If it conflicts with the source data → use the source data value instead
   - If source data is absent → apply the suggested correction as-is
6. All sentences with no listed corrections must be kept word-for-word identical.
7. Do not add or remove any sentences. Every sentence in the original must appear as a separate sentence in the rewrite.
8. The number of sentences in your output MUST EXACTLY EQUAL the number of sentences in the original text — no more, no less. Never split one sentence into two. Never increase the sentence count.
9. Preserve the original tone and length as much as possible.
10. Return only the rewritten text — no explanations, no JSON wrapper.

=== EXAMPLE OF CORRECT BEHAVIOUR ===
Original sentence: "Studies have linked GENE1 to cancer pathways and recent work shows promising results."
Correction: Replace "Studies have linked GENE1 to cancer pathways" With "No studies directly link GENE1 to cancer pathways"
Correct output: "No studies directly link GENE1 to cancer pathways and recent work shows promising results."
(Only the flagged fragment is replaced; the rest of the sentence is preserved.)
===
{source_section}{locked_section}
=== CORRECTIONS TO APPLY ===

{corrections_text}

=== ORIGINAL TEXT ===

{original_text}
"""

    return prompt


# ───────────────────────────────────────────────────────────────────────────
# Claim counting / accuracy
# ───────────────────────────────────────────────────────────────────────────

def _count_verdicts(claims: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"accurate": 0, "inaccurate": 0, "partial": 0}
    for c in claims:
        v = (c.get("verdict") or "").lower()
        if v == "accurate":
            counts["accurate"] += 1
        elif v == "partially_accurate":
            counts["partial"] += 1
        else:
            # inaccurate, or any unrecognised verdict, counts as inaccurate
            counts["inaccurate"] += 1
    return counts


def _accuracy(claims: List[Dict[str, Any]]) -> Optional[float]:
    c = _count_verdicts(claims)
    # All claims — accurate, inaccurate, and partially_accurate — count toward the denominator.
    # 100% accuracy requires every claim to be fully accurate.
    total = c["accurate"] + c["inaccurate"] + c["partial"]
    if total == 0:
        return None
    return (c["accurate"] / total) * 100


# ───────────────────────────────────────────────────────────────────────────
# Build the iteration dict Streamlit reads
# ───────────────────────────────────────────────────────────────────────────

def _match_claim_lineage(
    claims_pass: List[Dict[str, Any]],
    prev_iter_entries: List[Dict[str, Any]],
    corrections: List[Dict[str, Any]],
    prev_accurate_texts: set,
    section_name: str,
) -> None:
    """Stamp original_claim_id, parent_claim_id, final_status on each claim dict in claims_pass.

    Two deterministic rules, applied in order:
      1. Preserved-accurate: claim_text matches a prior accurate claim (hard-freeze set) →
         parent = that prior entry's claim_id, original_claim_id carried forward.
      2. Rewritten: claim_text == _first_sentence(correction["corrected"]) for some correction →
         parent = correction["source_claim_id"], original_claim_id carried forward from source.
      Anything else → orphan_introduced (new original_claim_id, parent=None).
    """
    _prev_text_to_entry = {
        (e.get("claim_text") or "").strip(): e for e in prev_iter_entries
    }
    _prev_cid_to_entry = {
        e.get("claim_id"): e for e in prev_iter_entries
    }
    _rewritten_text_to_correction = {
        _first_sentence(corr.get("corrected", "")).strip(): corr
        for corr in corrections
        if corr.get("corrected")
    }

    for i, claim in enumerate(claims_pass, 1):
        cid = f"{section_name}_{i}"
        ctext = (claim.get("claim_text") or "").strip()

        # Rule 1: preserved-accurate
        if ctext in prev_accurate_texts and ctext in _prev_text_to_entry:
            src = _prev_text_to_entry[ctext]
            claim["parent_claim_id"] = src.get("claim_id")
            claim["original_claim_id"] = src.get("original_claim_id", src.get("claim_id"))
            claim["final_status"] = "preserved_accurate"
            continue

        # Rule 2: rewritten (first-sentence-of-correction match)
        if ctext in _rewritten_text_to_correction:
            corr = _rewritten_text_to_correction[ctext]
            src_cid = corr.get("source_claim_id")
            src = _prev_cid_to_entry.get(src_cid)
            claim["parent_claim_id"] = src_cid
            claim["original_claim_id"] = (
                src.get("original_claim_id", src_cid) if src else src_cid
            )
            claim["final_status"] = "rewritten"
            continue

        # Orphan
        claim["original_claim_id"] = cid
        claim["parent_claim_id"] = None
        claim["final_status"] = "orphan_introduced"


def _build_iteration(
    index: int,
    section_name: str,
    text: str,
    claims: List[Dict[str, Any]],
    model: str,
    improvement: Optional[float] = None,
) -> Dict[str, Any]:
    counts = _count_verdicts(claims)
    acc = _accuracy(claims)

    all_entries: List[Dict[str, Any]] = []
    inaccurate_entries: List[Dict[str, Any]] = []
    partial_entries: List[Dict[str, Any]] = []

    for i, claim in enumerate(claims, 1):
        cid = f"{section_name}_{i}"
        verdict = (claim.get("verdict") or "accurate").lower()
        # Lineage baseline: iter 0 stamps each claim dict with original_claim_id=cid, parent=None.
        # Iter>0 expects the matcher to have already stamped these; fall back to baseline otherwise.
        if index == 0:
            claim.setdefault("original_claim_id", cid)
            claim.setdefault("parent_claim_id", None)
            claim.setdefault("final_status", None)
        entry = {
            "claim_id": cid,
            "claim_text": claim.get("claim_text", ""),
            "claim_type": claim.get("claim_type", "other"),
            "verdict": verdict,
            "is_accurate": verdict in ("accurate", "partially_accurate"),
            "confidence": claim.get("confidence", 0.8),
            "reasoning": claim.get("reasoning", ""),
            "original_sentence": claim.get("claim_text", ""),
            "original_claim_id": claim.get("original_claim_id", cid),
            "parent_claim_id": claim.get("parent_claim_id"),
            "final_status": claim.get("final_status"),
        }
        all_entries.append(entry)

        if verdict == "inaccurate":
            inaccurate_entries.append({
                "claim_id": cid,
                "claim_text": claim.get("claim_text", ""),
                "issue": claim.get("reasoning", ""),
                "correction": claim.get("correction"),
                "confidence": claim.get("confidence", 0.8),
            })
        elif verdict == "partially_accurate":
            partial_entries.append({
                "claim_id": cid,
                "claim_text": claim.get("claim_text", ""),
                "issue": claim.get("reasoning", ""),
                "correction": claim.get("correction"),
                "confidence": claim.get("confidence", 0.8),
            })

    return {
        "iteration": index,
        "timestamp": datetime.now().isoformat(),
        "summary_text": text,
        "claims_extracted": len(claims),
        "claims_accurate": counts["accurate"],
        "claims_inaccurate": counts["inaccurate"],
        "claims_partially_accurate": counts["partial"],
        "accuracy_percentage": acc,
        "counts": counts,
        "model_used": model,
        "improvement_from_previous": improvement,
        "was_refined": index > 0,
        "preserved_count": 0,
        "refined_count": 0,
        "corrections_made": 0,
        "hallucinations_corrected": counts["inaccurate"],
        "is_existence_check": False,
        "all_claims": all_entries,
        "inaccurate_claims": inaccurate_entries,
        "partially_accurate_claims": partial_entries,
        "refinement_changes": [],
    }


# ───────────────────────────────────────────────────────────────────────────
# Failure helper
# ───────────────────────────────────────────────────────────────────────────

def _fail(section_name: str, section_text: str, error: Exception | str) -> Dict[str, Any]:
    return {
        "display_name": SECTION_DISPLAY_NAMES.get(section_name, section_name),
        "status": "failed",
        "error_message": str(error),
        "initial_accuracy": None,
        "final_accuracy": None,
        "improvement": None,
        "original_summary": section_text,
        "final_summary": section_text,
        "claims_summary": {},
        "iterations": [],
    }


# ───────────────────────────────────────────────────────────────────────────
# Core: validate one section
# ───────────────────────────────────────────────────────────────────────────

async def validate_section(
    section_name: str,
    section_text: str,
    gene_symbol: str,
    state_data: Dict[str, Any],
    *,
    state_ref=None,
    node_name: Optional[str] = None,
    source_context_override: str = "",
    lite_mode: bool = False,
) -> Dict[str, Any]:
    """Validate a single section. Returns a Streamlit-ready dict.

    Args:
        source_context_override: If provided, use this as the source context
            instead of building it from state_data. Useful for literature
            finding validation where the abstract is the ground truth.
        lite_mode: If True, return after Pass 1 without attempting rewrite
            or Pass 2 re-validation. Useful for short claims (e.g., literature
            key findings) where full remediation is overkill.
    """

    # ── Empty section → skip ──
    if not section_text or not section_text.strip():
        return {
            "display_name": SECTION_DISPLAY_NAMES.get(section_name, section_name),
            "status": "no_claims",
            "error_message": "no_claims_extracted",
            "initial_accuracy": None,
            "final_accuracy": None,
            "improvement": None,
            "original_summary": section_text,
            "final_summary": section_text,
            "claims_summary": {},
            "iterations": [],
        }

    try:
        client = _get_client()
        source = (
            source_context_override
            if source_context_override
            else _build_source_context(state_data, gene_symbol, section_name)
        )
        max_tok = _max_tokens_for(section_text, section_name)

        # ── Pass 1: validate ──
        prompt1 = _prompt_validate(section_name, section_text, source, gene_symbol)
        raw1 = await _api_validate(
            client,
            prompt1,
            max_tok,
            f"{gene_symbol}/{section_name} pass1",
            state_ref=state_ref,
            node_name=node_name,
        )
        parsed1 = _parse_json(raw1, f"{section_name} pass1")
        claims1 = parsed1.get("claims", [])
        iter1 = _build_iteration(0, section_name, section_text, claims1, get_active_model())

        acc1 = iter1["accuracy_percentage"]
        needs_fix = any(
            c.get("verdict") in ("inaccurate", "partially_accurate")
            and c.get("correction")
            for c in claims1
        )

        # ── Flag inaccurate claims that have no correction ──
        unflagged_inaccuracies = set(
            c.get("claim_text", "")
            for c in claims1
            if c.get("verdict") in ("inaccurate", "partially_accurate")
            and not c.get("correction")
        )
        # Mark flagged claims in iteration entries (for UI display in validation panel)
        if unflagged_inaccuracies:
            for entry in iter1.get("all_claims", []):
                if entry.get("claim_text") in unflagged_inaccuracies:
                    entry["flagged"] = True
                    entry["flag_reason"] = "unresolved inaccuracy — no correction available"
            for entry in iter1.get("inaccurate_claims", []):
                if entry.get("claim_text") in unflagged_inaccuracies:
                    entry["flagged"] = True
                    entry["flag_reason"] = "unresolved inaccuracy — no correction available"
            for entry in iter1.get("partially_accurate_claims", []):
                if entry.get("claim_text") in unflagged_inaccuracies:
                    entry["flagged"] = True
                    entry["flag_reason"] = "unresolved inaccuracy — no correction available"

        # Record refinement changes on iter1 (Streamlit shows them)
        corrections = [
            {
                "original": c.get("claim_text", ""),
                "corrected": c.get("correction", ""),
                "reasoning": c.get("reasoning", ""),
                "verdict": c.get("verdict", ""),
                "claim_type": c.get("claim_type", ""),
                "source_claim_id": f"{section_name}_{i}",
            }
            for i, c in enumerate(claims1, 1)
            if c.get("verdict") in ("inaccurate", "partially_accurate")
            and c.get("correction")
        ]

        iter1["refinement_changes"] = [
            {
                "original": corr["original"],
                "refined": corr["corrected"],
                "reason": next(
                    (
                        c.get("reasoning", "")
                        for c in claims1
                        if c.get("claim_text") == corr["original"]
                    ),
                    "",
                ),
                "claim_id": f"{section_name}_{i + 1}",
            }
            for i, corr in enumerate(corrections)
        ]

        # ── Pass 1 sufficient? (or lite_mode — skip remediation) ──
        # Short-circuit when: nothing to fix, OR already passing, OR truly empty, OR lite_mode.
        _sc_no_issues = not needs_fix
        _sc_passed = acc1 is None or acc1 >= ACCURACY_THRESHOLD
        if _sc_no_issues or _sc_passed or lite_mode:
            # Determine status
            if acc1 is None:
                status = "no_claims"
            elif lite_mode and needs_fix and acc1 < ACCURACY_THRESHOLD:
                status = "lite_partial"
            else:
                status = "passed" if acc1 >= ACCURACY_THRESHOLD else "partial"
            return {
                "display_name": SECTION_DISPLAY_NAMES.get(section_name, section_name),
                "status": status,
                "error_message": None,
                "initial_accuracy": acc1,
                "final_accuracy": acc1,
                "improvement": 0.0,
                "original_summary": section_text,
                "final_summary": section_text,
                "claims_summary": {
                    "initial_claim_count": len(claims1),
                    "final_claim_count": len(claims1),
                    "accurate_count": iter1["counts"]["accurate"],
                    "inaccurate_count": iter1["counts"]["inaccurate"],
                    "partial_count": iter1["counts"]["partial"],
                },
                "iterations": [iter1],
            }

        # ── Multi-pass refinement loop (up to MAX_REWRITE_ATTEMPTS) ──
        iterations = [iter1]
        current_text = section_text
        current_claims = claims1
        previous_acc = acc1
        text_changed = False

        # Track best result across all iterations (in case later attempts degrade)
        best_iteration = iter1
        best_acc = acc1
        best_text = section_text

        # Loop for rewrite attempts (passes 2 through 1+MAX_REWRITE_ATTEMPTS)
        for attempt_num in range(1, MAX_REWRITE_ATTEMPTS + 1):
            pass_num = attempt_num + 1  # Pass number (2, 3, 4, ...)
            corrections = [
                {
                    "original": c.get("claim_text", ""),
                    "corrected": c.get("correction", ""),
                    "reasoning": c.get("reasoning", ""),
                    "verdict": c.get("verdict", ""),
                    "claim_type": c.get("claim_type", ""),
                    "source_claim_id": f"{section_name}_{i}",
                }
                for i, c in enumerate(current_claims, 1)
                if c.get("verdict") in ("inaccurate", "partially_accurate")
                and c.get("correction")
            ]

            # Pre-process: truncate any multi-sentence correction to its first sentence.
            # Multi-sentence replacements cause the rewriter to expand the text, fragmenting
            # one claim into multiple claims on re-validation.
            for corr in corrections:
                corr["corrected"] = _first_sentence(corr["corrected"])

            # Extract accurate claims to lock/preserve in rewrite
            locked_claims = [
                {
                    "claim_text": c.get("claim_text", ""),
                    "claim_type": c.get("claim_type", ""),
                }
                for c in current_claims
                if c.get("verdict") == "accurate"
            ]

            # Collect accurate claim texts for hard-freeze after re-validation
            _prev_accurate_texts = {
                c.get("claim_text", "").strip()
                for c in current_claims
                if c.get("verdict") == "accurate" and c.get("claim_text")
            }

            # If no corrections available, stop
            if not corrections:
                break

            # Rewrite with locked claims protection
            rewrite_prompt = _prompt_rewrite(section_name, current_text, corrections, source, locked_claims)
            rewritten = await _api_rewrite(
                client,
                rewrite_prompt,
                current_text,
                state_ref=state_ref,
                node_name=node_name,
            )

            # Calculate sentence counts for fragmentation check (applied after validation)
            _orig_sent_count = _count_sentences(current_text)
            _new_sent_count = _count_sentences(rewritten)
            _sentence_expanded = _new_sent_count > _orig_sent_count

            # Re-validate
            validate_prompt = _prompt_validate(section_name, rewritten, source, gene_symbol)
            max_tok_pass = _max_tokens_for(rewritten, section_name)
            try:
                raw_pass = await _api_validate(
                    client,
                    validate_prompt,
                    max_tok_pass,
                    f"{gene_symbol}/{section_name} pass{pass_num}",
                    state_ref=state_ref,
                    node_name=node_name,
                )
            except ValueError as exc:
                message = str(exc)
                if "finish_reason='length'" in message:
                    print(
                        f"  ⚠️  Pass {pass_num} truncated for {gene_symbol}/{section_name} — "
                        f"using best result from previous passes"
                    )
                    break
                raise

            parsed_pass = _parse_json(raw_pass, f"{section_name} pass{pass_num}")
            claims_pass = parsed_pass.get("claims", [])

            # Hard-freeze: restore accurate verdict for any claim whose text matches
            # a previously-accurate claim, regardless of what the re-validator said.
            if _prev_accurate_texts:
                for _c in claims_pass:
                    if _c.get("claim_text", "").strip() in _prev_accurate_texts:
                        _c["verdict"] = "accurate"
                        _c["correction"] = ""
                        _c["reasoning"] = "Preserved from previous iteration"

            # Stamp lineage (original_claim_id, parent_claim_id, final_status) on claims_pass
            # before building the iteration, so _build_iteration can carry them onto entries.
            _match_claim_lineage(
                claims_pass,
                iterations[-1].get("all_claims", []),
                corrections,
                _prev_accurate_texts,
                section_name,
            )

            acc_pass = _accuracy(claims_pass)

            # Post-rewrite guard: only reject if sentence count expanded AND accuracy dropped.
            # Accept rewrites that expand sentences if they maintain/improve accuracy.
            if _sentence_expanded:
                if acc_pass is not None and previous_acc is not None and acc_pass < previous_acc:
                    # Sentence expanded AND accuracy dropped - reject rewrite
                    print(
                        f"  ⚠️  Pass {pass_num}: rejected rewrite for {gene_symbol}/{section_name} "
                        f"— sentence count expanded AND accuracy dropped (keeping previous)"
                    )
                    continue
                else:
                    # Sentence expanded BUT accuracy held/improved - accept with note
                    print(
                        f"  ⚠️  Pass {pass_num}: accepted rewrite for {gene_symbol}/{section_name} "
                        f"— sentence count expanded but accuracy held/improved"
                    )

            iter_pass = _build_iteration(
                pass_num - 1,
                section_name,
                rewritten,
                claims_pass,
                get_active_model(),
                improvement=(
                    (acc_pass - previous_acc)
                    if (acc_pass is not None and previous_acc is not None)
                    else None
                ),
            )

            # Check if text actually changed
            if SequenceMatcher(None, current_text.strip(), rewritten.strip()).ratio() < 0.99:
                text_changed = True
                iter_pass["refined_count"] = len(corrections)
                iter_pass["corrections_made"] = len(corrections)

            iterations.append(iter_pass)
            current_text = rewritten
            current_claims = claims_pass
            previous_acc = acc_pass

            # Track best result (highest accuracy)
            if acc_pass is not None and (best_acc is None or acc_pass > best_acc):
                best_iteration = iter_pass
                best_acc = acc_pass
                best_text = rewritten

            # Stop early only when 100% accuracy AND no partial/inaccurate claims remain.
            # _accuracy() excludes partially_accurate from its denominator, so acc_pass
            # can reach 100.0 while partial claims still exist — check explicitly.
            if acc_pass is not None and acc_pass >= 100.0:
                remaining_issues = sum(
                    1 for c in claims_pass
                    if c.get("verdict") in ("partially_accurate", "inaccurate")
                )
                if remaining_issues == 0:
                    break

        # ── Build final result using best iteration across all attempts ──
        final_iteration = best_iteration
        final_acc = best_acc
        final_text = best_text
        final_claims = final_iteration.get("all_claims", [])

        if final_acc is not None and final_acc >= ACCURACY_THRESHOLD:
            status = "corrected" if text_changed else "passed"
        else:
            status = "partial"

        return {
            "display_name": SECTION_DISPLAY_NAMES.get(section_name, section_name),
            "status": status,
            "error_message": None,
            "initial_accuracy": acc1,
            "final_accuracy": final_acc,
            "improvement": (final_acc - acc1) if (final_acc is not None and acc1 is not None) else None,
            "original_summary": section_text,
            "final_summary": final_text,
            "claims_summary": {
                "initial_claim_count": len(claims1),
                "final_claim_count": len(final_claims),
                "accurate_count": final_iteration["counts"]["accurate"],
                "inaccurate_count": final_iteration["counts"]["inaccurate"],
                "partial_count": final_iteration["counts"]["partial"],
            },
            "iterations": iterations,
        }

    except Exception as exc:
        print(f"      ❌ validate_section failed for {gene_symbol}/{section_name}: {exc}")
        return _fail(section_name, section_text, exc)
