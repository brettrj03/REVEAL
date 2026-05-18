"""
Agent for generating academic-style individual gene summaries.
Grounded version using GO definitions to reduce hallucination.
"""

import asyncio
import re
from typing import Dict, List
from openai import AsyncOpenAI
import os
from src.graph.state import _accumulate_tokens
from src.config import get_active_model

_client = None


def get_client():
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


# -------------------------------
# Utility: format GO terms safely
# -------------------------------

def _format_go_terms(go_terms: List[Dict], namespace: str, max_terms: int = 5) -> List[str]:
    """Format GO terms with ID + name + definition for grounding."""
    filtered = [
        g for g in go_terms
        if g.get("namespace") == namespace
        and g.get("definition")
    ]

    # Deduplicate by GO ID
    seen = set()
    formatted = []
    for g in filtered:
        go_id = g.get("go_id")
        if go_id not in seen:
            seen.add(go_id)
            formatted.append(
                f"{go_id} | {g.get('name')}: {g.get('definition')}"
            )

    return formatted[:max_terms]


# -------------------------------
# Gene Description
# -------------------------------

async def generate_gene_description(
    gene_data: Dict,
    state=None,
    node_name: str = "InterpretAllGenes",
) -> str:
    gene_symbol = gene_data.get("gene_symbol", "Unknown")
    full_name = gene_data.get("full_name", "")
    go_terms = gene_data.get("go_terms", [])
    interactions = gene_data.get("interactions", [])
    literature = gene_data.get("literature", [])

    has_go = len(go_terms) > 0
    has_interactions = len(interactions) > 0
    has_literature = len(literature) > 0
    evidence_count = sum([has_go, has_interactions, has_literature])

    # Check for hallucination-prone pattern (interactions/literature but no GO terms)
    if not has_go and (has_interactions or has_literature):
        parts = []
        if has_interactions:
            partner_examples = [
                (entry.get('partner_symbol') or entry.get('partner') or '').strip()
                for entry in interactions[:2]
            ]
            partner_examples = [p for p in partner_examples if p]
            if partner_examples:
                partner_sample = ", ".join(partner_examples)
                interaction_note = (
                    f"{len(interactions)} protein interaction partners (e.g. {partner_sample})"
                )
            else:
                interaction_note = f"{len(interactions)} protein interaction partners"
            parts.append(interaction_note)
        if has_literature:
            parts.append(f"{len(literature)} literature records")

        evidence_str = " and ".join(parts)
        return (
            f"No GO annotations are recorded for {gene_symbol} in the Gene Ontology database. "
            f"Available evidence includes {evidence_str}. Functional characterisation based on GO "
            "terms is therefore not possible, though interaction and literature data provide "
            "limited contextual information."
        )

    if evidence_count == 0:
        return (
            "[LOW_EVIDENCE] No functional annotations, protein interactions, or literature "
            f"are currently available for {gene_symbol} in the queried databases."
        )

    if not has_go and evidence_count == 1:
        available = (
            f"{len(interactions)} interaction partners"
            if has_interactions else
            f"{len(literature)} literature records"
        )
        return (
            f"[LOW_EVIDENCE] No GO annotations are recorded for {gene_symbol}. Only {available} "
            "are available in the queried databases."
        )

    mf_terms = _format_go_terms(go_terms, "molecular_function", 5)
    bp_terms = _format_go_terms(go_terms, "biological_process", 5)

    prompt = f"""
Generate a concise 2–3 sentence academic description for the gene {gene_symbol} ({full_name}).

You MUST use only the provided GO term names and definitions.
Do NOT introduce mechanisms not directly supported by the definition text.
If the definitions are too generic to support mechanistic detail, remain high-level.

Molecular Functions:
{chr(10).join(mf_terms) if mf_terms else "None available"}

Biological Processes:
{chr(10).join(bp_terms) if bp_terms else "None available"}

Rules:
- 2–3 sentences maximum
- Academic tone
- Do NOT include clinical information
- Do NOT mention the gene symbol
- Do NOT introduce pathways or complexes unless explicitly named in definitions
- Do NOT list or cite GO term IDs in the summary text
- Do NOT end with "Evidence used: GO:XXXXXXX" or any similar citation of GO identifiers
- The GO terms are displayed separately in the UI
"""

    client = get_client()
    response = await client.chat.completions.create(
        model=get_active_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=400,
    )
    _accumulate_tokens(state, node_name, getattr(response, "usage", None))

    summary = response.choices[0].message.content.strip()

    # Post-processing: strip any "Evidence used: GO:XXXXXXX" citations
    summary = re.sub(
        r'\s*Evidence used:[\s\w:,\.]*GO:\d+[\w:,\.\s]*\.?',
        '',
        summary,
        flags=re.IGNORECASE
    ).strip()

    return summary


# -------------------------------
# Functional Interpretation
# -------------------------------

async def generate_functional_interpretation(
    gene_data: Dict,
    state=None,
    node_name: str = "InterpretAllGenes",
) -> str:
    gene_symbol = gene_data.get("gene_symbol", "Unknown")
    go_terms = gene_data.get("go_terms", [])
    interactions = gene_data.get("interactions", [])
    literature = gene_data.get("literature", [])

    has_go = len(go_terms) > 0
    has_interactions = len(interactions) > 0
    has_literature = len(literature) > 0

    if not has_go and not has_interactions:
        return (
            f"[LOW_EVIDENCE] Insufficient annotation data to generate experimental relevance "
            f"context for {gene_symbol}."
        )

    if not has_go:
        return (
            "No GO annotations are currently recorded in the Gene Ontology database for this gene. "
            "Functional characterisation cannot be determined from annotation data alone."
        )

    evidence_lines = []
    if not has_go:
        evidence_lines.append("- GO annotations: NONE")
    else:
        evidence_lines.append(f"- GO annotations: {len(go_terms)} terms")

    if has_interactions:
        evidence_lines.append(
            f"- Protein interactions: {len(interactions)} partners"
        )
    else:
        evidence_lines.append("- Protein interactions: NONE")

    if has_literature:
        evidence_lines.append(
            f"- Literature: {len(literature)} records"
        )
    else:
        evidence_lines.append("- Literature: NONE")

    evidence_inventory = "\n".join(evidence_lines)

    mf_terms = _format_go_terms(go_terms, "molecular_function", 5)
    bp_terms = _format_go_terms(go_terms, "biological_process", 5)
    cc_terms = _format_go_terms(go_terms, "cellular_component", 3)

    prompt = f"""
Generate a 3–4 sentence functional interpretation for {gene_symbol}
using ONLY the provided GO term names and definitions.

Available evidence for {gene_symbol}:
{evidence_inventory}

STRICT RULES:
- You MUST NOT state that no evidence was available — list what was consulted even if limited
- If GO annotations are absent, say so specifically but acknowledge what other evidence exists
- Base your summary only on the evidence listed above
- 2–3 sentences maximum
- If only interactions and literature are available with no GO terms, summarise what the interactions and literature suggest about the gene's function

Molecular Functions:
{chr(10).join(mf_terms) if mf_terms else "None"}

Biological Processes:
{chr(10).join(bp_terms) if bp_terms else "None"}

Cellular Components:
{chr(10).join(cc_terms) if cc_terms else "None"}

ADDITIONAL REQUIREMENTS:
- Academic tone
- Mechanistic statements must be directly supported by definition text
- Do NOT introduce external pathways or complexes unless explicitly named
- Do NOT include disease information
- If definitions are insufficient, explicitly state that mechanistic detail is limited
- Include ONE sentence about the cellular component localization and what it may suggest about the gene's biological role (e.g., "Localization to X suggests involvement in Y" or "Its presence in Z is consistent with a role in W")
- Do NOT end with "Evidence used:" or list GO IDs
"""

    client = get_client()
    response = await client.chat.completions.create(
        model=get_active_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=500,
    )
    _accumulate_tokens(state, node_name, getattr(response, "usage", None))

    return response.choices[0].message.content.strip()


# -------------------------------
# Expression Interpretation
# -------------------------------

async def generate_expression_interpretation(
    gene_data: Dict,
    state=None,
    node_name: str = "InterpretAllGenes",
) -> str:
    gene_symbol = gene_data.get("gene_symbol", "Unknown")
    expression_data = gene_data.get("expression_data", [])

    if not expression_data:
        return f"[LOW_EVIDENCE] No expression data available for {gene_symbol} in GTEx."

    # Compute statistics for the prompt
    tpm_values = [e.get("tpm", 0) for e in expression_data]
    mean_tpm = sum(tpm_values) / len(tpm_values) if tpm_values else 0
    min_tpm = min(tpm_values) if tpm_values else 0
    max_tpm = max(tpm_values) if tpm_values else 0
    num_tissues = len(expression_data)

    sorted_expr = sorted(expression_data, key=lambda x: x.get("tpm", 0), reverse=True)[:5]

    expr_summary = "\n".join(
        [f"{e['tissue']}: {e['tpm']:.2f} TPM" for e in sorted_expr]
    )

    # Determine if there's a clear outlier (>3x mean) - used for guidance
    has_outlier = max_tpm > (3 * mean_tpm) if mean_tpm > 0 else False
    expression_level = "low" if mean_tpm < 1 else ("moderate" if mean_tpm < 10 else "high")

    prompt = f"""
Analyse the expression pattern for {gene_symbol}.

IMPORTANT CONTEXT: Expression data is from GTEx and covers {num_tissues} specific tissue types. This is not a complete atlas of all human tissues — absence of a tissue does not mean the gene is unexpressed there. Frame all expression statements relative to the tissues measured.

Top expression tissues (within GTEx dataset):
{expr_summary}

STRICT RULES:
- 2–3 sentences, academic tone, paragraph form only
- Do NOT infer specific biological mechanisms unless phrased as cautious hypothesis ("may suggest")

LANGUAGE CONSTRAINTS (critical for validation):
- NEVER mention mean TPM, range, or specific TPM values unless one tissue is a clear outlier
- NEVER say "the mean TPM across all tissues is X"
- NEVER say "with a range from X to Y TPM"
- ONLY mention numbers if there is a striking difference between tissues that is essential to understanding the pattern — otherwise describe the pattern in words only (e.g. "LMNA is highly expressed in fibroblasts and vascular tissues, with notably lower levels elsewhere")
- NEVER say expression is "restricted" — instead say "enriched in specific tissues within the GTEx dataset" or "shows higher expression in X tissues relative to others measured"
- NEVER say TPM values are "identical" or "uniform" — if expression is similar across tissues, say "consistently {expression_level} across measured tissues with little variation"
- ALWAYS frame highest-expressing tissues as "highest within the GTEx dataset" or "among the tissues measured"
- If expression is uniformly low, note that this suggests either low overall expression or expression in tissues not captured by GTEx
- Only claim tissue specificity if there is a clear outlier tissue{"" if not has_outlier else " (outlier detected in this data)"}

Write only the paragraph.
"""

    client = get_client()
    response = await client.chat.completions.create(
        model=get_active_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=300,
    )
    _accumulate_tokens(state, node_name, getattr(response, "usage", None))

    return response.choices[0].message.content.strip()


# -------------------------------
# Network Interpretation
# -------------------------------

async def generate_network_interpretation(
    gene_data: Dict,
    state=None,
    node_name: str = "InterpretAllGenes",
) -> str:
    gene_symbol = gene_data.get("gene_symbol", "Unknown")
    interactions = gene_data.get("interactions", [])

    if not interactions:
        return f"[LOW_EVIDENCE] No protein interactions are recorded for {gene_symbol} in STRING."

    def _score(entry: Dict) -> float:
        score = entry.get("score")
        if score is not None:
            return float(score)
        combined = entry.get("combined_score")
        if combined is not None:
            combined = float(combined)
            return combined / 1000.0 if combined > 1 else combined
        return 0.0

    def _partner(entry: Dict) -> str:
        return entry.get("partner_symbol") or entry.get("partner") or "Unknown"

    sorted_int = sorted(interactions, key=_score, reverse=True)[:5]

    total_interactions = len(interactions)
    avg_score = sum(_score(i) for i in interactions) / len(interactions)
    top_partners = ", ".join([_partner(i) for i in sorted_int])

    # Determine qualitative confidence level for guidance
    if avg_score > 0.8:
        confidence_note = "Confidence is notably high — you may mention this qualitatively if relevant."
    elif avg_score < 0.4:
        confidence_note = "Confidence is notably low — you may mention this qualitatively if relevant."
    else:
        confidence_note = ""

    prompt = f"""
Analyse the protein interaction network for {gene_symbol}.

Total interactions: {total_interactions}
Top partners: {top_partners}

Rules:
- EXACTLY 2–3 sentences
- Academic tone
- Do NOT list all partners
- Do NOT claim pathway/complex membership unless strongly implied by ≥2 partners
- Avoid speculative mechanisms
- Do NOT state the average confidence score as a number
- If mentioning confidence, describe it qualitatively only: "with generally high confidence" (>0.7) or "with moderate confidence" (0.5-0.7) or "with lower confidence" (<0.5)
- Only mention confidence if it is notably high or notably low — otherwise omit it entirely
{confidence_note}

Write only the paragraph.
"""

    client = get_client()
    response = await client.chat.completions.create(
        model=get_active_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=300,
    )
    _accumulate_tokens(state, node_name, getattr(response, "usage", None))

    return response.choices[0].message.content.strip()


# -------------------------------
# Main Orchestration
# -------------------------------

async def summarize_single_gene(
    gene_data: Dict,
    state=None,
    node_name: str = "InterpretAllGenes",
) -> Dict[str, str]:
    gene_symbol = gene_data.get("gene_symbol", "Unknown")

    gene_desc, func_interp, expr_interp, net_interp = await asyncio.gather(
        generate_gene_description(gene_data, state=state, node_name=node_name),
        generate_functional_interpretation(gene_data, state=state, node_name=node_name),
        generate_expression_interpretation(gene_data, state=state, node_name=node_name),
        generate_network_interpretation(gene_data, state=state, node_name=node_name),
    )

    return {
        "gene_description": gene_desc,
        "functional_interpretation": func_interp,
        "expression_interpretation": expr_interp,
        "network_interpretation": net_interp,
    }


async def summarize_all_genes(
    all_gene_data: List[Dict],
    state=None,
    node_name: str = "InterpretAllGenes",
) -> Dict[str, Dict[str, str]]:
    tasks = [
        summarize_single_gene(gene_data, state=state, node_name=node_name)
        for gene_data in all_gene_data
    ]
    results = await asyncio.gather(*tasks)

    return {
        gene_data["gene_symbol"]: summary
        for gene_data, summary in zip(all_gene_data, results)
    }
