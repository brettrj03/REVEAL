"""
Unified report assembler - single format that adapts to mode.
Facts always shown, interpretations included when available.
"""

from __future__ import annotations

from typing import Iterable, List

from src.models.report_components import (
    GeneProfile,
    ReportMetadata,
    BiologicalSynthesis,
)
from src.reports.top_papers_formatter import format_top_papers
from src.reports.literature_analysis_formatter import format_literature_analysis_section as _format_literature_analysis_section
from src.reports.literature_findings_formatter import format_literature_findings_section, format_literature_findings_summary


def assemble_unified_report(
    gene_profiles: Iterable[GeneProfile],
    report_metadata: ReportMetadata | None,
    synthesis: BiologicalSynthesis | None,
    go_comparison_analysis: dict | None,
    network_overlap_analysis: dict | None,
    mode: str,
    pubmed_counts: dict | None = None,
    pubmed_pmids: dict | None = None,
    pubmed_records: dict | None = None,
    disease_filter: str = "",
    disease_terms: List[str] | None = None,
    gene_top_papers: dict | None = None,
    literature_findings_summary: dict | None = None,
) -> str:
    """Assemble a facts-first report with optional interpretations."""

    sections: List[str] = []

    # System Notice - Human Data Only Warning (at very top)
    sections.append("=" * 80)
    sections.append("⚠️  SYSTEM NOTICE: HUMAN DATA ONLY")
    sections.append("=" * 80)
    sections.append("This analysis uses human genome databases exclusively.")
    sections.append("All gene symbols, expression patterns, and protein interactions")
    sections.append("are from Homo sapiens. Results may not apply to other species.")
    sections.append("=" * 80)
    sections.append("")

    if report_metadata:
        sections.append(report_metadata.to_header_text())
    else:
        sections.append("=" * 80)
        sections.append("GENE ANALYSIS REPORT")
        sections.append("=" * 80)
        sections.append("No report metadata available")

    # REMOVED: Novelty & Plausibility Rankings section

    # Literature Findings Overview (What's Already Known)
    if literature_findings_summary:
        lit_overview = format_literature_findings_summary(literature_findings_summary)
        if lit_overview:
            sections.append(lit_overview)

    if synthesis and mode == "interpreted":
        sections.append("\n" + "-" * 80)
        sections.append("CROSS-GENE SYNTHESIS")
        sections.append("-" * 80)
        sections.append("")

        if synthesis.executive_summary:
            sections.append("EXECUTIVE SUMMARY:")
            sections.append(synthesis.executive_summary)

        if synthesis.key_findings:
            sections.append("\nKEY FINDINGS:")
            for finding in synthesis.key_findings:
                sections.append(f"• {finding}")

        if synthesis.cross_gene_insights:
            sections.append("\nCROSS-GENE INSIGHTS:")
            sections.append(synthesis.cross_gene_insights)

    sections.append("\n" + "=" * 80)
    sections.append("GENE-BY-GENE ANALYSIS")
    sections.append("=" * 80)

    for profile in gene_profiles:
        sections.append(profile.to_unified_format())

        # REMOVED: Novelty assessment
        # REMOVED: Plausibility assessment

        # Add literature findings section (What's Already Known)
        if literature_findings_summary and profile.gene_symbol in literature_findings_summary:
            findings_section = format_literature_findings_section(
                profile.gene_symbol,
                literature_findings_summary[profile.gene_symbol]
            )
            sections.append(findings_section)

        # Add literature analysis section with top papers
        if gene_top_papers and profile.gene_symbol in gene_top_papers:
            lit_section = _format_literature_analysis_section(
                profile.gene_symbol,
                gene_top_papers[profile.gene_symbol]
            )
            sections.append(lit_section)

        # Keep old literature section if available (for backward compatibility)
        if pubmed_counts and pubmed_pmids and pubmed_records:
            lit_section = _format_literature_section(
                profile.gene_symbol,
                pubmed_counts,
                pubmed_pmids,
                pubmed_records,
                disease_filter,
                disease_terms
            )
            if lit_section:
                sections.append(lit_section)

    if network_overlap_analysis:
        sections.append("\n" + "=" * 80)
        sections.append("CROSS-GENE NETWORK ANALYSIS")
        sections.append("=" * 80)
        sections.append("")
        sections.append(_format_network_overlap(network_overlap_analysis, mode))

    if go_comparison_analysis:
        sections.append("\n" + "=" * 80)
        sections.append("GENE ONTOLOGY OVERLAP ANALYSIS")
        sections.append("=" * 80)
        sections.append("")
        sections.append(_format_go_overlap(go_comparison_analysis, mode))

    sections.append("\n" + "=" * 80)
    sections.append("END OF REPORT")
    sections.append("=" * 80)

    return "\n".join(sections)


def _format_network_overlap(analysis: dict, mode: str) -> str:
    sections: List[str] = []
    hub_proteins = analysis.get("hub_proteins", [])
    direct_interactions = analysis.get("direct_interactions", [])
    stats = analysis.get("network_stats", {})

    sections.append("SHARED PROTEIN PARTNERS:\n")

    if hub_proteins:
        sections.append("Hub Proteins (interact with multiple query genes):")
        for hub in hub_proteins[:10]:
            partner = hub.get("protein") or hub.get("partner") or "Unknown"
            genes = ", ".join(hub.get("genes", []))
            sections.append(f"• {partner} - Genes: {genes}")

    if direct_interactions:
        sections.append("\nDirect Gene-Gene Interactions:")
        for interaction in direct_interactions:
            gene_a = interaction.get("gene_a") or interaction.get("gene1")
            gene_b = interaction.get("gene_b") or interaction.get("gene2")
            score = interaction.get("score") or interaction.get("confidence", 0)
            desc = interaction.get("description") or ""
            sections.append(
                f"• {gene_a} ↔ {gene_b} (score: {score:.3f}){f' - {desc}' if desc else ''}"
            )
        sections.append(f"Total direct interactions detected: {len(direct_interactions)}")

    if stats:
        sections.append("\nNetwork Statistics:")
        density = stats.get("density")
        if density is not None:
            sections.append(f"Density: {density:.2f}")
        shared = stats.get("shared_partners")
        unique = stats.get("unique_partners")
        if shared is not None:
            if unique:
                sections.append(
                    f"Shared partners: {shared} ({shared / unique * 100:.1f}% of unique partners)"
                )
            else:
                sections.append(f"Shared partners: {shared}")
        total_interactions = stats.get('total_interactions', 0)
        sections.append(f"Total interactions: {total_interactions}")
        hub_count = stats.get('hub_genes_count')
        if hub_count is not None:
            sections.append(f"Hub genes: {hub_count}")

    interpretation = analysis.get("interpretation")
    if interpretation and mode == "interpreted":
        sections.append("\nNETWORK INTERPRETATION:")
        sections.append(interpretation)

    return "\n".join(sections)


def _format_go_overlap(analysis: dict, mode: str) -> str:
    sections: List[str] = []
    shared_terms = analysis.get("shared_terms", [])
    stats = analysis.get("overlap_stats", {})

    sections.append("SHARED GO TERMS (present in 2+ genes):\n")

    if shared_terms:
        def _emit(category: str, label: str) -> None:
            terms = [t for t in shared_terms if t.get("category") == category]
            if not terms:
                return
            sections.append(label)
            for term in terms[:10]:
                genes = ", ".join(term.get("genes", []))
                sections.append(f"• {term.get('name')} ({term.get('go_id')}) - {genes}")
            sections.append("")

        _emit("biological_process", "Biological Process:")
        _emit("molecular_function", "Molecular Function:")
        _emit("cellular_component", "Cellular Component:")
    else:
        sections.append("No shared GO terms identified")

    if stats:
        sections.append("\nOverlap Statistics:")
        sections.append(f"Total unique GO terms: {stats.get('total_unique', 0)}")
        sections.append(
            f"Shared by all genes: {stats.get('shared_all', 0)} ({stats.get('shared_all_pct', 0):.1f}%)"
        )
        partial = stats.get('shared_partial', 0)
        if partial:
            sections.append(
                f"Shared by subset of genes (≥2 but not all): {partial} ({stats.get('shared_partial_pct', 0):.1f}%)"
            )
        sections.append(
            f"Gene-specific: {stats.get('gene_specific', 0)} ({stats.get('gene_specific_pct', 0):.1f}%)"
        )

    matrix = analysis.get("similarity_matrix")
    if matrix:
        genes = list(matrix.keys())
        sections.append("\nSimilarity Matrix:")
        header = "           " + "".join(f"{gene:>8}" for gene in genes)
        sections.append(header)
        for gene_a in genes:
            row = f"{gene_a:<10}"
            for gene_b in genes:
                value = matrix[gene_a].get(gene_b, 0)
                row += f"{value:>8.3f}"
            sections.append(row)

    interpretation = analysis.get("interpretation")
    if interpretation and mode == "interpreted":
        sections.append("\nGO OVERLAP INTERPRETATION:")
        sections.append(interpretation)

    return "\n".join(sections)


def _format_literature_section(
    gene_symbol: str,
    pubmed_counts: dict,
    pubmed_pmids: dict,
    pubmed_records: dict,
    disease_filter: str = "",
    disease_terms: List[str] = None
) -> str:
    """Format literature mining results for a single gene."""
    counts = pubmed_counts.get(gene_symbol, {})
    if not counts:
        return ""

    context_strict = counts.get("context_direct_strict", 0)
    context_broad = counts.get("context_direct_broad", 0)
    total_human = counts.get("total_gene_human")
    total_disease = counts.get("total_gene_human_disease")
    legacy_total = counts.get("total_gene") if total_human is None else None

    base_total = total_human if total_human is not None else legacy_total
    if (base_total is None or base_total == 0) and (total_disease is None or total_disease == 0):
        return ""

    def _pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    sections: List[str] = ["\n\nLITERATURE SUMMARY (PubMed):"]

    if total_human is not None:
        sections.append(f"Total publications (human): {total_human:,}")
    elif legacy_total is not None:
        sections.append(f"Total publications (legacy total_gene): {legacy_total:,}")

    if total_disease is not None:
        sections.append(f"Total publications (human + disease): {total_disease:,}")

    # Derived metrics
    if total_human and total_disease:
        disease_fraction = total_disease / total_human if total_human else 0.0
        sections.append(f"Disease fraction (disease / human): {_pct(disease_fraction)}")

    global_den = base_total if (base_total and base_total > 0) else None
    context_fraction_global = (context_strict / global_den) if global_den else 0.0
    sections.append(f"Context coverage (strict / global): {_pct(context_fraction_global)}")

    disease_den = total_disease if (total_disease and total_disease > 0) else global_den
    if disease_den:
        context_fraction_disease = context_strict / disease_den if disease_den else 0.0
        sections.append(f"Context coverage (strict / disease): {_pct(context_fraction_disease)}")

    sections.append(f"Context-specific (strict): {context_strict:,}")
    sections.append(f"Context-specific (broad): {context_broad:,}")

    # Show disease filter line (or None)
    if disease_filter and disease_terms:
        sections.append(f"Disease filter: {', '.join(disease_terms)}")
    elif disease_filter:
        sections.append(f"Disease filter: {disease_filter.strip()}")
    else:
        sections.append("Disease filter: None")

    # Get top 3 context_direct_strict papers first, then broad if not enough
    pmids_strict = pubmed_pmids.get(gene_symbol, {}).get("context_direct_strict", [])[:3]
    pmids_broad = pubmed_pmids.get(gene_symbol, {}).get("context_direct_broad", [])[:3]

    # Combine and deduplicate, prioritising strict
    all_pmids = []
    seen = set()
    for pmid in pmids_strict + pmids_broad:
        if pmid not in seen:
            all_pmids.append(pmid)
            seen.add(pmid)
            if len(all_pmids) >= 3:
                break

    if all_pmids:
        sections.append("\nTop Context-Specific Papers:")
        for pmid in all_pmids:
            rec = pubmed_records.get(pmid, {})
            title = rec.get("title", "Unknown title")
            year = rec.get("year", "n.d.")
            sections.append(f"- {title} ({year}) PMID: {pmid}")

    return "\n".join(sections)
