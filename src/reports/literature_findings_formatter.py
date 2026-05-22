"""
Literature Findings Formatter

Formats the literature analysis findings into a markdown section
for inclusion in the final report.
"""

from typing import Dict, Any, List


def format_literature_findings_section(
    gene_symbol: str,
    findings: Dict[str, Any]
) -> str:
    """
    Format literature findings for a single gene into report section.

    Args:
        gene_symbol: Gene symbol
        findings: Literature findings dictionary from AnalyzeLiteratureFindings

    Returns:
        Formatted markdown string
    """
    sections = []

    sections.append("\n" + "=" * 80)
    sections.append("LITERATURE CONTEXT (What's Already Known)")
    sections.append("=" * 80)

    if findings.get('analysis_status') == 'no_papers':
        sections.append("\nNo literature data available for analysis.")
        sections.append("")
        return "\n".join(sections)

    # =========================================================================
    # Publication Overview
    # =========================================================================
    sections.append("\n### Publication Overview\n")

    total = findings.get('total_papers', 0)
    recent = findings.get('recent_papers_count', 0)
    recent_years = findings.get('recent_years', 'recent')

    sections.append(f"Total papers analysed: {total:,}")
    sections.append(f"Recent publications ({recent_years}): {recent:,}")

    if total > 0:
        recent_pct = (recent / total) * 100
        sections.append(f"Recent activity: {recent_pct:.1f}% of papers from last 2 years")

        # Research activity indicator
        if recent_pct > 30:
            sections.append("Research status: ACTIVELY STUDIED")
        elif recent_pct > 15:
            sections.append("Research status: Moderately active")
        elif recent_pct > 5:
            sections.append("Research status: Limited recent attention")
        else:
            sections.append("Research status: Sparse recent literature")

    # Publication types
    pub_types = findings.get('publication_types', {})
    if pub_types:
        sections.append("\nPublication Types:")
        # Sort by count
        sorted_types = sorted(pub_types.items(), key=lambda x: x[1], reverse=True)
        for ptype, count in sorted_types[:5]:
            pct = (count / total) * 100 if total > 0 else 0
            sections.append(f"  - {ptype}: {count} ({pct:.1f}%)")

    # =========================================================================
    # Key Research Themes
    # =========================================================================
    themes = findings.get('research_themes', [])
    if themes:
        sections.append("\n### Key Research Themes\n")
        sections.append("Major research directions identified from literature:")
        for i, theme in enumerate(themes, 1):
            # Capitalize theme nicely
            theme_display = theme.title() if theme.islower() else theme
            sections.append(f"  {i}. {theme_display}")

    # Active research areas
    active_areas = findings.get('active_research_areas', [])
    if active_areas and active_areas != themes:
        sections.append("\nCurrently Active Areas (from recent papers):")
        for area in active_areas[:3]:
            area_display = area.title() if area.islower() else area
            sections.append(f"  - {area_display}")

    # =========================================================================
    # Notable Recent Studies
    # =========================================================================
    top_papers = findings.get('top_papers', [])
    if top_papers:
        sections.append("\n### Notable Recent Studies\n")
        sections.append("Top relevant papers for this research context:\n")

        for i, paper in enumerate(top_papers, 1):
            title = paper.get('title', 'Unknown title')
            year = paper.get('year', 'N/A')
            pmid = paper.get('pmid', 'N/A')
            journal = paper.get('journal', 'Unknown')
            relevance = paper.get('relevance_score', 0)
            key_finding = paper.get('key_finding', '')

            # Relevance indicator
            if relevance >= 4:
                relevance_indicator = "Highly Relevant"
            elif relevance >= 3:
                relevance_indicator = "Relevant"
            else:
                relevance_indicator = "Related"

            sections.append(f"{i}. **{title}**")
            sections.append(f"   {journal} ({year}) | PMID: {pmid}")
            sections.append(f"   Relevance: {relevance_indicator}")

            if key_finding:
                # Wrap long findings
                if len(key_finding) > 80:
                    sections.append(f"   Key Finding:")
                    # Simple word wrap
                    words = key_finding.split()
                    line = "   "
                    for word in words:
                        if len(line) + len(word) + 1 > 80:
                            sections.append(line)
                            line = "   " + word
                        else:
                            line += (" " + word) if len(line) > 3 else word
                    if line.strip():
                        sections.append(line)
                else:
                    sections.append(f"   Key Finding: {key_finding}")

            sections.append("")

    # =========================================================================
    # Research Timeline
    # =========================================================================
    timeline = findings.get('publication_timeline', {})
    if timeline:
        sections.append("### Research Timeline\n")
        sections.append("Publication activity over time:")
        sections.append("")

        # Get last 10 years of data
        years = sorted(timeline.keys(), reverse=True)[:10]
        years = sorted(years)  # Re-sort ascending for display

        if years:
            max_count = max(timeline.get(y, 0) for y in years)

            for year in years:
                count = timeline.get(year, 0)
                # Simple bar chart
                bar_length = int((count / max_count) * 20) if max_count > 0 else 0
                bar = "" * bar_length
                sections.append(f"  {year}: {bar} {count}")

            sections.append("")

            # Trend analysis
            if len(years) >= 3:
                recent_avg = sum(timeline.get(y, 0) for y in years[-2:]) / 2
                older_avg = sum(timeline.get(y, 0) for y in years[:-2]) / max(len(years) - 2, 1)

                if recent_avg > older_avg * 1.5:
                    sections.append("Trend: INCREASING research interest")
                elif recent_avg < older_avg * 0.5:
                    sections.append("Trend: Declining research activity")
                else:
                    sections.append("Trend: Stable research activity")

    sections.append("")
    return "\n".join(sections)


def format_literature_findings_summary(
    all_findings: Dict[str, Dict[str, Any]]
) -> str:
    """
    Format a summary of literature findings across all genes.

    Args:
        all_findings: Dictionary of gene -> findings

    Returns:
        Formatted summary string
    """
    if not all_findings:
        return ""

    sections = []
    sections.append("\n" + "=" * 80)
    sections.append("LITERATURE OVERVIEW (All Genes)")
    sections.append("=" * 80)

    # Aggregate statistics
    total_papers = sum(f.get('total_papers', 0) for f in all_findings.values())
    total_recent = sum(f.get('recent_papers_count', 0) for f in all_findings.values())
    genes_with_data = sum(1 for f in all_findings.values() if f.get('total_papers', 0) > 0)

    sections.append(f"\nGenes analysed: {len(all_findings)}")
    sections.append(f"Genes with literature: {genes_with_data}")
    sections.append(f"Total papers reviewed: {total_papers:,}")
    sections.append(f"Recent papers (last 2 years): {total_recent:,}")

    if total_papers > 0:
        sections.append(f"Recent publication rate: {(total_recent/total_papers)*100:.1f}%")

    # Per-gene summary table
    sections.append("\n### Per-Gene Literature Summary\n")
    sections.append(f"{'Gene':<12} {'Total':<10} {'Recent':<10} {'Top Theme':<30}")
    sections.append("-" * 65)

    for gene, findings in all_findings.items():
        total = findings.get('total_papers', 0)
        recent = findings.get('recent_papers_count', 0)
        themes = findings.get('research_themes', [])
        top_theme = themes[0] if themes else '-'

        # Truncate theme if too long
        if len(top_theme) > 28:
            top_theme = top_theme[:25] + "..."

        sections.append(f"{gene:<12} {total:<10} {recent:<10} {top_theme:<30}")

    sections.append("")
    return "\n".join(sections)
