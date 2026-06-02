from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field as PydanticField


def clean_gene_name(description: Optional[str]) -> str:
    """Extract a readable gene name from database descriptions."""
    if not description:
        return ""
    cleaned = description
    if '[Source:' in cleaned:
        cleaned = cleaned.split('[Source:')[0].strip()
    return cleaned.rstrip('.')

@dataclass
class GeneProfile:
    """Individual gene profile with factual and interpreted components"""

    # Basic gene info
    gene_symbol: str
    gene_id: str
    location: str
    gene_type: str
    full_name: str

    # Query mapping info
    search_term: str
    mapping_type: str  # "direct", "alias", "corrected"
    mapping_note: Optional[str] = None

    # Optional database description
    long_description: Optional[str] = None

    # FACTUAL data (from database - always populated)
    # GO terms can be either strings (legacy) or dicts with 'name' and 'go_id' keys
    molecular_functions: List[Union[str, Dict]] = field(default_factory=list)
    biological_processes: List[Union[str, Dict]] = field(default_factory=list)
    cellular_components: List[Union[str, Dict]] = field(default_factory=list)
    expression_data: List[Dict] = field(default_factory=list)  # [{"tissue": "X", "tpm": 123.45}]
    interactions: List[Dict] = field(default_factory=list)  # [{"partner": "X", "score": 0.99, "description": "Y"}]

    # INTERPRETED data (from LLM - only in interpreted mode)
    functional_summary: Optional[str] = None
    expression_interpretation: Optional[str] = None
    interaction_interpretation: Optional[str] = None
    experimental_relevance: Optional[str] = None

    def get_mapping_description(self) -> str:
        """Human-readable mapping status"""
        if self.mapping_type == "direct":
            return "direct match (official gene symbol)"
        elif self.mapping_type == "alias":
            return f"mapped from alias ({self.mapping_note})"
        elif self.mapping_type == "corrected":
            return f"corrected ({self.mapping_note})"
        return "unknown mapping"

    @staticmethod
    def clean_tissue_name(tissue_name: str) -> str:
        """Convert tissue names from database format to readable format"""
        # Replace underscores with spaces
        name = tissue_name.replace('_', ' ')

        # Special cases for better formatting
        if name.startswith('Cells '):
            name = name.replace('Cells ', 'Cells - ')
        if name.startswith('Brain ') and ' ' in name[6:]:
            # Keep Brain prefix but clean rest
            parts = name.split(' ', 1)
            if len(parts) > 1:
                name = f"Brain - {parts[1]}"

        return name

    def format_expression_table(self, top_n: int = 5) -> str:
        """Format expression as simple table"""
        if not self.expression_data:
            return "  No expression data available."

        sorted_expr = sorted(self.expression_data, key=lambda x: x.get('tpm', 0), reverse=True)
        top_tissues = sorted_expr[:top_n]

        lines = ["  Tissue                          TPM"]
        lines.append("  " + "-" * 45)

        for tissue_data in top_tissues:
            # Clean tissue name
            tissue_raw = tissue_data['tissue']
            tissue_clean = self.clean_tissue_name(tissue_raw)
            tissue = tissue_clean[:30].ljust(30)

            tpm = tissue_data.get('tpm', 0)
            tpm_str = f"{tpm:.2f}".rjust(10)

            lines.append(f"  {tissue} {tpm_str}")

        return "\n".join(lines)

    @staticmethod
    def _truncate_at_word(text: str, max_length: int = 50) -> str:
        """Truncate text at word boundary with ellipsis"""
        if len(text) <= max_length:
            return text
        # Find last space before max_length
        truncated = text[:max_length].rsplit(' ', 1)[0]
        return truncated + '...'

    def format_interaction_table(self, top_n: int = 5) -> str:
        """Format interactions as simple table without boxes"""
        if not self.interactions:
            return "  No interaction data available."

        sorted_int = sorted(self.interactions, key=lambda x: x.get('score', 0), reverse=True)
        top_interactions = sorted_int[:top_n]

        lines = ["  Partner      Score   Functional Context"]
        lines.append("  " + "-" * 70)

        for interaction in top_interactions:
            partner = interaction['partner'].ljust(12)
            score = f"{interaction['score']:.3f}"
            desc = self._truncate_at_word(interaction.get('description', ''), max_length=50)
            lines.append(f"  {partner} {score}   {desc}")

        return "\n".join(lines)

    @staticmethod
    def _wrap_text(text: str, width: int = 78, indent: int = 2) -> str:
        """Wrap text with indentation, handling numbered lists and paragraphs"""
        import re

        # First, try to split on numbered list items (e.g., "  1. ", "  2. ", etc.)
        # This handles cases where LLM returns everything on one line
        text = re.sub(r'\s+(\d+\.\s+\*\*)', r'\n\1', text)  # Split before numbered items

        # Split by newlines to preserve paragraph breaks
        paragraphs = text.split('\n')
        wrapped_paragraphs = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Just indent and preserve as-is - don't try to wrap since it breaks formatting
            # The LLM has already formatted it, we just add indentation
            lines = para.split('\n')
            for line in lines:
                if line.strip():
                    wrapped_paragraphs.append(' ' * indent + line.strip())

        return '\n'.join(wrapped_paragraphs)

    def to_detailed_text(self) -> str:
        """LONG VERSION: Full academic-style profile"""
        lines = [
            "=" * 80,
            f"{self.gene_symbol} - {self.full_name.upper()}",
            "=" * 80,
            f"Gene ID:        {self.gene_id}",
            f"Location:       {self.location}",
            f"Type:           {self.gene_type}",
            f"Query Term:     {self.search_term} ({self.get_mapping_description()})",
            ""
        ]

        # Molecular Function
        lines.append("Molecular Function:")
        lines.append("-" * 80)
        if self.molecular_functions:
            for func in self.molecular_functions[:5]:
                # Handle both string and dict formats
                if isinstance(func, dict):
                    name = func.get('name', '')
                    go_id = func.get('go_id', '')
                    lines.append(f"  • {name} ({go_id})")
                else:
                    lines.append(f"  • {func}")
        else:
            lines.append("  No molecular function annotations available.")
        lines.append("")

        if self.functional_summary:
            lines.append("Interpretation:")
            lines.append(self._wrap_text(self.functional_summary, width=78, indent=2))
            lines.append("")

        # Biological Process
        if self.biological_processes:
            lines.append("Biological Process:")
            lines.append("-" * 80)
            for proc in self.biological_processes[:5]:
                # Handle both string and dict formats
                if isinstance(proc, dict):
                    name = proc.get('name', '')
                    go_id = proc.get('go_id', '')
                    lines.append(f"  • {name} ({go_id})")
                else:
                    lines.append(f"  • {proc}")
            lines.append("")

        # Cellular Component
        if self.cellular_components:
            lines.append("Cellular Component:")
            lines.append("-" * 80)
            for comp in self.cellular_components[:3]:
                # Handle both string and dict formats
                if isinstance(comp, dict):
                    name = comp.get('name', '')
                    go_id = comp.get('go_id', '')
                    lines.append(f"  • {name} ({go_id})")
                else:
                    lines.append(f"  • {comp}")
            lines.append("")

        # Expression
        lines.append("Tissue Expression:")
        lines.append("-" * 80)
        lines.append(self.format_expression_table(top_n=5))
        lines.append("")

        if self.expression_interpretation:
            lines.append("Expression Analysis:")
            lines.append(self._wrap_text(self.expression_interpretation, width=78, indent=2))
            lines.append("")

        # Interactions
        lines.append("Protein Interactions (STRING v12.0):")
        lines.append("-" * 80)
        lines.append(self.format_interaction_table(top_n=5))
        lines.append("")

        if self.interaction_interpretation:
            lines.append("Network Context:")
            lines.append(self._wrap_text(self.interaction_interpretation, width=78, indent=2))
            lines.append("")

        # Experimental Relevance
        if self.experimental_relevance:
            lines.append("Experimental Relevance:")
            lines.append("-" * 80)
            lines.append(self._wrap_text(self.experimental_relevance, width=78, indent=2))
            lines.append("")

        return "\n".join(lines)

    def to_concise_text(self) -> str:
        """SHORT VERSION: One-line summary"""
        summary = self.functional_summary if self.functional_summary else "; ".join(self.molecular_functions[:2])
        return f"{self.gene_symbol} ({self.location}) - {summary}"

    def to_factual_only_text(self) -> str:
        """FACTUAL VERSION: Database data only, no interpretations"""
        lines = [
            "=" * 80,
            f"{self.gene_symbol} - {self.full_name.upper()}",
            "=" * 80,
            f"Gene ID:    {self.gene_id}",
            f"Location:   {self.location}",
            f"Type:       {self.gene_type}",
            ""
        ]

        if self.molecular_functions:
            lines.append("Molecular Functions:")
            for func in self.molecular_functions[:5]:
                lines.append(f"  • {func}")
            lines.append("")

        if self.biological_processes:
            lines.append("Biological Processes:")
            for proc in self.biological_processes[:5]:
                lines.append(f"  • {proc}")
            lines.append("")

        lines.append("Expression Data:")
        lines.append(self.format_expression_table(top_n=5))
        lines.append("")

        lines.append("Protein Interactions:")
        lines.append(self.format_interaction_table(top_n=5))
        lines.append("")

        return "\n".join(lines)

    def to_unified_format(self) -> str:
        """Render this profile in facts-first unified format."""

        sections: List[str] = []
        sections.append("\n" + "=" * 80)
        if self.full_name:
            sections.append(f"{self.gene_symbol} - {self.full_name}")
        else:
            sections.append(f"{self.gene_symbol}")
        sections.append("=" * 80 + "\n")

        sections.append("IDENTIFICATION:")
        sections.append(f"Official Symbol: {self.gene_symbol}")
        sections.append(f"Gene ID: {self.gene_id}")
        sections.append(f"Location: {self.location}")
        sections.append(f"Type: {self.gene_type}")
        sections.append(f"Query Term: {self.search_term} ({self.get_mapping_description()})")

        if self.long_description:
            sections.append("\n\nDATABASE FUNCTION DESCRIPTION:")
            sections.append(self.long_description)

        # === GO TERMS ===
        sections.append("\n\nGENE ONTOLOGY TERMS:\n")

        def _emit_terms(title: str, items: List[Union[str, Dict]]) -> None:
            if not items:
                return
            sections.append(title)
            for entry in items[:10]:
                if isinstance(entry, dict):
                    name = entry.get('name', 'Unknown')
                    go_id = entry.get('go_id', 'GO:?')
                    sections.append(f"• {name} ({go_id})")
                else:
                    sections.append(f"• {entry}")
            sections.append("")

        if any([self.molecular_functions, self.biological_processes, self.cellular_components]):
            _emit_terms("Molecular Function (top 10):", self.molecular_functions)
            _emit_terms("Biological Process (top 10):", self.biological_processes)
            _emit_terms("Cellular Component (top 10):", self.cellular_components)
        else:
            sections.append("No GO terms available in database")

        if self.functional_summary:
            sections.append("\n\nFUNCTIONAL INTERPRETATION:")
            sections.append(self.functional_summary)

        # === EXPRESSION ===
        raw_expression = self.expression_data or []
        sorted_expression = sorted(
            raw_expression,
            key=lambda entry: float(entry.get('tpm') or entry.get('level') or 0.0),
            reverse=True
        )
        total_tissues = len(sorted_expression)
        display_count = min(5, total_tissues)
        sections.append(
            f"\n\nEXPRESSION DATA (top {display_count} of {total_tissues} tissues):\n"
        )
        if sorted_expression:
            sections.append(f"{'Tissue':<30} {'TPM':>10} {'Percentile (%)':>15} {'Classification':>30}")
            sections.append("-" * 95)

            def _percentile_for_index(index: int) -> float:
                if total_tissues <= 1:
                    return 100.0
                return round((1 - index / (total_tissues - 1)) * 100.0, 1)

            for idx, data in enumerate(sorted_expression[:display_count]):
                tissue = (
                    data.get('tissue_name')
                    or data.get('tissue_detailed')
                    or data.get('tissue')
                    or 'Unknown'
                )
                tpm_val = float(data.get('tpm') or data.get('level') or 0.0)
                percentile = _percentile_for_index(idx)
                classification = self._classify_expression(percentile)
                sections.append(
                    f"{tissue:<30} {tpm_val:>10.1f} {percentile:>15.1f} {classification:>30}"
                )

        else:
            sections.append("No expression data available")

        if self.expression_interpretation:
            sections.append("\n\nEXPRESSION INTERPRETATION:")
            sections.append(self.expression_interpretation)

        sections.extend(self._format_compact_interactions())

        if self.interaction_interpretation:
            sections.append("\n\nNETWORK INTERPRETATION:")
            sections.append(self.interaction_interpretation)

        if self.experimental_relevance:
            sections.append("\n\nEXPERIMENTAL RELEVANCE:")
            sections.append(self.experimental_relevance)

        return "\n".join(sections)

    @staticmethod
    def _classify_expression(percentile: float) -> str:
        """Classify expression percentile into descriptor."""
        if percentile is None:
            return "Unknown"
        try:
            value = float(percentile)
        except (TypeError, ValueError):
            return "Unknown"

        if value >= 90:
            return "Very High (≥90th percentile)"
        if value >= 75:
            return "High (75th-89th percentile)"
        if value >= 50:
            return "Moderate (50th-74th percentile)"
        if value >= 25:
            return "Low (25th-49th percentile)"
        return "Very Low (<25th percentile)"

    def _format_compact_interactions(self) -> List[str]:
        """Format interactions as a compact block with top partners and stats."""

        sections: List[str] = ["\n\nPROTEIN INTERACTIONS:\n"]
        interactions = self.interactions or []
        if not interactions:
            sections.append("No interaction data available")
            return sections

        def _score(entry: Dict) -> float:
            score = entry.get('score')
            if score is None:
                raw = entry.get('combined_score')
                if raw:
                    score = float(raw) / 1000.0 if raw > 1 else float(raw)
            return float(score) if score is not None else 0.0

        def _partner(entry: Dict) -> str:
            return (
                entry.get('partner_symbol')
                or entry.get('partner')
                or entry.get('protein_name')
                or 'Unknown'
            )

        def _brief_description(entry: Dict) -> str:
            desc = (entry.get('description') or '').strip()
            if not desc:
                return ''
            snippet = desc.split(';')[0].split('.')[0].strip()
            if len(snippet) > 100:
                snippet = snippet[:97] + "..."
            return snippet

        high_conf = [i for i in interactions if _score(i) > 0.7]
        med_conf = [i for i in interactions if 0.4 <= _score(i) <= 0.7]

        if high_conf:
            sections.append(f"Top 5 High-Confidence Partners ({len(high_conf)} total):")
            for interaction in high_conf[:5]:
                partner = _partner(interaction)
                score = _score(interaction)
                brief = _brief_description(interaction)
                if brief:
                    sections.append(f"- {partner} ({score:.3f}) - {brief}")
                else:
                    sections.append(f"- {partner} ({score:.3f})")

        total = len(interactions)
        sections.append(
            f"\nNetwork Statistics: {total} interactions ({len(high_conf)} high, {len(med_conf)} medium confidence)"
        )

        return sections


class ReportMetadata(BaseModel):
    """Report header metadata — Pydantic model so pydantic_graph serializes it cleanly."""
    query: str
    date: datetime
    organism: str
    comparison: Optional[str] = None
    cell_type: Optional[str] = None
    treatment: Optional[str] = None
    hypothesis: Optional[str] = None

    # Gene mapping stats
    input_genes: int = 0
    unique_genes: int = 0
    successfully_mapped: int = 0
    direct_matches: List[str] = PydanticField(default_factory=list)
    mapped_via_correction: List[Dict] = PydanticField(default_factory=list)
    not_found: List[Dict] = PydanticField(default_factory=list)

    def to_header_text(self) -> str:
        """Generate metadata header"""
        lines = [
            "=" * 80,
            "GENE ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Query: {self.query}",
            f"Date: {self.date.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Experiment Context:",
            f"  Organism: {self.organism}",
        ]

        if self.comparison:
            lines.append(f"  Comparison: {self.comparison}")
        if self.cell_type:
            lines.append(f"  Cell type: {self.cell_type}")

        lines.extend(["", "=" * 80, "GENES ANALYZED", "=" * 80, ""])
        lines.append(f"Input genes: {self.input_genes}")
        lines.append(f"Unique genes: {self.unique_genes}")
        lines.append(f"Successfully mapped: {self.successfully_mapped}/{self.unique_genes}")

        if self.direct_matches:
            lines.append(f"\nDirect matches ({len(self.direct_matches)}):")
            lines.append(", ".join(self.direct_matches))

        if self.mapped_via_correction:
            lines.append(f"\nMapped via correction ({len(self.mapped_via_correction)}):")
            for m in self.mapped_via_correction:
                lines.append(f"  • {m['from']} → {m['to']} ({m['note']})")

        if self.not_found:
            lines.append(f"\nNot found ({len(self.not_found)}):")
            for nf in self.not_found:
                suggestion = f" (try: {nf['suggestion']})" if nf.get('suggestion') else ""
                lines.append(f"  • {nf['gene']}{suggestion}")

        return "\n".join(lines)


class BiologicalSynthesis(BaseModel):
    """Cross-gene synthesis"""
    executive_summary: str
    cross_gene_insights: Optional[str] = None
    key_findings: List[str] = PydanticField(default_factory=list)

    @staticmethod
    def _clean_text(text: Optional[str]) -> str:
        if not text:
            return ""
        return text.replace('**', '').strip()

    def to_comprehensive_text(self) -> str:
        """Full synthesis for comprehensive report"""
        lines = [
            "=" * 80,
            "BIOLOGICAL SYNTHESIS",
            "=" * 80,
            "",
            self._clean_text(self.executive_summary),
            ""
        ]

        if self.cross_gene_insights:
            lines.extend([
                "Cross-Gene Integration:",
                "-" * 80,
                self._clean_text(self.cross_gene_insights),
                ""
            ])

        if self.key_findings:
            lines.append("Key Findings:")
            lines.append("-" * 80)
            for finding in self.key_findings:
                lines.append(f"  • {self._clean_text(finding)}")
            lines.append("")

        return "\n".join(lines)

    def to_concise_text(self) -> str:
        """Brief summary only"""
        return f"Summary:\n{self._clean_text(self.executive_summary)}"
