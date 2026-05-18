"""Node responsible for assembling the unified final report."""

from __future__ import annotations

from dataclasses import dataclass
import time
from pydantic_graph.nodes import BaseNode, GraphRunContext, End

from src.graph.state import GeneState
from src.reports.unified_report_assembler import assemble_unified_report
from src.validation_config.validation_config import ACCURACY_THRESHOLD


@dataclass
class FinalSummary(BaseNode[GeneState]):
    """Create the final report using the unified assembler.

    Note: Removed extra type parameters (None, str) to match all other nodes
    and fix graph routing issue that caused hangs when transitioning from
    ValidateGeneSummaries to FinalSummary.
    """

    async def run(self, ctx: GraphRunContext[GeneState]) -> End[str]:
        _t0 = time.perf_counter()
        print(f"\n{'='*70}")
        print("NODE: Final Summary")
        print(f"{'='*70}")
        print(f"Synthesizing {len(ctx.state.gene_profiles)} gene profiles")

        try:
            profiles = [
                ctx.state.gene_profiles[symbol]
                for symbol in sorted(ctx.state.gene_profiles.keys())
            ]

            mode = getattr(ctx.state, "output_mode", "interpreted")

            final_report = assemble_unified_report(
                gene_profiles=profiles,
                report_metadata=ctx.state.report_metadata,
                synthesis=ctx.state.synthesis,
                go_comparison_analysis=ctx.state.go_comparison_analysis,
                network_overlap_analysis=ctx.state.network_overlap_analysis,
                mode=mode,
                pubmed_counts=ctx.state.pubmed_counts,
                pubmed_pmids=ctx.state.pubmed_pmids,
                pubmed_records=ctx.state.pubmed_records,
                disease_filter=ctx.state.literature_disease_filter_applied,
                disease_terms=ctx.state.literature_disease_terms_detected,
                gene_top_papers=ctx.state.gene_top_papers,
                literature_findings_summary=ctx.state.literature_findings_summary,
            )

            validation_block = self._build_validation_block(ctx.state)
            if validation_block:
                final_report = f"{final_report}\n\n{validation_block}"

            ctx.state.final_interpretation = final_report

            print(f"✓ Report assembled ({len(final_report):,} characters)")
            print(f"  Mode: {mode}")
            print(f"  Genes: {len(profiles)}")

            return End(final_report)

        except Exception as exc:  # pragma: no cover - defensive
            ctx.state.error = f"Final summary failed: {exc}"
            print(f"✗ ERROR: {ctx.state.error}")
            import traceback

            traceback.print_exc()
            return End(f"Error: {ctx.state.error}")
        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )

    def _build_validation_block(self, state: GeneState) -> str | None:
        """Render a validation metadata block for the final report."""
        if not state.accuracy_scores:
            return None

        lines = [
            "---",
            "VALIDATION METADATA",
            "---"
        ]

        for gene in sorted(state.accuracy_scores.keys()):
            accuracy = state.accuracy_scores.get(gene, 0.0)
            log_entry = state.validation_logs.get(gene, {})
            passes_used = log_entry.get('passes')

            if accuracy >= ACCURACY_THRESHOLD:
                line = f"- ✅ {gene}: Accuracy 100%"
            else:
                passes_text = f"{passes_used}" if passes_used not in (None, 0) else "multiple"
                line = (
                    f"- ⚠️ {gene}: Accuracy {accuracy:.1f}% — "
                    f"issues remain after {passes_text} refinement passes"
                )
                if state.low_confidence_flags.get(gene):
                    line += " (manual review recommended)"
            lines.append(line)

            summary = state.validation_summary.get(gene)
            if summary:
                lines.append(f"  {summary}")

        return "\n".join(lines)
