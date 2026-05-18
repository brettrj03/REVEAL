"""Phase 3 node for cross-gene synthesis generation."""

from __future__ import annotations

from dataclasses import dataclass
import time
import re

from pydantic_ai import Agent
from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.graph.state import GeneState, _accumulate_tokens
from src.models.report_components import BiologicalSynthesis
from src.utils.tracing import trace_event
from src.config import get_active_model


_synthesis_agent = Agent(
    'openai:gpt-4.1-mini',
    system_prompt="""You are a molecular biologist creating an interpretive synthesis of multiple gene analyses.
Your role is to interpret patterns in the data and explain their biological significance.

CRITICAL REQUIREMENTS:
1. ANSWER THE USER'S QUERY: Explicitly interpret what the gene set means in the context of their biological question.
2. INTERPRET, DON'T JUST LIST: Go beyond stating overlaps — explain what shared functions mean biologically.
   - Bad: "Gene X and Y share GO term Z"
   - Good: "The convergence of X and Y on shared transcriptional machinery suggests a common chromatin remodelling pathway"
3. GROUP GENES MEANINGFULLY: Identify functional clusters and explain what unites them and what separates outliers.
4. FLAG DATA-ABSENT GENES: Genes with limited overlap should be noted as potentially acting through separate mechanisms.
5. GROUND EVERY CLAIM IN DATA: Only make interpretations supported by explicit GO terms, STRING interactions, or expression patterns in the evidence provided.

CONSTRAINTS:
- Only discuss connections where there is EXPLICIT evidence (shared STRING partners, shared GO terms, expression patterns)
- Do NOT speculate beyond what the data supports
- Do NOT fabricate thematic connections between unrelated genes
- If genes operate in completely different biological domains, state this and explain why based on the evidence
"""
)


@dataclass
class GenerateCrossGeneSynthesis(BaseNode[GeneState]):
    """Generate executive summary + key findings when in interpreted mode."""

    async def run(self, ctx: GraphRunContext[GeneState]) -> "ValidateCrossGeneSynthesis":
        _t0 = time.perf_counter()
        try:
            from src.nodes.validate_cross_gene_synthesis import ValidateCrossGeneSynthesis

            print(f"\n{'='*70}")
            print("NODE: Generate Cross-Gene Synthesis")
            print(f"{'='*70}")

            if ctx.state.output_mode == "factual" or not ctx.state.gene_profiles:
                print("Skipping cross-gene synthesis (factual mode)")
                ctx.state.synthesis = None
                ctx.state.interpretations_complete = True
                return ValidateCrossGeneSynthesis()

            # Use gene interpretations directly
            interpretations = ctx.state.gene_interpretations

            try:
                synthesis = await _generate_synthesis(
                    ctx.state.gene_profiles,
                    interpretations,
                    ctx.state.experiment_context,
                    network_overlap=ctx.state.network_overlap_analysis,
                    go_comparison=ctx.state.go_comparison_analysis,
                    state=ctx.state,
                    user_query=ctx.state.user_query,
                )
                ctx.state.synthesis = synthesis
                ctx.state.interpretations_complete = True
                print("✓ Cross-gene synthesis generated")
                trace_event(
                    "interpretation.cross_gene_synthesis",
                    genes=sorted(ctx.state.gene_profiles.keys()),
                    inputs=['gene_profiles', 'gene_interpretations', 'network_overlap_analysis', 'go_comparison_analysis'],
                    key_findings=len(synthesis.key_findings or [])
                )
            except Exception as exc:  # pragma: no cover - best effort
                print(f"⚠️  Cross-gene synthesis failed: {exc}")
                ctx.state.synthesis = BiologicalSynthesis(
                    executive_summary=f"Analysis of {len(ctx.state.gene_profiles)} genes completed.",
                    key_findings=[]
                )
                trace_event(
                    "interpretation.cross_gene_synthesis_failure",
                    genes=len(ctx.state.gene_profiles),
                    error=str(exc)
                )

            return ValidateCrossGeneSynthesis()
        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )


async def _generate_synthesis(
    gene_profiles: dict,
    gene_interpretations: dict,
    experiment_context,
    network_overlap: dict = None,
    go_comparison: dict = None,
    state: GeneState | None = None,
    user_query: str = None,
) -> BiologicalSynthesis:
    gene_summaries = []
    for gene_symbol, profile in sorted(gene_profiles.items()):
        summary = f"{gene_symbol}: "
        if profile.functional_summary:
            summary += profile.functional_summary[:200]
        else:
            names = []
            for func in profile.molecular_functions[:2]:
                if isinstance(func, dict):
                    names.append(func.get('name', 'Unknown'))
                else:
                    names.append(str(func))
            summary += ", ".join(names) if names else "No function data"
        if gene_interpretations.get(gene_symbol):
            expr = gene_interpretations[gene_symbol].get('expression_interpretation')
            if expr:
                summary += f" | Expression: {expr[:120]}"
        gene_summaries.append(summary)

    # ── Single-gene path ─────────────────────────────────────────────────
    if len(gene_profiles) == 1:
        gene_symbol = sorted(gene_profiles.keys())[0]
        prompt = f"""Gene analysed: {gene_symbol}

Summary:
{gene_summaries[0]}

Experimental context: {experiment_context.model_dump() if experiment_context else 'General analysis'}

Write a focused 2-3 sentence biological overview of {gene_symbol} in this experimental context. Cover its core function, relevance to the context, and notable biological properties (expression, disease associations, or pathway membership) that are grounded in the provided summary.

RULES:
- Plain prose only — no bullet points, headers, or numbered lists
- Do NOT mention cross-gene comparisons or other genes
- Avoid speculative phrasing unless the underlying evidence is explicitly uncertain
- Ground every claim in the summary data provided above
"""
        result = await _synthesis_agent.run(prompt, model=f'openai:{get_active_model()}')
        usage = result.usage() if callable(getattr(result, "usage", None)) else None
        _accumulate_tokens(state, "GenerateCrossGeneSynthesis", usage)
        executive_summary = result.output.strip()
        executive_summary = re.sub(
            r'^[\d\.\s]*\*{0,2}executive summary:?\*{0,2}\s*',
            '',
            executive_summary,
            flags=re.IGNORECASE
        ).strip()

        return BiologicalSynthesis(
            executive_summary=executive_summary,
            cross_gene_insights=None,
            key_findings=[]
        )

    # Format cross-gene evidence
    cross_gene_evidence = []

    if network_overlap:
        hub_proteins = network_overlap.get('hub_proteins', [])
        direct_interactions = network_overlap.get('direct_interactions', [])
        if hub_proteins:
            cross_gene_evidence.append(f"Shared interaction partners: {len(hub_proteins)} proteins interact with multiple query genes")
            for hub in hub_proteins[:5]:
                partner = hub.get('protein') or hub.get('partner', 'Unknown')
                genes = ', '.join(hub.get('genes', []))
                cross_gene_evidence.append(f"  - {partner} interacts with: {genes}")
        if direct_interactions:
            cross_gene_evidence.append(f"Direct gene-gene interactions: {len(direct_interactions)} pairs")

    if go_comparison:
        shared_terms = go_comparison.get('shared_terms', [])
        if shared_terms:
            cross_gene_evidence.append(f"Shared GO terms: {len(shared_terms)} terms present in 2+ genes")
            for term in shared_terms[:5]:
                genes = ', '.join(term.get('genes', []))
                cross_gene_evidence.append(f"  - {term.get('name', 'Unknown')}: {genes}")

    evidence_section = "\n".join(cross_gene_evidence) if cross_gene_evidence else "No shared interactions or GO terms identified between the query genes."

    # Build user query context
    query_context = f"User's biological question: {user_query}" if user_query else "General gene analysis"

    prompt = f"""Genes analysed: {', '.join(sorted(gene_profiles.keys()))}

{query_context}

Summaries:
{chr(10).join(gene_summaries)}

Cross-Gene Evidence (from network and GO analysis):
{evidence_section}

Experimental context: {experiment_context.model_dump() if experiment_context else 'General analysis'}

TASK: Create an interpretive synthesis that answers the user's biological question.

CRITICAL: Every gene in the "Genes analysed" list MUST be mentioned by name in your output — no gene can be omitted. If a gene has no shared GO terms or interaction partners with other genes, it must still be mentioned and explicitly described as operating through a distinct or currently uncharacterised mechanism relative to the other genes.

REQUIREMENTS:
1. ANSWER THE USER'S QUERY: Explicitly interpret what this gene set means in the context of their biological question above.

2. INTERPRET PATTERNS, DON'T JUST LIST:
   - Bad: "Gene X and Y share GO term Z"
   - Good: "The convergence of X and Y on transcriptional regulation (GO:0006355) suggests coordinated control of gene expression programs"
   - Explain what shared functions, interactions, or expression patterns mean biologically

3. GROUP GENES MEANINGFULLY:
   - Identify functional clusters based on the evidence above (shared GO terms, shared interaction partners, expression patterns)
   - Explain what unites each cluster biologically
   - Flag outlier genes that don't fit the dominant pattern and explain why (based on lack of shared terms/partners)

4. GROUND EVERY CLAIM IN THE EVIDENCE ABOVE:
   - Only cite GO terms, interaction partners, or expression patterns explicitly listed in the evidence section
   - When interpreting a shared function, reference the specific GO term or interaction partner from the evidence
   - Do NOT speculate beyond what the data supports

5. BE HONEST ABOUT DATA GAPS:
   - If genes have no demonstrated connections, state: "These genes operate through distinct mechanisms with no detected functional overlap"
   - If a gene has limited data, note it may act through a separate pathway

Provide:
1. Executive summary (4-6 sentences):
   - Answer the user's query in the first sentence
   - Identify major functional themes or clusters
   - Explain biological significance based on the evidence

2. Key findings (4-6 bullet points):
   - Each must be interpretive, not just a list item
   - Each must reference specific evidence (GO term name, interaction partner, expression pattern)
   - Group related findings together

FORMATTING RULES:
- Each bullet must be a SINGLE line: bold label + colon + explanation all together
- Do NOT put the label on one line and the explanation on the next line as separate bullets
- Format: **Label:** Explanation text here on the same line
- Never split a finding across two bullet points

Examples of correct formatting:
- **Shared transcriptional regulation:** MED12, EOMES, PEG3, and ZIM2 converge on chromatin remodelling pathways (GO:0006355) suggesting coordinated control of gene expression programs
- **Neuronal connectivity:** PCDHA6 and PCDHGA3 share homophilic cell adhesion functions (GO:0007156) indicating a role in neural circuit formation
- **Distinct mechanism:** MIMT1 operates through a lncRNA-mediated regulatory pathway with no detected functional overlap with other query genes
"""

    result = await _synthesis_agent.run(prompt, model=f'openai:{get_active_model()}')
    usage = result.usage() if callable(getattr(result, "usage", None)) else None
    _accumulate_tokens(state, "GenerateCrossGeneSynthesis", usage)
    response_text = result.output

    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    executive_lines = []
    key_findings = []
    in_findings = False

    for line in lines:
        lowered = line.lower()
        if 'key findings' in lowered or lowered.startswith('findings'):
            in_findings = True
            continue
        if in_findings:
            cleaned = line.lstrip('-•0123456789. ').strip()
            if cleaned:
                key_findings.append(cleaned)
        else:
            executive_lines.append(line)

    executive_summary = ' '.join(executive_lines) if executive_lines else response_text
    executive_summary = re.sub(
        r'^[\d\.\s]*\*{0,2}executive summary:?\*{0,2}\s*',
        '',
        executive_summary,
        flags=re.IGNORECASE
    ).strip()

    if not key_findings:
        key_findings = executive_lines[:5]

    return BiologicalSynthesis(
        executive_summary=executive_summary,
        cross_gene_insights=None,
        key_findings=key_findings[:5]
    )
