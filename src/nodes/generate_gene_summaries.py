"""
Generate comprehensive one-paragraph gene summaries synthesizing all data sources.

This node runs AFTER all interpretations and literature analysis are complete,
creating a high-level executive summary for each gene that synthesizes:
- Basic function and GO terms
- Expression patterns
- Protein interactions
- Recent literature findings
- Experimental relevance

This is distinct from the individual functional/expression/network interpretations.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.graph.state import GeneState, _accumulate_tokens
from src.utils.tracing import trace_event
from src.utils.phoenix_tracing import node_span
from src.config import get_active_model



# Agent for generating comprehensive gene summaries
# Using gpt-4.1-mini - the prompt contains only pre-processed LLM outputs, no raw abstracts
_summary_agent = Agent(
    'openai:gpt-4.1-mini',
    model_settings={'max_tokens': 800},
    system_prompt="""You write comprehensive gene summaries in the style of a review article introduction — flowing, integrated, and scientifically precise.

STRUCTURE (strictly follow this order):
1. FUNCTION FIRST — Open with what the gene/protein does. Draw from GO terms, database description, and weave in interaction partners as context for function (not as a standalone statistic)
2. DISEASE/CLINICAL — Synthesise the clinical picture thematically: what conditions are associated, inheritance pattern, phenotypic spectrum. Do NOT include specific variant nomenclature (e.g. p.Arg1138Trp) or study methodology details (e.g. "an iPSC line was established...")
3. LITERATURE BREADTH — End with a sentence summarising the breadth of literature themes. Do NOT invent or state specific paper counts, study numbers, or year counts. Use the format: "The literature on [GENE] spans themes including [theme1], [theme2], and [theme3], with recent studies investigating [theme]."

WRITING RULES:
- Single cohesive paragraph, 4-6 sentences, review article tone
- Use the gene symbol naturally throughout
- Every claim must be grounded in the provided data — no hallucination
- Plain text only (no markdown, bullets, or headers)
"""
)


@dataclass
class GenerateGeneSummaries(BaseNode[GeneState]):
    """
    Generate comprehensive one-paragraph summaries for each gene.

    Synthesizes all available data into an executive summary:
    - Basic function and GO terms
    - Expression patterns
    - Protein interactions
    - Recent literature findings
    - Experimental relevance to user's query

    Runs AFTER: ValidateLiteratureFindings (so summaries use validated key findings)
    Runs BEFORE: ValidateGeneSummaries
    """

    async def run(self, ctx: GraphRunContext[GeneState]) -> BaseNode[GeneState]:
        from src.nodes.validate_gene_summaries import ValidateGeneSummaries
        
        state = ctx.state
        _t0 = time.perf_counter()

        try:
            print(f"\n{'='*70}")
            print("NODE: Generate Comprehensive Gene Summaries")
            print(f"{'='*70}")

            if state.output_mode == "factual":
                print("Skipping gene summaries (factual mode)")
                return ValidateGeneSummaries()

            genes = state.get_genes_found()

            if not genes:
                print("No genes found to summarise")
                return ValidateGeneSummaries()

            print(f"Generating comprehensive summaries for {len(genes)} genes in parallel...")
            print()

            results = await asyncio.gather(*[
                _generate_single_summary(
                    gene=gene,
                    state=state,
                    all_genes=genes
                )
                for gene in genes
            ])

            for gene, summary in results:
                state.gene_summaries[gene] = summary
                print(f"  {gene}: Summary stored ({len(summary)} chars)")

            print(f"\nGenerated {len(state.gene_summaries)} comprehensive gene summaries")
            print(f"{'='*70}\n")

            return ValidateGeneSummaries()
        finally:
            state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )


async def _generate_single_summary(
    gene: str,
    state: GeneState,
    all_genes: list
) -> tuple[str, str]:
    """
    Generate summary for a single gene.

    Returns (gene_symbol, summary_string).
    """
    gene_data: dict = {}
    interpretations: dict = {}

    with node_span("GenerateGeneSummaries", gene=gene) as span:
        try:
            # Gather all data for this gene
            gene_data = state.all_gene_data.get(gene, {})
            interpretations = state.gene_interpretations.get(gene, {})
            if isinstance(interpretations, str):
                interpretations = {}
            papers = state.gene_top_papers.get(gene, {}).get('top_papers', [])

            # Get additional context
            literature_findings = state.literature_findings_summary.get(gene, {})
            cross_gene_synthesis = getattr(state, 'synthesis', None)
            if isinstance(cross_gene_synthesis, dict):
                cross_gene_synthesis = cross_gene_synthesis.get('executive_summary', '')
            elif not isinstance(cross_gene_synthesis, str):
                cross_gene_synthesis = ''

            total_papers_available = (
                literature_findings.get('total_papers')
                if literature_findings else None
            )
            if not total_papers_available:
                total_papers_available = (
                    state.gene_top_papers.get(gene, {}).get('total_candidates')
                    or len(papers)
                )

            # Build prompt with all available data
            prompt = _build_comprehensive_prompt(
                gene=gene,
                gene_data=gene_data,
                interpretations=interpretations,
                papers=papers,
                query=state.user_query,
                context=state.experiment_context,
                literature_findings=literature_findings,
                cross_gene_synthesis=cross_gene_synthesis,
                all_genes=all_genes,
                literature_total=total_papers_available
            )

            # Generate summary
            result = await _summary_agent.run(prompt, model=f'openai:{get_active_model()}')
            usage = result.usage() if callable(getattr(result, "usage", None)) else None
            _accumulate_tokens(state, "GenerateGeneSummaries", usage)
            summary_text = result.output.strip()

            trace_event(
                "interpretation.comprehensive_summary",
                gene=gene,
                summary_length=len(summary_text),
                state_inputs=['all_gene_data', 'gene_interpretations', 'gene_top_papers', 'literature_findings_summary', 'synthesis']
            )
            span.set_attribute("summary_length_chars", len(summary_text))

            return (gene, summary_text)

        except Exception as e:
            print(f"  {gene}: Failed to generate summary - {e}")
            # Fallback: create a basic summary from available data
            fallback = _create_fallback_summary(gene, gene_data, interpretations)
            trace_event(
                "interpretation.comprehensive_summary_fallback",
                gene=gene,
                error=str(e),
                used_interpretations=bool(interpretations)
            )
            span.set_attribute("fallback", True)
            span.set_attribute("error", str(e))
            span.set_attribute("summary_length_chars", len(fallback))
            return (gene, fallback)


def _build_comprehensive_prompt(
    gene: str,
    gene_data: dict,
    interpretations: dict,
    papers: list,
    query: str,
    context,
    literature_findings: dict = None,
    cross_gene_synthesis: str = None,
    all_genes: list = None,
    literature_total: int | None = None
) -> str:
    """Build comprehensive prompt with all available data."""

    # Basic function (increased limit)
    function_desc = gene_data.get('long_description', '')
    if not function_desc:
        function_desc = gene_data.get('description', 'Unknown function')
    if len(function_desc) > 800:
        function_desc = function_desc[:800] + "..."

    # Highest-expressed tissue only (or two if nearly equal)
    expression = gene_data.get('expression', [])
    top_tissues = sorted(expression, key=lambda x: x.get('level', 0) or 0, reverse=True)
    if top_tissues:
        highest = top_tissues[0]
        highest_name = highest.get('tissue', 'Unknown').replace('_', ' ')
        highest_level = highest.get('level', 0) or 0
        # Check if second tissue is within 20% of highest
        if len(top_tissues) > 1:
            second = top_tissues[1]
            second_level = second.get('level', 0) or 0
            if highest_level > 0 and second_level >= highest_level * 0.8:
                second_name = second.get('tissue', 'Unknown').replace('_', ' ')
                tissue_str = f"{highest_name} and {second_name}"
            else:
                tissue_str = highest_name
        else:
            tissue_str = highest_name
    else:
        tissue_str = "Expression data not available"

    # Top GO terms (biological process) - deduplicated
    go_terms = gene_data.get('go_terms', [])
    bp_terms = [t for t in go_terms if t.get('namespace') == 'biological_process']
    seen_names = set()
    unique_bp = []
    for term in bp_terms:
        name = term.get('name', '')
        if name and name not in seen_names:
            seen_names.add(name)
            unique_bp.append(name)
        if len(unique_bp) >= 5:
            break
    go_str = ", ".join(unique_bp) if unique_bp else "No biological process annotations"

    # Top GO terms (cellular component) - for localisation context
    cc_terms = [t for t in go_terms if t.get('namespace') == 'cellular_component']
    seen_cc_names = set()
    unique_cc = []
    for term in cc_terms:
        name = term.get('name', '')
        if name and name not in seen_cc_names:
            seen_cc_names.add(name)
            unique_cc.append(name)
        if len(unique_cc) >= 3:
            break
    cellular_component_str = ", ".join(unique_cc) if unique_cc else "No cellular component annotations"

    # Network info
    interactions = gene_data.get('interactions', [])
    num_interactions = len(interactions)
    top_partners = [
        i.get('partner_symbol')
        or i.get('partner')
        or 'Unknown'
        for i in interactions[:5]
    ]
    partners_str = ", ".join(top_partners) if top_partners else "No interactions found"

    # Literature findings from papers - get top 3 key findings (increased limit)
    paper_findings = []
    for paper in papers[:3]:
        finding = paper.get('key_finding', '')
        if finding:
            paper_findings.append(finding[:300])
    paper_findings_str = " | ".join(paper_findings) if paper_findings else "No paper findings available"

    # Literature quality signal — based on ranker scores (1-5)
    # Score ≤ 2 = gene only mentioned incidentally; score ≥ 4 = paper specifically about the gene
    relevance_scores = [p.get('relevance_score', 0) for p in papers if p.get('relevance_score')]
    max_relevance = max(relevance_scores) if relevance_scores else 0
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

    if max_relevance <= 2:
        literature_quality_note = f"""
⚠️  LITERATURE QUALITY WARNING — SPARSE/INDIRECT LITERATURE:
The highest relevance score across all top-ranked papers is {max_relevance}/5 (average: {avg_relevance:.1f}/5).
This means NO paper in the set specifically studies {gene} — they only mention it incidentally
(e.g. as one of many hits in a GWAS, multi-gene list, or genome-wide screen).

For SENTENCE 6 (LITERATURE BREADTH), do NOT write as if these papers are about {gene}.
Instead use this format:
"{gene} has limited dedicated literature. It has appeared as an incidental finding across
studies of [conditions/topics from the themes], but no papers in this set focus specifically on {gene}."
"""
    elif max_relevance == 3:
        literature_quality_note = f"""
ℹ️  LITERATURE NOTE — MIXED LITERATURE:
The highest relevance score is {max_relevance}/5 (average: {avg_relevance:.1f}/5).
Some papers discuss {gene} but not as their primary subject.
For SENTENCE 6, frame findings carefully — use "has been studied in the context of" rather than
"the literature on {gene} covers" when themes come from papers where {gene} is not the main focus.
"""
    else:
        literature_quality_note = ""

    total_papers_available = literature_total if literature_total is not None else len(papers)
    literature_coverage = (
        f"- Candidate papers available: {total_papers_available}\n"
        f"- Top-ranked papers summarised here: {len(papers)}"
    )

    # Literature findings summary from AnalyzeLiteratureFindings node
    lit_context_str = ""
    if literature_findings:
        themes = literature_findings.get('research_themes', [])
        recent_count = literature_findings.get('recent_papers_count', 0)
        total_papers = literature_findings.get('total_papers', 0)
        known_findings = literature_findings.get('known_findings', [])[:3]

        lit_context_str = f"""
**Literature Analysis Summary:**
- Total papers analysed: {total_papers}
- Recent papers (last 2 years): {recent_count}
- Key research themes: {', '.join(themes[:5]) if themes else 'N/A'}
- Established findings: {'; '.join(known_findings) if known_findings else 'N/A'}
"""

    # Cross-gene synthesis (for multi-gene queries)
    cross_gene_str = ""
    other_genes = [g for g in (all_genes or []) if g != gene]
    if cross_gene_synthesis and other_genes:
        cross_gene_str = f"""
**Cross-Gene Relationships:**
Other genes in query: {', '.join(other_genes)}
Synthesis: {cross_gene_synthesis[:500]}
"""

    # Context info
    context_str = "human"
    if context:
        if hasattr(context, 'model_dump'):
            ctx_dict = context.model_dump()
        elif hasattr(context, '__dict__'):
            ctx_dict = context.__dict__
        else:
            ctx_dict = {}
        cell_type = ctx_dict.get('cell_type', '')
        disease = ctx_dict.get('disease', '')
        context_str = f"Organism: {ctx_dict.get('organism', 'human')}"
        if cell_type:
            context_str += f", Cell type: {cell_type}"
        if disease:
            context_str += f", Disease context: {disease}"

    prompt = f"""Write a review-style summary paragraph for {gene} (4-6 sentences).

RESEARCH QUESTION: {query}
CONTEXT: {context_str}

=== SOURCE DATA (ground every claim in this data) ===

FUNCTION (from database): {function_desc}

GO BIOLOGICAL PROCESSES: {go_str}

CELLULAR LOCALISATION: {cellular_component_str}

HIGHEST-EXPRESSED TISSUE: {tissue_str}

PROTEIN INTERACTIONS: {num_interactions} interactions; top partners: {partners_str}

LITERATURE ({len(papers)} top-ranked of {total_papers_available} total):
{paper_findings_str}
{lit_context_str}{cross_gene_str}

=== REQUIRED STRUCTURE (follow this order exactly) ===

SENTENCE 1-2 (FUNCTION + INTERACTIONS):
Split across two clean sentences:
- Sentence 1: What {gene} encodes and its core molecular function/biological role only. Keep this tight — do not include interactions, localisation, or secondary pathways here.
- Sentence 2: Name the top interaction partners and state the number of total interactions.
  Do NOT claim that interactions occur in a specific subcellular compartment — the interaction
  data only provides partner names and confidence scores, not locations. If you want to mention
  where the protein is localised, do it in a separate clause that is clearly about the protein's
  annotation, not about where interactions happen (e.g. "…partners within the Mediator complex;
  MED12 is annotated to the nucleoplasm" is acceptable — "interacts with X within the nucleoplasm"
  is NOT).

Example:
"RFC1 encodes the large subunit of the replication factor C clamp-loader complex, where it plays a central role in DNA-templated replication. It coordinates with 182 partners — most prominently RFC5, RFC4, and PCNA — to load the replication clamp at the replication fork."

SENTENCE 3 (EXPRESSION):
One sentence only. State ONLY the highest-expressed tissue provided above.
Example: "Expression is highest in the [tissue]."

SENTENCE 4-5 (DISEASE/CLINICAL):
Synthesise the clinical picture thematically from the literature findings:
- What conditions/diseases are caused by or strongly associated with {gene}
- Use direct causal language if the gene is an established disease gene ("{gene} causes Y")
- Inheritance pattern if mentioned (e.g. X-linked, autosomal dominant, de novo)
- Phenotypic spectrum (broad strokes only — conditions and features, not mechanisms)

STRICT SOURCE RULE: Every clinical claim must be directly traceable to either the LITERATURE findings or Literature Analysis Summary above. Do not infer, extrapolate, or generalise beyond what is explicitly stated in those fields. If a clinical detail is not in the source data, do not include it.

DO NOT include: specific variant nomenclature, study methodology, functional or mechanistic evidence details, or incidental pathway associations

SENTENCE 6 (LITERATURE BREADTH):
{literature_quality_note}End with a sentence summarising the breadth of literature themes. If the literature quality warning above is present, follow its format exactly. Otherwise use:
"The literature on {gene} spans themes including [theme1], [theme2], and [theme3], with recent studies investigating [most recent or prominent theme]."

Use the themes from the Literature Analysis Summary but do NOT reproduce any counts or numbers. Do not use the words 'recent', 'latest', or time-relative phrases such as 'in the last X years'. Only make claims about paper counts or specific named findings. Do not claim papers are 'consistent with' or 'support' themes.

=== STRICT RULES ===
- Every claim must trace to the source data above — no hallucination
- Use {gene} naturally throughout
- Plain text only (no markdown, bullets, or headers)
- Only mention interaction partners from the list provided
- Do NOT list multiple expression tissues — only the highest one
- Do NOT include specific variant names or study methodology details
- Do NOT load the opening sentence with more than one main idea — function first, interactions second, always in separate sentences
- Only mention secondary pathways (e.g. ubiquitination) if they are a primary, well-established function — do not include incidental associations
- If the gene is a known causative disease gene, state this directly — do not use hedging language like "consistent with", "implicated in", or "associated with" when the causal relationship is established
- Do NOT reference functional or mechanistic study details as clinical evidence (e.g. "functional data in neural cells point to...", "mouse models show...") — describe phenotypic outcomes only
- Every clinical claim must trace to the LITERATURE or Literature Analysis Summary fields — if it isn't in the source data, do not include it
- Do NOT include specific paper counts, study numbers, or year counts in the summary
- Do NOT make temporal trend claims such as "recent work is increasingly focused on X" or "growing interest in Y" unless the literature metadata explicitly provides year-by-year counts showing an upward trend — instead describe current themes factually: "Recent studies have investigated X, including [examples]"
- Do NOT assert that protein interactions occur in a specific subcellular compartment (e.g.
  "interacts with X within the nucleoplasm"). The interaction dataset provides partner names
  and confidence scores only — compartment information comes from GO cellular component
  annotations and describes the protein's location, not the location of specific interactions.
  If you wish to convey both, write two separate clauses: one for the interaction partners,
  one for the GO CC annotation.
"""

    return prompt


def _create_fallback_summary(gene: str, gene_data: dict, interpretations: dict) -> str:
    """Create a basic fallback summary if LLM fails."""

    # Try to use existing interpretations
    if interpretations:
        func = interpretations.get('functional_summary', '')
        expr = interpretations.get('expression_interpretation', '')
        if func:
            return func
        if expr:
            return expr

    # Use database description
    desc = gene_data.get('long_description', gene_data.get('description', ''))
    if desc:
        return desc[:500]

    return f"Comprehensive summary not available for {gene}."
