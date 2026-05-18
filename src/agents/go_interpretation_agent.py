"""LLM agent for interpreting GO term overlap results."""

from typing import Any, Dict, List, Optional

import json
import textwrap

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from src.graph.state import _accumulate_tokens
from src.config import get_active_model


class EnrichedGoTerm(BaseModel):
    term_name: str
    go_id: str
    namespace: str
    definition: str = ""
    depth: int = 0
    genes: List[str] = Field(default_factory=list)


class GoInterpretationContext(BaseModel):
    genes: List[str]
    shared_terms: List[EnrichedGoTerm]
    overlap_stats: Dict[str, Any]
    experimental_context: Optional[Dict[str, Any]] = None


GO_SYSTEM_PROMPT = """
You are a bioinformatics analyst interpreting shared Gene Ontology (GO) 
term overlap across a gene set.

You will be given enriched GO term data including term names, GO IDs, 
biological definitions, depth scores (higher = more specific), and which 
genes share each term.

Every claim you make must be directly traceable to one of these four 
sources:
1. term_name and go_id — from shared_terms
2. gene attribution — from the genes list for each term
3. biological meaning — use only the definition field provided, 
   do not paraphrase or extend it beyond one clause
4. counts and ratio — from overlap_stats

Do not make any statement not directly derivable from these four sources.
Do not infer mechanisms, pathways, or interactions beyond what the
definition field states.
Use appropriately hedged language such as "suggests", "may indicate", or
"is consistent with" when making inferences beyond direct GO annotation.
Maximum 5 sentences.
""".strip()


go_interpreter = Agent(
    'openai:gpt-4.1-mini',
    deps_type=GoInterpretationContext,
    system_prompt=GO_SYSTEM_PROMPT,
)
_NAMESPACE_CODES = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}


def _format_shared_terms_block(shared_terms: List[EnrichedGoTerm]) -> str:
    if not shared_terms:
        return "No shared terms provided."

    lines: List[str] = []
    for term in shared_terms:
        namespace_code = _NAMESPACE_CODES.get(term.namespace, term.namespace or "—")
        genes = ", ".join(term.genes) if term.genes else "None"
        definition = term.definition.strip()
        definition_segment = f'definition: "{definition}"' if definition else "definition: (not provided)"
        lines.append(
            f"- {term.term_name} ({term.go_id}) | depth: {term.depth} | namespace: {namespace_code} | "
            f"genes: {genes} | {definition_segment}"
        )
    return "\n".join(lines)


def _get_instruction_block(term_count: int) -> str:
    """Return prose format instructions based on the number of shared terms."""
    prose_reinforcement = (
        "Output must be flowing prose paragraphs. Do not use markdown formatting, "
        "numbered lists, bullet points, bold text, or headers of any kind."
    )

    if term_count <= 2:
        return f"""There are 1-2 shared terms. Write in prose sentences only — no lists,
no bullet points, no numbered items. Structure as:
- Sentence 1: State the total shared terms, total GO terms, and overlap ratio.
- Sentence 2-3: For each shared term, state the term name, GO ID, which genes
  share it, and its biological meaning using only the definition provided —
  maximum one clause per term.
- Final sentence: State what the overlap pattern indicates about functional
  similarity or divergence across the gene set.

{prose_reinforcement}"""

    if term_count <= 7:
        return f"""There are 3-7 shared terms. Write in prose sentences only — no lists,
no bullet points, no numbered items. Structure as:
- Sentence 1: State the total shared terms, total GO terms, and overlap ratio.
- One sentence per namespace that has shared terms: name the highest-depth
  term in that namespace, state which genes share it, and use only the
  definition field for biological meaning — one clause maximum.
- Final sentence: State what the overall overlap pattern indicates about
  shared biology across the gene set.

{prose_reinforcement}"""

    return f"""There are 8 or more shared terms. Write in prose sentences only — no
lists, no bullet points, no numbered items. Structure as:
- Sentence 1: State the total shared terms, total GO terms, and overlap ratio.
- Sentence 2: Name the namespace that dominates and identify the most specific
  functional cluster within it (highest depth terms), stating which genes share them.
- Sentence 3: Identify the highest-depth terms in any other namespaces with
  shared terms and which genes share them.
- Final sentence: State what the overlap ratio and term distribution indicates
  about whether this gene set operates in a coordinated biological programme
  or independent functional domains.

{prose_reinforcement}"""


async def generate_go_interpretation(
    context: GoInterpretationContext,
    *,
    state=None,
    node_name: str = "InterpretGoPatterns",
) -> str:
    """Produce a natural-language interpretation of GO overlap."""

    shared_terms_formatted = _format_shared_terms_block(context.shared_terms)

    term_count = len(context.shared_terms)
    instruction_block = _get_instruction_block(term_count)

    genes_str = ", ".join(context.genes) if context.genes else "No genes specified"

    exp_context_str = (
        json.dumps(context.experimental_context, indent=2)
        if context.experimental_context
        else "an unspecified experimental context"
    )

    total_shared = (
        context.overlap_stats.get('shared_go_terms')
        or context.overlap_stats.get('shared_terms')
        or term_count
    )
    total_go_terms = (
        context.overlap_stats.get('total_go_terms')
        or context.overlap_stats.get('total_terms')
        or 0
    )
    ratio_value = None
    if context.overlap_stats:
        ratio_value = (
            context.overlap_stats.get('overlap_ratio')
            if context.overlap_stats.get('overlap_ratio') is not None
            else context.overlap_stats.get('shared_total_ratio')
        )
    if ratio_value is None:
        ratio_value = 0
    try:
        ratio_display = f"{float(ratio_value):.3f}"
    except (TypeError, ValueError):
        ratio_display = str(ratio_value)

    term_selection_rules = textwrap.dedent(
        """
        Term selection rules — apply in this order:
        1. Prefer terms with the highest depth score
        2. Among equally deep terms, prefer those most relevant to the experimental context
        3. Never highlight terms with depth <= 2 unless they are the only shared terms available
        4. If any term name directly references one of the queried genes (e.g. "p53 binding" when TP53 is queried), prioritise that term above all others

        For each term you highlight, state:
        - The exact term name and GO ID as provided
        - Which specific genes share it
        - The biological meaning using only the definition field provided — maximum one clause, do not extend or infer beyond it

        Apply the same conditional structure based on term count:
        {instruction_block}
        """
    ).strip().replace("{instruction_block}", instruction_block)

    prompt = textwrap.dedent(
        f"""
        Interpret the GO term overlap for {genes_str} in the context of {exp_context_str}.

        Overlap statistics:
        - Total shared terms: {total_shared}
        - Total GO terms across gene set: {total_go_terms}
        - Overlap ratio: {ratio_display}

        Shared terms sorted by depth (most specific first):
        {shared_terms_formatted}

        {term_selection_rules}
        """
    ).strip()

    result = await go_interpreter.run(prompt, deps=context, model=f'openai:{get_active_model()}')
    usage = result.usage() if callable(getattr(result, "usage", None)) else None
    _accumulate_tokens(state, node_name, usage)
    return result.output
