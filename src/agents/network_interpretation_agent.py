"""LLM agent for interpreting network overlap findings."""

from typing import Any, Dict, List, Optional

import json
import textwrap

from pydantic import BaseModel
from pydantic_ai import Agent
from src.graph.state import _accumulate_tokens
from src.config import get_active_model


class NetworkInterpretationContext(BaseModel):
    genes: List[str]
    hub_proteins: List[Dict[str, Any]]
    direct_interactions: List[Dict[str, Any]]
    network_stats: Optional[Dict[str, Any]] = None
    experimental_context: Optional[Dict[str, Any]] = None


network_interpreter = Agent(
    'openai:gpt-4.1-mini',
    deps_type=NetworkInterpretationContext,
    system_prompt="""
You are a molecular biology expert interpreting protein-protein
interaction network data for REVEAL (Retrieval and Evidence-based
Validated Interpretation Analysis for gene Lists).

You will be given:
- The query genes being analysed
- Their shared interaction partners
- Any direct interactions between query genes
- Summary network statistics

Your task is to write a single interpretive paragraph of exactly
5 to 6 sentences about what the interaction pattern suggests
biologically.

STRICT RULES — follow all of these without exception:

1. DO NOT mention any specific numbers, scores, counts, or
   percentages. These are already displayed in tables above
   your output. Do not write things like "296 shared partners",
   "a score of 0.794", "10 hub proteins", or "1717 total
   interactions".

2. DO NOT list or name every shared partner. You may name 1–2
   partners maximum, only if doing so adds meaningful biological
   context that cannot be expressed without naming them.

3. DO NOT describe what the data shows — describe what it
   suggests or implies biologically. Every sentence should
   contain an interpretive claim, not a factual restatement.
   Avoid phrases like "the data shows", "the table indicates",
   "we can see that".

4. Focus on biological meaning: what does the interaction
   pattern suggest about co-regulation, shared pathway
   membership, functional redundancy, or disease relevance?
   What might this mean for how these genes coordinate their
   activity in the cell?

5. Acknowledge uncertainty appropriately. Use language like
   "suggests", "may indicate", "is consistent with", "points
   towards", rather than stating conclusions as established fact.

6. Write for a researcher audience — use correct molecular
   biology terminology but do not over-explain basic concepts.

7. The paragraph must be exactly 5 to 6 sentences. No more,
   no less. Do not use bullet points, headers, or subheadings.
   Plain prose only.

GOOD EXAMPLE SENTENCE:
"The convergence of both genes on shared partners involved in
DNA damage sensing suggests they may participate in a coordinated
checkpoint response, potentially acting through parallel rather
than identical mechanisms."

BAD EXAMPLE SENTENCE (too factual, repeats table data):
"BRCA1 and MDM2 share 296 common interaction partners with an
average confidence score of 0.857, including TP53, H2AX, and RAD51."
""",
)


async def generate_network_interpretation(
    context: NetworkInterpretationContext,
    *,
    state=None,
    node_name: str = "InterpretNetworkOverlap",
) -> str:
    """Run the LLM agent and return its narrative output."""

    lines: List[str] = []
    if context.genes:
        lines.append(f"Genes analysed: {', '.join(context.genes)}")

    if context.hub_proteins:
        lines.append("Top shared partners:")
        for partner in context.hub_proteins[:10]:
            genes = ", ".join(partner.get('genes', []))
            conf = partner.get('avg_confidence')
            lines.append(
                f"  - {partner.get('protein')} (avg confidence {conf if conf is not None else 'n/a'}) links {genes}"
            )

    if context.direct_interactions:
        lines.append("Direct interactions:")
        for interaction in context.direct_interactions[:10]:
            lines.append(
                f"  - {interaction.get('gene_a')} ↔ {interaction.get('gene_b')} (score {interaction.get('score'):0.3f})"
            )

    if context.network_stats:
        lines.append("Network statistics:")
        lines.append(json.dumps(context.network_stats, indent=2))

    if context.experimental_context:
        lines.append("Experimental context:")
        lines.append(json.dumps(context.experimental_context, indent=2))

    prompt = textwrap.dedent(
        f"""
        Using the summarised data below, explain the biological implications of the
        protein interaction overlap. Reference the specific genes, hub partners, and
        any direct interactions to describe coordination between pathways.

        {chr(10).join(lines)}
        """
    )

    result = await network_interpreter.run(prompt, deps=context, model=f'openai:{get_active_model()}')
    usage = result.usage() if callable(getattr(result, "usage", None)) else None
    _accumulate_tokens(state, node_name, usage)
    return result.output
