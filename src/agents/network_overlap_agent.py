"""
Network overlap analysis agent.
Analyzes protein-protein interaction overlaps between genes in the dataset.
"""

import asyncio
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
from openai import AsyncOpenAI
import os
from src.config import get_active_model
from src.graph.state import _accumulate_tokens

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def analyze_network_overlap(
    all_gene_data: Dict[str, Dict],
    *,
    state=None,
    node_name: str = "AnalyzeNetworkOverlap"
) -> Dict[str, Any]:
    """
    Analyze protein-protein interaction overlap across all genes.

    Args:
        all_gene_data: Dictionary mapping gene symbols to their data
                      Each gene data must contain 'interactions' key
        state: Optional state object for token tracking
        node_name: Name of the calling node for token tracking

    Returns:
        Dictionary containing network overlap analysis results
    """

    genes = list(all_gene_data.keys())

    # Step 1: Extract all interactions for each gene
    gene_interactions = {}
    for gene_symbol, gene_data in all_gene_data.items():
        interactions = gene_data.get('interactions', [])
        gene_interactions[gene_symbol] = interactions

    # Step 2: Find shared interaction partners
    shared_partners = find_shared_partners(gene_interactions)

    # Step 3: Check for direct interactions between query genes
    direct_interactions = find_direct_interactions(genes, gene_interactions)

    # Step 4: Classify genes as hubs
    hub_analysis = classify_hubs(gene_interactions)

    # Step 5: Identify functional modules
    network_modules = identify_modules(genes, shared_partners, gene_interactions)

    # Step 6: Calculate network statistics
    network_stats = calculate_network_statistics(gene_interactions, shared_partners)

    # Step 7: Generate LLM interpretation
    llm_summary = await generate_network_interpretation(
        genes, shared_partners, hub_analysis, network_modules, network_stats,
        state=state,
        node_name=node_name
    )

    # Assemble results
    return {
        'genes_analyzed': genes,
        'total_interactions': sum(len(interactions) for interactions in gene_interactions.values()),
        'shared_partners': shared_partners,
        'direct_interactions': direct_interactions,
        'hub_analysis': hub_analysis,
        'network_modules': network_modules,
        'network_statistics': network_stats,
        'llm_summary': llm_summary,
        'analysis_timestamp': None,  # Will be set by node
        'genes_with_no_interactions': [g for g, i in gene_interactions.items() if not i]
    }


def find_shared_partners(gene_interactions: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Find proteins that interact with multiple query genes.

    Returns list of dicts with:
    - partner: protein name
    - shared_by: list of genes
    - confidence_scores: dict of gene -> score
    - functional_context: description
    """

    # Build mapping: partner -> [(gene, score, description)]
    partner_map = defaultdict(list)

    for gene, interactions in gene_interactions.items():
        for interaction in interactions:
            partner = interaction.get('partner_symbol') or interaction.get('partner')
            if not partner:
                continue

            raw_score = interaction.get('score', interaction.get('combined_score', 0)) or 0
            score = raw_score / 1000.0 if raw_score > 1 else raw_score
            description = interaction.get('partner_annotation') or interaction.get('description', '')

            partner_map[partner].append({
                'gene': gene,
                'score': score,
                'description': description
            })

    # Find partners shared by multiple genes (2+)
    shared = []
    for partner, gene_list in partner_map.items():
        if len(gene_list) >= 2:
            genes_sharing = [g['gene'] for g in gene_list]
            scores = {g['gene']: g['score'] for g in gene_list}
            descriptions = [g['description'] for g in gene_list if g['description']]

            shared.append({
                'partner': partner,
                'shared_by': genes_sharing,
                'confidence_scores': scores,
                'functional_context': descriptions[0] if descriptions else f"{partner} interacts with {', '.join(genes_sharing)}"
            })

    # Sort by number of genes sharing
    shared.sort(key=lambda x: len(x['shared_by']), reverse=True)

    return shared


def find_direct_interactions(query_genes: List[str], gene_interactions: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Check if any query genes interact directly with each other.
    """

    direct = []
    query_set = set(query_genes)

    for gene_a in query_genes:
        interactions = gene_interactions.get(gene_a, [])
        partners = {
            (i.get('partner_symbol') or i.get('partner'))
            for i in interactions if (i.get('partner_symbol') or i.get('partner'))
        }

        # Check if any other query gene is a partner
        for gene_b in query_set - {gene_a}:
            if gene_b in partners:
                # Find the interaction details
                interaction = next(
                    i for i in interactions
                    if (i.get('partner_symbol') or i.get('partner')) == gene_b
                )
                raw_score = interaction.get('score', interaction.get('combined_score', 0)) or 0
                score = raw_score / 1000.0 if raw_score > 1 else raw_score

                direct.append({
                    'gene_a': gene_a,
                    'gene_b': gene_b,
                    'confidence': score,
                    'exists': True,
                    'note': f"Direct interaction detected with confidence {score:.3f}"
                })

    # If no direct interactions, add note
    if not direct:
        for i, gene_a in enumerate(query_genes):
            for gene_b in query_genes[i+1:]:
                direct.append({
                    'gene_a': gene_a,
                    'gene_b': gene_b,
                    'confidence': 0.0,
                    'exists': False,
                    'note': "No direct interaction detected"
                })

    return direct


def classify_hubs(gene_interactions: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Classify genes as network hubs (threshold: 50+ interactions).
    """

    hub_analysis = []

    for gene, interactions in gene_interactions.items():
        interaction_count = len(interactions)
        is_hub = interaction_count >= 50

        # Determine hub classification based on gene function
        # This is a simple heuristic - could be enhanced with GO terms
        hub_type = "Network hub" if is_hub else "Non-hub"

        hub_analysis.append({
            'gene': gene,
            'total_interactions': interaction_count,
            'is_hub': is_hub,
            'hub_classification': hub_type,
            'description': f"{gene} has {interaction_count} interactions" +
                          (" and functions as a network hub" if is_hub else "")
        })

    # Sort by interaction count
    hub_analysis.sort(key=lambda x: x['total_interactions'], reverse=True)

    return hub_analysis


def identify_modules(query_genes: List[str], shared_partners: List[Dict],
                     gene_interactions: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Identify functional modules based on shared interaction patterns.

    A module is a group of 2+ query genes with multiple shared partners.
    """

    modules = []

    # Group genes by shared partners
    # gene_pair -> set of shared partners
    pair_partners = defaultdict(set)

    for shared in shared_partners:
        genes = shared['shared_by']
        partner = shared['partner']

        # For each pair in genes
        for i, gene_a in enumerate(genes):
            for gene_b in genes[i+1:]:
                pair_key = tuple(sorted([gene_a, gene_b]))
                pair_partners[pair_key].add(partner)

    # Find pairs with multiple shared partners (3+)
    strong_pairs = [(pair, partners) for pair, partners in pair_partners.items()
                    if len(partners) >= 3]

    # Sort by number of shared partners
    strong_pairs.sort(key=lambda x: len(x[1]), reverse=True)

    # Create modules from strong pairs
    for idx, (pair, partners) in enumerate(strong_pairs[:5], 1):  # Top 5 modules
        gene_a, gene_b = pair

        modules.append({
            'module_name': f"Module {idx}: {gene_a}-{gene_b}",
            'genes': list(pair),
            'shared_partners': sorted(partners)[:10],  # Top 10 partners
            'shared_partner_count': len(partners),
            'functional_theme': "Shared regulatory network",
            'description': f"{gene_a} and {gene_b} share {len(partners)} interaction partners"
        })

    return modules


def calculate_network_statistics(gene_interactions: Dict[str, List[Dict]],
                                 shared_partners: List[Dict]) -> Dict[str, Any]:
    """
    Calculate overall network statistics.
    """

    # Collect all unique partners
    all_partners = set()
    total_interactions = 0
    confidence_scores = []

    for gene, interactions in gene_interactions.items():
        total_interactions += len(interactions)
        for interaction in interactions:
            partner = interaction.get('partner_symbol') or interaction.get('partner')
            if partner:
                all_partners.add(partner)

            raw_score = interaction.get('score', interaction.get('combined_score', 0)) or 0
            score = raw_score / 1000.0 if raw_score > 1 else raw_score
            confidence_scores.append(score)

    hub_count = sum(1 for _, interactions in gene_interactions.items()
                   if len(interactions) >= 50)

    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    return {
        'total_unique_partners': len(all_partners),
        'shared_partner_count': len(shared_partners),
        'connectivity_index': len(shared_partners) / len(all_partners) if all_partners else 0,
        'average_confidence': avg_confidence,
        'total_interactions': total_interactions,
        'hub_genes_count': hub_count,
        'isolated_genes_count': sum(1 for _, i in gene_interactions.items() if not i)
    }


async def generate_network_interpretation(genes: List[str],
                                         shared_partners: List[Dict],
                                         hub_analysis: List[Dict],
                                         network_modules: List[Dict],
                                         network_stats: Dict,
                                         *,
                                         state=None,
                                         node_name: str = "AnalyzeNetworkOverlap") -> str:
    """
    Generate natural language interpretation of network analysis using LLM.

    Args:
        genes: List of gene symbols
        shared_partners: List of shared interaction partners
        hub_analysis: Hub classification data
        network_modules: Network module data
        network_stats: Network statistics
        state: Optional state object for token tracking
        node_name: Name of the calling node for token tracking

    Returns:
        Natural language interpretation string
    """

    # Prepare summary data for LLM
    genes_str = ", ".join(genes)

    top_shared = shared_partners[:5]
    shared_summary = "\n".join([
        f"- {s['partner']}: shared by {', '.join(s['shared_by'])}"
        for s in top_shared
    ])

    hub_summary = "\n".join([
        f"- {h['gene']}: {h['total_interactions']} interactions ({'hub' if h['is_hub'] else 'non-hub'})"
        for h in hub_analysis
    ])

    modules_summary = "\n".join([
        f"- {m['module_name']}: {m['shared_partner_count']} shared partners"
        for m in network_modules[:3]
    ])

    prompt = f"""Analyse the protein interaction network for these genes: {genes_str}

Network Data:
- Total unique interaction partners: {network_stats['total_unique_partners']}
- Shared partners: {network_stats['shared_partner_count']}
- Hub genes: {network_stats['hub_genes_count']}/{len(genes)}

Top Shared Partners:
{shared_summary if shared_summary else "None"}

Hub Analysis:
{hub_summary}

Network Modules:
{modules_summary if modules_summary else "No clear modules identified"}

Write a 3-4 paragraph interpretation covering:
1. Overall network structure and connectivity
2. Functional modules and what they suggest about gene relationships
3. Key shared partners and their biological significance
4. Implications for understanding how these genes work together

Requirements:
- Academic tone
- Focus on biological insights
- Explain what the network topology tells us
- 3-4 paragraphs, no bullet points
- Do not repeat raw numbers already shown elsewhere
"""

    response = await client.chat.completions.create(
        model=get_active_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=500
    )

    # Track token usage
    usage = getattr(response, "usage", None)
    _accumulate_tokens(state, node_name, usage)

    return response.choices[0].message.content.strip()
