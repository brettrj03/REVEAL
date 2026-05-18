"""
Gene Network Visualisation Module

Creates interactive Plotly network visualisations showing connections
between query genes based on protein-protein interaction data.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go


def create_query_gene_network(
    gene_profiles: Dict[str, Any],
    query_genes: List[str],
    min_confidence: float = 0.7,
    max_shared_partners: int = 10,
    show_gene_specific: bool = False,
    layout_style: str = 'centered'
) -> go.Figure:
    """
    Creates interactive Plotly network showing connections between query genes.

    Args:
        gene_profiles: Dict from state.gene_profiles with interaction data
        query_genes: List of gene symbols from the query
        min_confidence: Minimum interaction score to include (0-1 scale)
        max_shared_partners: Maximum shared partners to display
        show_gene_specific: Include partners that only interact with one query gene
        layout_style: Layout algorithm ('centered', 'circular')

    Returns:
        Plotly Figure object
    """
    # Normalize query genes to uppercase for matching
    query_genes_upper = [g.upper() for g in query_genes]
    query_genes_set = set(query_genes_upper)

    # Handle edge case: no genes
    if not query_genes:
        return _create_empty_figure("No genes provided")

    # Handle edge case: single gene
    if len(query_genes) == 1:
        return _create_single_gene_network(
            gene_profiles, query_genes[0], min_confidence
        )

    # Extract interaction data for each query gene
    gene_interactions = {}
    for gene in query_genes:
        gene_upper = gene.upper()
        profile = gene_profiles.get(gene) or gene_profiles.get(gene_upper)

        if profile:
            # Handle both GeneProfile objects and dicts
            if hasattr(profile, 'interactions'):
                interactions = profile.interactions
            else:
                interactions = profile.get('interactions', [])

            # Build set of partners with confidence scores
            partners = {}
            for inter in interactions:
                partner = (
                    inter.get('partner_symbol')
                    or inter.get('partner')
                    or ''
                ).upper()

                # Get confidence score (normalize to 0-1)
                raw_score = inter.get('score', inter.get('combined_score', 0)) or 0
                score = raw_score / 1000 if raw_score > 1 else raw_score

                if partner and score >= min_confidence:
                    partners[partner] = max(partners.get(partner, 0), score)

            gene_interactions[gene_upper] = partners

    # Check if we have any interaction data
    if not any(gene_interactions.values()):
        return _create_empty_figure("No interaction data available for query genes")

    # Find direct interactions between query genes
    direct_edges = []
    for i, gene1 in enumerate(query_genes_upper):
        for gene2 in query_genes_upper[i+1:]:
            partners1 = gene_interactions.get(gene1, {})
            partners2 = gene_interactions.get(gene2, {})

            # Check if gene2 is in gene1's partners or vice versa
            score = 0
            if gene2 in partners1:
                score = max(score, partners1[gene2])
            if gene1 in partners2:
                score = max(score, partners2[gene1])

            if score > 0:
                direct_edges.append((gene1, gene2, score))

    # Find shared interaction partners
    shared_partners = {}
    all_partners = set()
    for partners in gene_interactions.values():
        all_partners.update(partners.keys())

    # Remove query genes from potential shared partners
    all_partners -= query_genes_set

    for partner in all_partners:
        # Find which query genes share this partner
        shared_by = []
        total_score = 0
        for gene in query_genes_upper:
            partners = gene_interactions.get(gene, {})
            if partner in partners:
                shared_by.append(gene)
                total_score += partners[partner]

        if len(shared_by) >= 2:
            avg_score = total_score / len(shared_by)
            shared_partners[partner] = {
                'shared_by': shared_by,
                'avg_score': avg_score,
                'count': len(shared_by)
            }

    # Sort shared partners by number of genes sharing and avg score
    sorted_shared = sorted(
        shared_partners.items(),
        key=lambda x: (x[1]['count'], x[1]['avg_score']),
        reverse=True
    )[:max_shared_partners]

    # Check if we have any connections
    if not direct_edges and not sorted_shared:
        return _create_empty_figure(
            f"No direct or shared interactions found between {', '.join(query_genes)}"
        )

    # Build network visualisation
    return _build_network_figure(
        query_genes=query_genes,
        direct_edges=direct_edges,
        shared_partners=dict(sorted_shared),
        gene_interactions=gene_interactions,
        layout_style=layout_style
    )


def _create_empty_figure(message: str) -> go.Figure:
    """Create a figure with just an informational message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray")
    )
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


def _create_single_gene_network(
    gene_profiles: Dict[str, Any],
    gene: str,
    min_confidence: float
) -> go.Figure:
    """Create network for a single gene showing its top partners."""
    gene_upper = gene.upper()
    profile = gene_profiles.get(gene) or gene_profiles.get(gene_upper)

    if not profile:
        return _create_empty_figure(f"No data available for {gene}")

    # Get interactions
    if hasattr(profile, 'interactions'):
        interactions = profile.interactions
    else:
        interactions = profile.get('interactions', [])

    if not interactions:
        return _create_empty_figure(f"No interaction data available for {gene}")

    # Get top 10 partners by confidence
    partners = []
    for inter in interactions:
        partner = (
            inter.get('partner_symbol')
            or inter.get('partner')
            or ''
        )
        raw_score = inter.get('score', inter.get('combined_score', 0)) or 0
        score = raw_score / 1000 if raw_score > 1 else raw_score

        if partner and score >= min_confidence:
            partners.append((partner, score))

    partners = sorted(partners, key=lambda x: x[1], reverse=True)[:10]

    if not partners:
        return _create_empty_figure(
            f"No high-confidence interactions found for {gene}"
        )

    # Create radial layout
    node_x = [0]  # Center gene
    node_y = [0]
    node_text = [gene]
    node_colors = ['#1f77b4']  # Blue for query gene
    node_sizes = [40]

    edge_x = []
    edge_y = []

    for i, (partner, score) in enumerate(partners):
        # Position partners in a circle around center
        angle = 2 * math.pi * i / len(partners)
        x = math.cos(angle) * 1.5
        y = math.sin(angle) * 1.5

        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{partner}<br>Score: {score:.3f}")
        node_colors.append('#7f7f7f')  # Gray for partners
        node_sizes.append(25)

        # Add edge
        edge_x.extend([0, x, None])
        edge_y.extend([0, y, None])

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='#888'),
        hoverinfo='none'
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=node_sizes, color=node_colors),
        text=[gene] + [p[0] for p in partners],
        textposition='top center',
        hovertext=node_text,
        hoverinfo='text'
    ))

    fig.update_layout(
        title=f"Top Interaction Partners for {gene}",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def _build_network_figure(
    query_genes: List[str],
    direct_edges: List[Tuple[str, str, float]],
    shared_partners: Dict[str, Dict],
    gene_interactions: Dict[str, Dict[str, float]],
    layout_style: str
) -> go.Figure:
    """Build the main network figure with query genes and shared partners."""

    # Calculate positions
    positions = _calculate_positions(
        query_genes, shared_partners, layout_style
    )

    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_colors = []
    node_sizes = []

    # Add query gene nodes
    for gene in query_genes:
        gene_upper = gene.upper()
        pos = positions.get(gene_upper, (0, 0))
        node_x.append(pos[0])
        node_y.append(pos[1])
        node_text.append(gene)

        # Build hover text
        partners = gene_interactions.get(gene_upper, {})
        hover = f"<b>{gene}</b> (Query Gene)<br>"
        hover += f"Total partners: {len(partners)}<br>"

        # Count shared partners
        shared_count = sum(
            1 for p in shared_partners
            if gene_upper in shared_partners[p]['shared_by']
        )
        hover += f"Shared partners: {shared_count}"
        node_hover.append(hover)

        node_colors.append('#1f77b4')  # Blue for query genes
        node_sizes.append(45)

    # Add shared partner nodes
    for partner, info in shared_partners.items():
        pos = positions.get(partner, (0, 0))
        node_x.append(pos[0])
        node_y.append(pos[1])
        node_text.append(partner)

        # Build hover text
        shared_by_str = ', '.join(info['shared_by'])
        hover = f"<b>{partner}</b> (Shared Partner)<br>"
        hover += f"Shared by: {shared_by_str}<br>"
        hover += f"Avg confidence: {info['avg_score']:.3f}"
        node_hover.append(hover)

        node_colors.append('#ff7f0e')  # Orange for shared partners
        node_sizes.append(30)

    # Prepare edge data
    edge_traces = []

    # Add direct edges between query genes (thick blue)
    for gene1, gene2, score in direct_edges:
        pos1 = positions.get(gene1, (0, 0))
        pos2 = positions.get(gene2, (0, 0))

        edge_traces.append(go.Scatter(
            x=[pos1[0], pos2[0], None],
            y=[pos1[1], pos2[1], None],
            mode='lines',
            line=dict(width=3, color='#1f77b4'),
            hoverinfo='text',
            hovertext=f"Direct: {gene1} ↔ {gene2}<br>Confidence: {score:.3f}",
            showlegend=False
        ))

    # Add edges from query genes to shared partners (thinner, gray)
    for partner, info in shared_partners.items():
        pos_partner = positions.get(partner, (0, 0))
        for gene in info['shared_by']:
            pos_gene = positions.get(gene, (0, 0))
            gene_score = gene_interactions.get(gene, {}).get(partner, 0)

            edge_traces.append(go.Scatter(
                x=[pos_gene[0], pos_partner[0], None],
                y=[pos_gene[1], pos_partner[1], None],
                mode='lines',
                line=dict(width=1, color='rgba(150, 150, 150, 0.5)'),
                hoverinfo='text',
                hovertext=f"{gene} → {partner}<br>Confidence: {gene_score:.3f}",
                showlegend=False
            ))

    # Create figure
    fig = go.Figure()

    # Add all edges first (so they're behind nodes)
    for trace in edge_traces:
        fig.add_trace(trace)

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=node_text,
        textposition='top center',
        textfont=dict(size=11),
        hovertext=node_hover,
        hoverinfo='text',
        showlegend=False
    ))

    # Add legend annotations
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=15, color='#1f77b4'),
        name='Query Genes',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=12, color='#ff7f0e'),
        name='Shared Partners',
        showlegend=True
    ))

    # Summary text
    summary = f"{len(query_genes)} query genes, {len(direct_edges)} direct connections, {len(shared_partners)} shared partners"

    fig.update_layout(
        title=dict(
            text="Query Gene Network",
            subtitle=dict(text=summary, font=dict(size=12, color="gray"))
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        height=550,
        margin=dict(l=20, r=20, t=70, b=20),
        hovermode='closest'
    )

    return fig


def _calculate_positions(
    query_genes: List[str],
    shared_partners: Dict[str, Dict],
    layout_style: str
) -> Dict[str, Tuple[float, float]]:
    """Calculate node positions for the network layout."""
    positions = {}
    query_genes_upper = [g.upper() for g in query_genes]

    if layout_style == 'circular':
        # Query genes in inner circle
        for i, gene in enumerate(query_genes_upper):
            angle = 2 * math.pi * i / len(query_genes_upper)
            positions[gene] = (math.cos(angle) * 1.5, math.sin(angle) * 1.5)

        # Shared partners in outer circle
        for i, partner in enumerate(shared_partners.keys()):
            angle = 2 * math.pi * i / max(len(shared_partners), 1) + 0.3
            positions[partner] = (math.cos(angle) * 3, math.sin(angle) * 3)

    else:  # 'centered' layout
        # Position query genes based on count
        n_genes = len(query_genes_upper)

        if n_genes == 2:
            positions[query_genes_upper[0]] = (-1.5, 0)
            positions[query_genes_upper[1]] = (1.5, 0)
        elif n_genes == 3:
            positions[query_genes_upper[0]] = (0, 1.5)
            positions[query_genes_upper[1]] = (-1.3, -0.75)
            positions[query_genes_upper[2]] = (1.3, -0.75)
        else:
            # Circular for 4+
            for i, gene in enumerate(query_genes_upper):
                angle = 2 * math.pi * i / n_genes - math.pi / 2
                positions[gene] = (math.cos(angle) * 1.5, math.sin(angle) * 1.5)

        # Position shared partners around the edges
        # Group by which genes they're shared by
        partner_list = list(shared_partners.items())

        for i, (partner, info) in enumerate(partner_list):
            shared_by = info['shared_by']

            if len(shared_by) == len(query_genes_upper):
                # Shared by all - place in center area
                angle = 2 * math.pi * i / max(len(partner_list), 1)
                r = 0.5 + (i % 3) * 0.3
                positions[partner] = (math.cos(angle) * r, math.sin(angle) * r)
            else:
                # Place between the genes that share this partner
                avg_x = sum(positions[g][0] for g in shared_by) / len(shared_by)
                avg_y = sum(positions[g][1] for g in shared_by) / len(shared_by)

                # Push outward from center
                dist = math.sqrt(avg_x**2 + avg_y**2)
                if dist > 0:
                    scale = 2.5 / dist
                else:
                    scale = 2.5
                    angle = 2 * math.pi * i / max(len(partner_list), 1)
                    avg_x, avg_y = math.cos(angle), math.sin(angle)

                # Add slight offset to prevent overlap
                offset = (i % 5 - 2) * 0.2
                positions[partner] = (avg_x * scale + offset, avg_y * scale + offset * 0.5)

    return positions
