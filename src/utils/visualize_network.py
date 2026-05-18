"""
PPI Network Visualiser
======================
Generates an interactive HTML protein–protein interaction network from
the pipeline state data for any set of genes.

Usage
-----
# From the command line (any genes):
    python src/utils/visualize_network.py LMNA LMNB1 TP53

# Auto-read genes from the most recent pipeline run:
    python src/utils/visualize_network.py --latest

# From Python / as a pipeline utility:
    from src.utils.visualize_network import build_network_html
    html_path = build_network_html(["LMNA", "LMNB1"], output_dir="results/")
"""

import math
from typing import Dict, List, Tuple

QUERY_GENE_COLORS = ['#e41a1c', '#1f4e8c', '#33a02c', '#984ea3', '#1f9e8e']
SHARED_PARTNER_COLOR = '#ff7f00'
DIRECT_EDGE_COLOR    = '#e41a1c'

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


def create_query_gene_network(
    gene_profiles,
    query_genes,
    min_confidence=0.7,
    max_shared_partners=10,
    show_gene_specific=False,
    layout_style='centered'
):
    if not _PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for create_query_gene_network")

    query_genes_upper = [g.upper() for g in query_genes]
    query_genes_set   = set(query_genes_upper)

    if not query_genes:
        return _create_empty_figure("No genes provided")
    if len(query_genes) == 1:
        return _create_single_gene_network(gene_profiles, query_genes[0], min_confidence)

    gene_interactions = {}
    for gene in query_genes:
        gene_upper = gene.upper()
        profile    = gene_profiles.get(gene) or gene_profiles.get(gene_upper)
        if profile:
            interactions = (
                profile.interactions if hasattr(profile, 'interactions')
                else profile.get('interactions', [])
            )
            partners = {}
            for inter in interactions:
                partner   = (inter.get('partner_symbol') or inter.get('partner') or '').upper()
                raw_score = inter.get('score', inter.get('combined_score', 0)) or 0
                score     = raw_score / 1000 if raw_score > 1 else raw_score
                if partner and score >= min_confidence:
                    partners[partner] = max(partners.get(partner, 0), score)
            gene_interactions[gene_upper] = partners

    if not any(gene_interactions.values()):
        return _create_empty_figure("No interaction data available for query genes")

    direct_edges = []
    seen_pairs   = set()
    for i, gene1 in enumerate(query_genes_upper):
        for gene2 in query_genes_upper[i + 1:]:
            pair = frozenset([gene1, gene2])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            p1    = gene_interactions.get(gene1, {})
            p2    = gene_interactions.get(gene2, {})
            score = max(p1.get(gene2, 0), p2.get(gene1, 0))
            if score > 0:
                direct_edges.append((gene1, gene2, score))

    all_partners = set()
    for partners in gene_interactions.values():
        all_partners.update(partners.keys())
    all_partners -= query_genes_set

    shared_partners = {}
    for partner in all_partners:
        shared_by   = []
        total_score = 0.0
        for gene in query_genes_upper:
            if partner in gene_interactions.get(gene, {}):
                shared_by.append(gene)
                total_score += gene_interactions[gene][partner]
        if len(shared_by) >= 2:
            shared_partners[partner] = {
                'shared_by': shared_by,
                'avg_score': total_score / len(shared_by),
                'count':     len(shared_by),
            }

    sorted_shared = dict(
        sorted(shared_partners.items(),
               key=lambda x: (x[1]['count'], x[1]['avg_score']),
               reverse=True)[:max_shared_partners]
    )

    if not direct_edges and not sorted_shared:
        return _create_empty_figure(
            f"No direct or shared interactions found between {', '.join(query_genes)}"
        )

    return _build_network_figure(
        query_genes=query_genes,
        direct_edges=direct_edges,
        shared_partners=sorted_shared,
        gene_interactions=gene_interactions,
        layout_style=layout_style,
    )


def _create_empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=300, margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def _create_single_gene_network(gene_profiles, gene, min_confidence):
    gene_upper = gene.upper()
    profile    = gene_profiles.get(gene) or gene_profiles.get(gene_upper)
    if not profile:
        return _create_empty_figure(f"No data available for {gene}")
    interactions = (
        profile.interactions if hasattr(profile, 'interactions')
        else profile.get('interactions', [])
    )
    if not interactions:
        return _create_empty_figure(f"No interaction data available for {gene}")
    partners = []
    for inter in interactions:
        partner   = (inter.get('partner_symbol') or inter.get('partner') or '')
        raw_score = inter.get('score', inter.get('combined_score', 0)) or 0
        score     = raw_score / 1000 if raw_score > 1 else raw_score
        if partner and score >= min_confidence:
            partners.append((partner, score))
    partners = sorted(partners, key=lambda x: x[1], reverse=True)[:10]
    if not partners:
        return _create_empty_figure(f"No high-confidence interactions found for {gene}")

    node_x = [0]; node_y = [0]; node_text = [gene]
    node_colors = [QUERY_GENE_COLORS[0]]; node_sizes = [40]
    edge_x = []; edge_y = []
    for i, (partner, score) in enumerate(partners):
        angle = 2 * math.pi * i / len(partners)
        x, y  = math.cos(angle) * 1.5, math.sin(angle) * 1.5
        node_x.append(x); node_y.append(y)
        node_text.append(f"{partner}<br>Score: {score:.3f}")
        node_colors.append('#7f7f7f'); node_sizes.append(25)
        edge_x.extend([0, x, None]); edge_y.extend([0, y, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                             line=dict(width=1, color='#888'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                             marker=dict(size=node_sizes, color=node_colors),
                             text=[gene] + [p[0] for p in partners],
                             textposition='top center',
                             hovertext=node_text, hoverinfo='text'))
    fig.update_layout(title=f"Top Interaction Partners for {gene}", showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      height=500, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def _build_network_figure(query_genes, direct_edges, shared_partners,
                           gene_interactions, layout_style):
    positions    = _calculate_positions(query_genes, shared_partners, layout_style)
    node_x = []; node_y = []; node_text = []; node_hover = []
    node_colors = []; node_sizes = []

    for idx, gene in enumerate(query_genes):
        gene_upper = gene.upper()
        pos        = positions.get(gene_upper, (0, 0))
        node_x.append(pos[0]); node_y.append(pos[1]); node_text.append(gene)
        partners     = gene_interactions.get(gene_upper, {})
        shared_count = sum(1 for p in shared_partners
                           if gene_upper in shared_partners[p]['shared_by'])
        node_hover.append(
            f"<b>{gene}</b> (Query Gene)<br>"
            f"Total partners: {len(partners)}<br>"
            f"Shared partners: {shared_count}"
        )
        node_colors.append(QUERY_GENE_COLORS[idx % len(QUERY_GENE_COLORS)])
        node_sizes.append(45)

    for partner, info in shared_partners.items():
        pos = positions.get(partner, (0, 0))
        node_x.append(pos[0]); node_y.append(pos[1]); node_text.append(partner)
        node_hover.append(
            f"<b>{partner}</b> (Shared Partner)<br>"
            f"Shared by: {', '.join(info['shared_by'])}<br>"
            f"Avg confidence: {info['avg_score']:.3f}"
        )
        node_colors.append(SHARED_PARTNER_COLOR); node_sizes.append(30)

    edge_traces = []
    for gene1, gene2, score in direct_edges:
        pos1 = positions.get(gene1, (0, 0)); pos2 = positions.get(gene2, (0, 0))
        edge_traces.append(go.Scatter(
            x=[pos1[0], pos2[0], None], y=[pos1[1], pos2[1], None], mode='lines',
            line=dict(width=max(1, score * 5), color=DIRECT_EDGE_COLOR),
            hoverinfo='text',
            hovertext=f"Direct: {gene1} ↔ {gene2}<br>Confidence: {score:.3f}",
            showlegend=False,
        ))
    for partner, info in shared_partners.items():
        pos_partner = positions.get(partner, (0, 0))
        for gene in info['shared_by']:
            pos_gene   = positions.get(gene, (0, 0))
            gene_score = gene_interactions.get(gene, {}).get(partner, 0)
            edge_traces.append(go.Scatter(
                x=[pos_gene[0], pos_partner[0], None],
                y=[pos_gene[1], pos_partner[1], None], mode='lines',
                line=dict(width=max(0.5, gene_score * 2), color='rgba(150,150,150,0.4)'),
                hoverinfo='text',
                hovertext=f"{gene} → {partner}<br>Confidence: {gene_score:.3f}",
                showlegend=False,
            ))

    fig = go.Figure()
    for trace in edge_traces:
        fig.add_trace(trace)
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
        text=node_text, textposition='top center', textfont=dict(size=11),
        hovertext=node_hover, hoverinfo='text', showlegend=False,
    ))
    for idx, gene in enumerate(query_genes):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(size=14, color=QUERY_GENE_COLORS[idx % len(QUERY_GENE_COLORS)]),
                                 name=f'{gene} (query gene)', showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(size=12, color=SHARED_PARTNER_COLOR),
                             name='Shared partner', showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(width=3, color=DIRECT_EDGE_COLOR),
                             name='Direct gene-gene interaction', showlegend=True))
    fig.update_layout(
        title=dict(text=" / ".join(query_genes) + " PPI Network"),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.85)", borderwidth=1),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        height=550, margin=dict(l=20, r=20, t=70, b=20), hovermode='closest',
    )
    return fig


def _calculate_positions(query_genes, shared_partners, layout_style):
    positions         = {}
    query_genes_upper = [g.upper() for g in query_genes]
    if layout_style == 'circular':
        for i, gene in enumerate(query_genes_upper):
            angle = 2 * math.pi * i / len(query_genes_upper)
            positions[gene] = (math.cos(angle) * 1.5, math.sin(angle) * 1.5)
        for i, partner in enumerate(shared_partners.keys()):
            angle = 2 * math.pi * i / max(len(shared_partners), 1) + 0.3
            positions[partner] = (math.cos(angle) * 3, math.sin(angle) * 3)
    else:
        n = len(query_genes_upper)
        if n == 2:
            positions[query_genes_upper[0]] = (-1.5, 0)
            positions[query_genes_upper[1]] = (1.5, 0)
        elif n == 3:
            positions[query_genes_upper[0]] = (0, 1.5)
            positions[query_genes_upper[1]] = (-1.3, -0.75)
            positions[query_genes_upper[2]] = (1.3, -0.75)
        else:
            for i, gene in enumerate(query_genes_upper):
                angle = 2 * math.pi * i / n - math.pi / 2
                positions[gene] = (math.cos(angle) * 1.5, math.sin(angle) * 1.5)
        partner_list = list(shared_partners.items())
        for i, (partner, info) in enumerate(partner_list):
            shared_by = info['shared_by']
            if len(shared_by) == len(query_genes_upper):
                angle = 2 * math.pi * i / max(len(partner_list), 1)
                r     = 0.5 + (i % 3) * 0.3
                positions[partner] = (math.cos(angle) * r, math.sin(angle) * r)
            else:
                avg_x = sum(positions[g][0] for g in shared_by) / len(shared_by)
                avg_y = sum(positions[g][1] for g in shared_by) / len(shared_by)
                dist  = math.sqrt(avg_x**2 + avg_y**2)
                if dist > 0:
                    scale = 2.5 / dist
                else:
                    scale = 2.5
                    angle = 2 * math.pi * i / max(len(partner_list), 1)
                    avg_x, avg_y = math.cos(angle), math.sin(angle)
                offset = (i % 5 - 2) * 0.2
                positions[partner] = (avg_x * scale + offset, avg_y * scale + offset * 0.5)
    return positions

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ── Paths (relative to project root) ──────────────────────────────────────────
_HERE   = Path(__file__).resolve().parent
_ROOT   = _HERE.parent.parent          # .../gene-annotation/
_DB     = _ROOT / "src" / "database" / "gene_database.sqlite"
_RUNS   = _ROOT / "results" / "stateful_pipeline"


# ── Database helpers ───────────────────────────────────────────────────────────

def _get_protein_id(cur, symbol: str) -> str | None:
    cur.execute("SELECT protein_id FROM string_proteins WHERE symbol = ?", (symbol,))
    row = cur.fetchone()
    return row[0] if row else None


def _fetch_interactions(cur, gene: str, min_score: int, limit: int) -> list[tuple]:
    """Return list of (gene, partner, score) sorted by score desc."""
    pid = _get_protein_id(cur, gene)
    if not pid:
        return []
    cur.execute(
        """
        SELECT sp1.symbol, sp2.symbol, si.combined_score
        FROM   string_interactions si
        JOIN   string_proteins sp1 ON si.protein_id_1 = sp1.protein_id
        JOIN   string_proteins sp2 ON si.protein_id_2 = sp2.protein_id
        WHERE  (si.protein_id_1 = ? OR si.protein_id_2 = ?)
          AND  si.combined_score >= ?
        ORDER  BY si.combined_score DESC
        LIMIT  ?
        """,
        (pid, pid, min_score, limit),
    )
    results = []
    for s1, s2, score in cur.fetchall():
        partner = s2 if s1 == gene else s1
        results.append((gene, partner, score))
    return results


# ── Colour palette (cycles for >5 seed genes) ─────────────────────────────────
_SEED_COLOURS  = ["#c0392b", "#1a5276", "#6c3483", "#1a6634", "#7d6608"]
_SHARED_COLOUR = "#ff7f00"   # Yellow/orange for shared interactors (2+ genes)
_SINGLE_COLOUR = "#95C8A0"   # Lighter grey-green for single-gene interactors (fallback)


def _node_colour(
    symbol: str,
    seeds: List[str],
    proteins_to_show: Set[str],
    protein_to_query_genes: Dict[str, Set[str]],
    is_fallback_mode: bool,
) -> tuple[str, str]:
    """Return (fill_colour, group_label)."""
    if symbol in seeds:
        idx = seeds.index(symbol)
        return _SEED_COLOURS[idx % len(_SEED_COLOURS)], "seed"

    connected_genes = protein_to_query_genes.get(symbol, set())
    if len(connected_genes) >= 2:
        return _SHARED_COLOUR, "shared"

    if is_fallback_mode:
        return _SINGLE_COLOUR, "single_gene_interactor"

    return "#cccccc", "unknown"


# ── Score helpers ──────────────────────────────────────────────────────────────

def _normalize_score(raw: float) -> float:
    """Convert STRING score to 0–1 scale regardless of input scale."""
    if raw is None:
        return 0.0
    raw = float(raw)
    return raw / 1000.0 if raw > 1.0 else raw


def _to_string_scale(score_01: float) -> int:
    """Convert a 0–1 confidence score to STRING 0–1000 display scale."""
    return round(score_01 * 1000)


# ── Core HTML builder ──────────────────────────────────────────────────────────

def build_network_html_string(
    genes: list[str],
    *,
    min_score: int = 700,
    top_n: int = 25,
    db_path: str | Path | None = None,
    state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build an interactive PPI network and return the HTML as a string.

    Implements 3-layer filtering:
    - Layer 1: Always show direct gene-gene interactions
    - Layer 2: Show proteins shared by 2+ query genes, top N by score
    - Layer 3: Fallback to top 10 per gene if no shared proteins exist

    Parameters
    ----------
    genes       : list of HGNC gene symbols, e.g. ["LMNA", "LMNB1"]
    min_score   : minimum STRING combined_score in 0–1000 scale (default 700)
    top_n       : max interaction partners per seed gene (default 25)
    db_path     : path to gene_database.sqlite (auto-detected if None)
    state       : pipeline state dict containing network_overlap_analysis and all_gene_data

    Returns
    -------
    HTML string ready to pass to st.components.v1.html() or write to a file.
    """
    db = Path(db_path) if db_path else _DB

    query_genes     = [g.upper() for g in genes]
    query_genes_set = set(query_genes)

    # FIX: convert min_score (0–1000 STRING scale) to 0–1 for comparison with
    # normalized scores stored in state.  Database path already uses raw ints.
    min_confidence = min_score / 1000.0  # e.g. 700 → 0.70

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1: Direct gene-gene interactions from state
    # ══════════════════════════════════════════════════════════════════════════
    direct_interactions: List[Dict] = []
    direct_edges_raw: List[tuple]   = []   # may contain both A→B and B→A

    if state:
        network_overlap     = state.get("network_overlap_analysis", {})
        direct_interactions = network_overlap.get("direct_interactions", [])

        for interaction in direct_interactions:
            if isinstance(interaction, dict):
                gene_a = (interaction.get("gene_a") or interaction.get("gene1") or interaction.get("from") or "")
                gene_b = (interaction.get("gene_b") or interaction.get("gene2") or interaction.get("to")   or "")
                score  = interaction.get("score") or interaction.get("combined_score") or 0
            elif isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                gene_a, gene_b = interaction[0], interaction[1]
                score = interaction[2] if len(interaction) > 2 else 0
            else:
                continue

            if gene_a and gene_b and _normalize_score(score) >= 0.4:
                # Normalize score to 0–1 regardless of how it was stored
                direct_edges_raw.append((gene_a.upper(), gene_b.upper(), _normalize_score(score)))

    # FIX: deduplicate bidirectional pairs (A→B and B→A are one interaction)
    seen_direct: Set[frozenset] = set()
    direct_edges: List[tuple]   = []
    for gene_a, gene_b, score in direct_edges_raw:
        key = frozenset([gene_a, gene_b])
        if key not in seen_direct:
            seen_direct.add(key)
            direct_edges.append((gene_a, gene_b, score))

    n_direct = len(direct_edges)   # FIX: count after dedup, not before

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 2 & 3: Protein interaction data from state or database
    # ══════════════════════════════════════════════════════════════════════════
    protein_to_query_genes: Dict[str, Set[str]] = defaultdict(set)
    protein_best_score:     Dict[str, float]    = defaultdict(float)
    gene_interactions:      Dict[str, List[Dict]] = {}

    if state and state.get("all_gene_data"):
        all_gene_data    = state.get("all_gene_data", {})
        gene_data_lookup = {k.upper(): v for k, v in all_gene_data.items()}

        for gene in query_genes:
            gene_data    = gene_data_lookup.get(gene, {})
            interactions = gene_data.get("interactions", [])
            gene_interactions[gene] = interactions

            for interaction in interactions:
                partner = (
                    interaction.get("partner_symbol") or
                    interaction.get("partner") or
                    interaction.get("protein_b") or
                    interaction.get("gene_b") or
                    interaction.get("to") or
                    interaction.get("interactor")
                )
                raw_score = (
                    interaction.get("score") or
                    interaction.get("combined_score") or
                    interaction.get("string_score") or
                    0
                )
                # FIX: normalize to 0–1 before comparing with min_confidence
                score = _normalize_score(raw_score)

                if partner and partner.upper() not in query_genes_set:
                    partner_upper = partner.upper()
                    # FIX: compare normalized score against normalized threshold
                    meets_threshold = score >= min_confidence
                    if meets_threshold:
                        protein_to_query_genes[partner_upper].add(gene)
                        protein_best_score[partner_upper] = max(
                            protein_best_score[partner_upper], score
                        )

    else:
        # Database path — scores are already in 0–1000 range
        if not db.exists():
            raise FileNotFoundError(f"Database not found: {db}")

        conn = sqlite3.connect(db)
        cur  = conn.cursor()

        for gene in query_genes:
            edges = _fetch_interactions(cur, gene, min_score, top_n * 2)
            interactions = [{"partner": e[1], "score": e[2]} for e in edges]
            gene_interactions[gene] = interactions

            for _, partner, raw_score in edges:
                if partner not in query_genes_set:
                    score = _normalize_score(raw_score)
                    protein_to_query_genes[partner].add(gene)
                    protein_best_score[partner] = max(protein_best_score[partner], score)

        conn.close()

    # ── Layer 2: proteins shared by 2+ query genes ────────────────────────────
    shared_proteins = {
        p: genes_set
        for p, genes_set in protein_to_query_genes.items()
        if len(genes_set) >= 2
    }

    n_seeds = len(query_genes)
    limit_n = 20  # Top 20 shared interactors (yellow nodes)

    proteins_to_show: Set[str] = set()
    is_fallback_mode            = False

    if shared_proteins:
        sorted_shared    = sorted(
            shared_proteins.keys(),
            key=lambda p: protein_best_score[p],
            reverse=True,
        )[:limit_n]
        proteins_to_show = set(sorted_shared)
        filter_description = f"top {len(proteins_to_show)} shared interactors"

    else:
        # ── Layer 3 Fallback: top 10 per gene ────────────────────────────────
        is_fallback_mode = True

        for gene in query_genes:
            interactions = gene_interactions.get(gene, [])

            sorted_interactions = sorted(
                interactions,
                key=lambda x: _normalize_score(
                    x.get("score") or x.get("combined_score") or 0
                ),
                reverse=True,
            )[:10]

            for interaction in sorted_interactions:
                partner = (
                    interaction.get("partner_symbol") or
                    interaction.get("partner") or
                    interaction.get("protein_b") or
                    interaction.get("gene_b") or
                    interaction.get("to") or
                    interaction.get("interactor")
                )
                if partner:
                    partner_upper = partner.upper()
                    if partner_upper not in query_genes_set:
                        proteins_to_show.add(partner_upper)
                        protein_to_query_genes[partner_upper].add(gene)
                        score = _normalize_score(
                            interaction.get("score") or interaction.get("combined_score") or 0
                        )
                        protein_best_score[partner_upper] = max(
                            protein_best_score[partner_upper], score
                        )

        filter_description = "no shared interactors · top 10 per gene"

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD NODES AND EDGES
    # ══════════════════════════════════════════════════════════════════════════
    nodes_dict: Dict[str, dict] = {}
    edges_list: List[dict]      = []
    seen_edges: Set[tuple]      = set()

    def _add_node(symbol: str):
        if symbol in nodes_dict:
            return
        colour, group = _node_colour(
            symbol, query_genes, proteins_to_show, protein_to_query_genes, is_fallback_mode
        )
        is_seed   = symbol in query_genes_set
        node_data = {
            "id":    symbol,
            "label": symbol,
            "color": colour,
            "size":  40 if is_seed else (22 if group == "shared" else 16),
            "group": group,
        }
        if is_seed:
            node_data["font"] = {"size": 14, "color": "#1a1a1a", "bold": True}
        nodes_dict[symbol] = node_data

    for gene in query_genes:
        _add_node(gene)

    # ── Layer 1: direct gene-gene edges ───────────────────────────────────────
    for gene_a, gene_b, score in direct_edges:
        _add_node(gene_a)
        _add_node(gene_b)
        key = tuple(sorted([gene_a, gene_b]))
        if key not in seen_edges:
            seen_edges.add(key)
            # FIX: display score in STRING scale for the tooltip (×1000), keep
            #      edge "value" as normalized float so vis.js scaling works correctly
            string_score = _to_string_scale(score)
            edges_list.append({
                "from":   gene_a,
                "to":     gene_b,
                "value":  score,           # 0–1 for vis.js width scaling
                "title":  f"<b>{gene_a} ↔ {gene_b}</b><br>Direct interaction · STRING score: {string_score}",
                "color":  "#E74C3C",
                "width":  4,
                "dashes": False,
            })

    # ── Layer 2/3: edges for proteins_to_show ─────────────────────────────────
    for protein in proteins_to_show:
        _add_node(protein)
        connected_genes = protein_to_query_genes.get(protein, set())
        for gene in connected_genes:
            if gene in query_genes_set:
                key = tuple(sorted([gene, protein]))
                if key not in seen_edges:
                    seen_edges.add(key)
                    score        = protein_best_score.get(protein, min_confidence)
                    string_score = _to_string_scale(score)
                    is_shared    = len(connected_genes) >= 2
                    edges_list.append({
                        "from":  gene,
                        "to":    protein,
                        "value": score,    # 0–1 for vis.js width scaling
                        "title": f"<b>{gene} ↔ {protein}</b><br>STRING score: {string_score}",
                        "color": _SHARED_COLOUR if is_shared else _SINGLE_COLOUR,
                    })


    # ══════════════════════════════════════════════════════════════════════════
    # LEGEND AND STATS
    # ══════════════════════════════════════════════════════════════════════════
    legend_items = ""

    # Query gene
    legend_items += (
        '<div class="legend-item">'
        '<div class="dot" style="background:#1f4e8c;"></div> '
        'Query gene</div>\n'
    )

    # Shared interactors
    n_shared = len([p for p in proteins_to_show if len(protein_to_query_genes.get(p, set())) >= 2])
    if n_shared > 0:
        legend_items += (
            f'<div class="legend-item">'
            f'<div class="dot" style="background:#ff7f00;"></div> '
            f'Shared interactor</div>\n'
        )

    if is_fallback_mode:
        n_single = len([p for p in proteins_to_show if len(protein_to_query_genes.get(p, set())) == 1])
        if n_single > 0:
            legend_items += (
                f'<div class="legend-item">'
                f'<div class="dot" style="background:{_SINGLE_COLOUR};"></div> '
                f'Individual interactors ({n_single})</div>\n'
            )

    shared_with = {
        p: protein_to_query_genes[p]
        for p in proteins_to_show
        if len(protein_to_query_genes.get(p, set())) >= 2
    }
    shared_list_html = "".join(
        f'<div style="padding:3px 0;border-bottom:1px solid #eee;">{p} '
        f'<span style="color:#888;font-size:0.75rem;">({", ".join(sorted(shared_with[p]))})</span></div>'
        for p in sorted(shared_with)
    ) or "<em style='color:#888'>No shared interactors found</em>"

    title_str = " / ".join(query_genes) + " PPI Network"
    subtitle   = (
        f"STRING score ≥ {min_score} · {filter_description} · "
        f"top {top_n} per gene · {n_direct} direct gene interaction{'s' if n_direct != 1 else ''}"
    )

    nodes_json = json.dumps(list(nodes_dict.values()))
    edges_json = json.dumps(edges_list)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title_str}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link  href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Segoe UI", Arial, sans-serif; background: #f0f4f8; }}
  header {{
    background: linear-gradient(135deg, #1a2a4a, #2c5364);
    color: white; padding: 16px 24px;
  }}
  header h1 {{ font-size: 1.25rem; font-weight: 700; }}
  header p  {{ font-size: 0.80rem; opacity: .75; margin-top: 3px; }}
  #main {{ display: flex; height: calc(100vh - 72px); }}
  #network {{ flex: 1; background: #fafbfc; border-right: 1px solid #dde3ea; }}
  #sidebar {{
    width: 270px; padding: 16px 14px; overflow-y: auto;
    background: white; font-size: 0.81rem; color: #333;
  }}
  #sidebar h2 {{ font-size: 0.90rem; font-weight: 700; color: #1a2a4a;
                 margin: 14px 0 8px; border-top: 1px solid #eee; padding-top: 10px; }}
  #sidebar h2:first-child {{ margin-top: 0; border-top: none; padding-top: 0; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }}
  .dot {{ width: 13px; height: 13px; border-radius: 50%; flex-shrink: 0; }}
  #info-box {{
    padding: 10px; background: #f0f4f8; border-radius: 7px;
    border: 1px solid #dde3ea; min-height: 54px;
    font-size: 0.79rem; color: #444; line-height: 1.5;
  }}
  #search {{ width: 100%; padding: 6px 9px; border: 1px solid #ccd;
             border-radius: 6px; font-size: 0.80rem; margin-bottom: 12px; }}
  .stat {{ display: flex; justify-content: space-between; padding: 4px 0;
           border-bottom: 1px solid #eee; }}
  .stat-val {{ font-weight: 700; color: #2c5364; }}
</style>
</head>
<body>
<header>
  <h1>{title_str}</h1>
  <p>{subtitle}</p>
</header>
<div id="main">
  <div id="network"></div>
  <div id="sidebar">
    <input id="search" type="text" placeholder="Search protein…"
           oninput="searchNode(this.value)">

    <h2>Legend</h2>
    {legend_items}
    <div class="legend-item">
      <div style="width:28px;height:3px;background:linear-gradient(90deg,#ddd,#333);
                  margin:0 2px;"></div>
      Edge width = STRING score
    </div>

    <h2>Stats</h2>
    <div class="stat"><span>Query genes</span><span class="stat-val">{len(query_genes)}</span></div>
    <div class="stat"><span>Direct interactions</span><span class="stat-val">{n_direct}</span></div>
    <div class="stat"><span>Total nodes</span><span class="stat-val">{len(nodes_dict)}</span></div>
    <div class="stat"><span>Total edges</span><span class="stat-val">{len(edges_list)}</span></div>
    <div class="stat"><span>Shared interactors</span><span class="stat-val">{len(shared_with)}</span></div>

    <h2>Click Details</h2>
    <div id="info-box">👆 Click a node or edge for details.</div>

    <h2>Shared Interactors ({len(shared_with)})</h2>
    {shared_list_html}
  </div>
</div>
<script>
var nodesData = {nodes_json};
var edgesData = {edges_json};

nodesData.forEach(function(n) {{
  var seeds = {json.dumps(query_genes)};
  var isSeed = seeds.indexOf(n.id) >= 0;
  n.title = "<b>" + n.id + "</b><br>Group: " + n.group.replace(/_/g," ");
  n.font  = isSeed ? {{size:14, color:"#1a1a1a", bold:true}} : {{size:10, color:"#333"}};
  n.borderWidth = isSeed ? 3 : 1.5;
  n.shadow = isSeed;
  if (isSeed) {{ n.size = 40; }}
}});

var net = new vis.Network(
  document.getElementById("network"),
  {{ nodes: new vis.DataSet(nodesData), edges: new vis.DataSet(edgesData) }},
  {{
    physics: {{
      solver: "forceAtlas2Based",
      forceAtlas2Based: {{
        gravitationalConstant: -55,
        centralGravity: 0.005,
        springLength: 120
      }},
      stabilization: {{ iterations: 200 }}
    }},
    edges: {{
      smooth: {{ type: "continuous" }},
      scaling: {{ min: 1, max: 7 }},
      color: {{ inherit: false }},
      hoverWidth: 3
    }},
    nodes: {{ shape: "dot" }},
    interaction: {{ hover: true, tooltipDelay: 80, hideEdgesOnDrag: true }}
  }}
);

net.on("click", function(p) {{
  var box = document.getElementById("info-box");
  if (p.nodes.length) {{
    var n = new vis.DataSet(nodesData).get(p.nodes[0]);
    box.innerHTML = "<b>" + p.nodes[0] + "</b><br>Group: " + n.group.replace(/_/g," ");
  }} else if (p.edges.length) {{
    var e = new vis.DataSet(edgesData).get(p.edges[0]);
    box.innerHTML = e.title || ("<b>" + e.from + " ↔ " + e.to + "</b>");
  }} else {{
    box.innerHTML = "👆 Click a node or edge for details.";
  }}
}});

function searchNode(val) {{
  if (!val) {{ net.unselectAll(); return; }}
  val = val.toUpperCase();
  var found = nodesData
    .filter(function(n) {{ return n.id.toUpperCase().includes(val); }})
    .map(function(n) {{ return n.id; }});
  if (found.length) {{
    net.selectNodes(found);
    net.focus(found[0], {{scale: 1.4, animation: true}});
  }}
}}
</script>
</body>
</html>
"""
    return html


def build_network_html(
    genes: list[str],
    *,
    min_score: int = 700,
    top_n: int = 25,
    output_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    state: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Build an interactive PPI network HTML and write it to *output_dir*.

    Returns
    -------
    Path to the written HTML file.
    """
    html = build_network_html_string(
        genes, min_score=min_score, top_n=top_n, db_path=db_path, state=state
    )

    if output_dir is None:
        output_dir = _ROOT / "results"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "_".join(genes) + f"_PPI_network_{ts}.html"
    out_path = out_dir / filename
    out_path.write_text(html, encoding="utf-8")
    print(f"  ✓  Network saved → {out_path}")
    return out_path


# ── Pipeline helper: read genes from most recent run ──────────────────────────

def genes_from_latest_run() -> list[str]:
    """Return the gene list from the most recent pipeline state.json."""
    runs = sorted(_RUNS.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No pipeline runs found in {_RUNS}")
    state_file = runs[-1] / "state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"No state.json in {runs[-1]}")
    entries = json.loads(state_file.read_text())
    for entry in reversed(entries):
        s     = entry.get("state", {})
        genes = s.get("genes_to_process") or list((s.get("all_gene_data") or {}).keys())
        if genes:
            print(f"  ℹ  Loaded genes from: {runs[-1].name}")
            print(f"     Query: {s.get('user_query','').strip()}")
            print(f"     Genes: {genes}")
            return genes
    raise ValueError("Could not extract gene list from latest pipeline run.")


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an interactive PPI network HTML for a list of genes."
    )
    parser.add_argument("genes", nargs="*", help="Gene symbols, e.g. LMNA LMNB1 TP53")
    parser.add_argument("--latest", action="store_true",
                        help="Auto-load genes from the most recent pipeline run")
    parser.add_argument("--min-score", type=int, default=700,
                        help="Minimum STRING combined_score (default 700)")
    parser.add_argument("--top-n", type=int, default=25,
                        help="Max interaction partners per seed gene (default 25)")
    parser.add_argument("--out", default=None, help="Output directory (default: results/)")
    args = parser.parse_args()

    if args.latest:
        gene_list = genes_from_latest_run()
    elif args.genes:
        gene_list = [g.upper() for g in args.genes]
    else:
        parser.error("Provide gene symbols or use --latest")

    build_network_html(
        gene_list,
        min_score=args.min_score,
        top_n=args.top_n,
        output_dir=args.out,
    )
