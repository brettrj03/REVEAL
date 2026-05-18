"""
Network Analysis page for REVEAL.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from app.pages.executive_summary import render_cross_gene_validation_panel


def _build_partner_dict(gene: str, state: dict) -> dict[str, float]:
    """
    Returns {partner_symbol_upper: normalized_score_0_to_1}
    for all interactions of `gene` in state['all_gene_data'].
    Returns ALL partners regardless of score — filtering happens in caller.
    """
    # Case-insensitive gene lookup
    gene_data_lookup = {k.upper(): v for k, v in (state.get('all_gene_data') or {}).items()}
    profile = gene_data_lookup.get(gene.upper(), {})

    # Try multiple keys for interactions list
    interactions = (
        profile.get('interactions') or
        profile.get('protein_interactions') or
        profile.get('string_interactions') or
        profile.get('ppi') or
        []
    )

    partners = {}
    for inter in interactions:
        # Try all possible field names for partner symbol
        partner = (
            inter.get('partner_symbol') or
            inter.get('partner') or
            inter.get('protein_b') or
            inter.get('gene_b') or
            inter.get('to') or
            inter.get('interactor')
        )

        # Try all possible field names for score
        raw = None
        for key in ('score', 'combined_score', 'string_score', 'confidence', 'weight'):
            val = inter.get(key)
            if val is not None:
                raw = val
                break
        if raw is None:
            raw = 0

        # Normalize score
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = 0.0
        score = raw / 1000.0 if raw > 1.0 else raw

        if partner:
            p = partner.upper()
            partners[p] = max(partners.get(p, 0.0), score)

    return partners


def render_network_analysis(state: Dict[str, Any]):
    """Render the Network Analysis tab."""
    st.markdown("## Network Analysis")

    # ── Interactive PPI Network ───────────────────────────────────────────────
    genes = state.get("genes_to_process") or list(
        (state.get("all_gene_data") or state.get("gene_mapping") or {}).keys()
    )
    if genes:
        try:
            import warnings
            import logging
            import streamlit.components.v1 as components
            from src.utils.visualize_network import build_network_html_string

            with st.spinner("Building interactive network…"):
                net_html = build_network_html_string(
                    genes,
                    min_score=400,
                    top_n=25,
                    state=state,
                )
            st.caption("STRING PPI network · click nodes/edges for details")
            # Suppress the st.components.v1.html deprecation warning — it logs
            # via Python's logging module, not the warnings module
            _st_log = logging.getLogger("streamlit")
            _prev_level = _st_log.level
            _st_log.setLevel(logging.ERROR)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    components.html(net_html, height=560, scrolling=False)
            finally:
                _st_log.setLevel(_prev_level)

        except Exception as e:
            st.warning(f"Could not generate interactive network: {e}")
    else:
        st.info("Run the pipeline to generate the interactive network.")

    # ── Network Statistics (moved here, after network graph) ──────────────────
    network = state.get('network_overlap_analysis')
    if network:
        stats = network.get(
            'network_stats',
            network.get('network_statistics', network.get('statistics', {}))
        )
        if stats:
            st.markdown("**Network Statistics**")
            stat_cols = st.columns(3)

            # 1. Direct Query Gene-Gene Interactions (count of pairs in direct_interactions)
            with stat_cols[0]:
                # For single-gene queries, show "—" since no gene-gene pairs exist
                if len(genes) == 1:
                    st.metric(label="Direct Query Gene-Gene Interactions", value="—")
                else:
                    direct_list = network.get('direct_interactions', [])
                    # Deduplicate bidirectional pairs for accurate count
                    seen = set()
                    for inter in direct_list:
                        if isinstance(inter, dict):
                            a = inter.get('gene_a', inter.get('gene1', ''))
                            b = inter.get('gene_b', inter.get('gene2', ''))
                        elif isinstance(inter, (list, tuple)) and len(inter) >= 2:
                            a, b = inter[0], inter[1]
                        else:
                            continue
                        seen.add(frozenset([a, b]))
                    direct_count = len(seen)
                    st.metric(label="Direct Query Gene-Gene Interactions", value=str(direct_count))

            # 2. Shared protein partners
            with stat_cols[1]:
                val2 = stats.get('shared_partners') or stats.get('shared_partner_count')
                display2 = f"{val2:.3f}" if isinstance(val2, float) else (str(val2) if val2 is not None else "—")
                st.metric(label="Shared protein partners", value=display2)

            # 3. Total interactions
            with stat_cols[2]:
                val3 = stats.get('total_interactions')
                display3 = f"{val3:.3f}" if isinstance(val3, float) else (str(val3) if val3 is not None else "—")
                st.metric(label="Total interactions", value=display3)

    st.markdown("---")

    # ── Protein Interactions Section ──────────────────────────────────────────
    st.markdown("### Protein Interactions")

    if network:
        # ── Direct Gene-Gene Interactions Table ───────────────────────────────
        direct = network.get('direct_interactions', [])
        if direct:
            st.subheader("Direct Gene-Gene Interactions for query genes")

            MIN_SCORE = 0.4

            # Deduplicate bidirectional pairs
            seen_pairs    = set()
            unique_direct = []
            for interaction in direct:
                if isinstance(interaction, dict):
                    gene_a = interaction.get('gene_a', interaction.get('gene1', '?'))
                    gene_b = interaction.get('gene_b', interaction.get('gene2', '?'))
                elif isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                    gene_a, gene_b = interaction[0], interaction[1]
                else:
                    continue
                pair = frozenset([gene_a, gene_b])
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    unique_direct.append(interaction)

            # Filter out zero/low-score interactions
            unique_direct = [i for i in unique_direct if (i.get('score') if isinstance(i, dict) else (i[2] if len(i) > 2 else 0)) >= MIN_SCORE]

            interaction_rows = []
            for interaction in unique_direct:
                if isinstance(interaction, dict):
                    gene_a = interaction.get('gene_a', interaction.get('gene1', '?'))
                    gene_b = interaction.get('gene_b', interaction.get('gene2', '?'))
                    score  = interaction.get('score')
                    # FIX: :.3f so 0.794 stays 0.794, not rounded to 1
                    score_display = f"{score:.3f}" if score is not None else "—"
                    interaction_rows.append({"Gene A": gene_a, "Gene B": gene_b, "Score": score_display})
                elif isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                    score = interaction[2] if len(interaction) > 2 else None
                    score_display = f"{score:.3f}" if score is not None else "—"
                    interaction_rows.append({"Gene A": interaction[0], "Gene B": interaction[1], "Score": score_display})

            if interaction_rows:
                st.dataframe(
                    pd.DataFrame(interaction_rows),
                    hide_index=True,
                    width='stretch',
                    column_config={
                        "Gene A": st.column_config.TextColumn(width="small"),
                        "Gene B": st.column_config.TextColumn(width="small"),
                        "Score":  st.column_config.TextColumn(width="small"),
                    },
                )
            else:
                st.info("No direct interactions detected between query genes.")

        # ── Shared Protein Partners Table ─────────────────────────────────
        # Show proteins that connect to 2+ query genes, ranked by sharing count
        st.divider()
        st.subheader("Shared Protein Partners")

        hub_proteins = network.get('hub_proteins', [])
        query_genes_upper = [g.upper() for g in genes]

        # Build list of shared partners
        shared_partners = []
        for hub in hub_proteins:
            if not isinstance(hub, dict):
                continue
            genes_connected = [g.upper() for g in hub.get('genes', [])]
            # Only include query genes that this hub connects to
            query_genes_connected = [g for g in genes_connected if g in query_genes_upper]

            if len(query_genes_connected) >= 2:
                avg_conf = hub.get('avg_confidence') or 0.0
                # Normalize score if needed
                if avg_conf > 1.0:
                    avg_conf = avg_conf / 1000.0

                shared_partners.append({
                    'protein': hub.get('protein', ''),
                    'genes': sorted(query_genes_connected),
                    'count': len(query_genes_connected),
                    'score': avg_conf
                })

        # Sort by number of sharing genes (descending), then by score (descending)
        shared_partners.sort(key=lambda x: (x['count'], x['score']), reverse=True)

        if not shared_partners:
            st.info("No shared protein partners found connecting 2 or more query genes.")
        else:
            # Helper to render green gene chips
            def _render_gene_chips(genes: list) -> str:
                return ''.join(
                    f'<span style="display:inline-block; background:#dcfce7; color:#166534; '
                    f'font-size:0.7rem; font-weight:600; padding:2px 6px; border-radius:10px; '
                    f'margin-left:4px; font-family:monospace;">{g}</span>'
                    for g in genes
                )

            # Helper to render a partner row with alternating background color
            def _render_partner_row(partner: dict, row_index: int) -> str:
                # Even rows get soft blue, odd rows get white
                bg_color = '#F0F9FF' if row_index % 2 == 0 else '#ffffff'
                gene_chips = _render_gene_chips(partner['genes'])
                return (
                    f'<div style="display:flex; align-items:center; '
                    f'padding:6px 12px; border-bottom:1px solid #f3f4f6; font-size:0.85rem; '
                    f'background:{bg_color}; gap:16px;">'
                    f'<span style="color:#374151; width:28%; flex-shrink:0;">{partner["protein"]}</span>'
                    f'<span style="flex:1; text-align:center;">{gene_chips}</span></div>'
                )

            # Render header row
            header_html = (
                '<div style="display:flex; align-items:center; '
                'padding:8px 12px; background:#f0f4f8; font-size:0.75rem; font-weight:700; '
                'text-transform:uppercase; letter-spacing:0.05em; color:#4b5563; '
                'border-bottom:2px solid #e5e7eb; gap:16px;">'
                '<span style="width:28%; flex-shrink:0;">Shared Partner</span>'
                '<span style="flex:1; text-align:center;">Query Genes</span></div>'
            )

            # Render all data rows in scrollable container with alternating colors
            all_rows_html = ''
            for idx, partner in enumerate(shared_partners):
                all_rows_html += _render_partner_row(partner, idx)

            # Display header + scrollable container
            st.markdown(
                f'<div style="background:#fff; border:1px solid #f3f4f6; border-radius:6px;">'
                f'{header_html}'
                f'<div style="overflow-y:auto; max-height:400px;">{all_rows_html}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Search Protein Partners (only show for 2+ genes) ─────────────
        # Single-gene queries cannot have shared partners between genes
        if len(genes) >= 2:
            st.markdown("### Search Protein Partners")
            all_gene_data = state.get('all_gene_data')
            if not all_gene_data:
                st.info("Run the pipeline to use the shared partner explorer.")
            else:
                # Use only query genes for dropdown options
                query_genes = state.get('genes_to_process') or list(all_gene_data.keys())
                protein_options = [g.upper() for g in query_genes]

                st.caption("Select two genes from the dropdowns below to explore their shared protein interaction partners.")

                # Selectboxes - two columns only
                col1, col2 = st.columns(2)

                with col1:
                    default_a = 0  # First query gene
                    protein_a = st.selectbox(
                        "Protein A",
                        options=protein_options,
                        index=default_a,
                        key="explorer_protein_a"
                    )

                with col2:
                    default_b = 1 if len(protein_options) > 1 else 0
                    protein_b = st.selectbox(
                        "Protein B",
                        options=protein_options,
                        index=default_b,
                        key="explorer_protein_b"
                    )

                # Guard: same protein selected
                if protein_a == protein_b:
                    st.warning("Please select two different proteins to compare.")
                else:
                    # Compute shared partners
                    partners_a = _build_partner_dict(protein_a, state)
                    partners_b = _build_partner_dict(protein_b, state)

                    shared_rows = []

                    # Fallback: if partners_A and partners_B are both empty, use hub_proteins
                    if not partners_a and not partners_b:
                        # fall back to hub_proteins from pre-computed analysis
                        hub_proteins = (state.get('network_overlap_analysis') or {}).get('hub_proteins', [])
                        for hub in hub_proteins:
                            if not isinstance(hub, dict):
                                continue
                            genes_connected = [g.upper() for g in hub.get('genes', [])]
                            if protein_a.upper() in genes_connected and protein_b.upper() in genes_connected:
                                avg = hub.get('avg_confidence') or 0.0
                                if avg > 1.0:
                                    avg = avg / 1000.0
                                if avg >= 0.4:
                                    shared_rows.append({
                                        'Partner': hub.get('protein', ''),
                                        f'Score ({protein_a})': avg,
                                        f'Score ({protein_b})': avg,
                                        '_avg': avg,  # internal for sorting
                                    })
                    else:
                        # Normal intersection flow
                        # Find intersection, excluding A and B themselves
                        shared_keys = set(partners_a.keys()) & set(partners_b.keys())
                        shared_keys -= {protein_a.upper(), protein_b.upper()}

                        # Build rows with avg score filtering (hardcoded threshold 0.4)
                        for partner in shared_keys:
                            score_a = partners_a[partner]
                            score_b = partners_b[partner]
                            avg_score = (score_a + score_b) / 2
                            if avg_score >= 0.4:
                                shared_rows.append({
                                    "Partner": partner,
                                    f"Score ({protein_a})": score_a,
                                    f"Score ({protein_b})": score_b,
                                    "_avg": avg_score,  # internal for sorting
                                })

                    # Sort by average of both scores descending
                    shared_rows.sort(key=lambda x: x["_avg"], reverse=True)

                    # Table or info message (no summary metrics)
                    if not shared_rows:
                        st.info(f"No shared partners found between {protein_a} and {protein_b}.")
                    else:
                        # Build DataFrame with only 3 columns (exclude _avg)
                        df = pd.DataFrame(shared_rows)
                        df = df[["Partner", f"Score ({protein_a})", f"Score ({protein_b})"]]

                        # Search box to filter partners
                        search_term = st.text_input(
                            "Search partners",
                            placeholder="Filter by partner name...",
                            key="partner_search"
                        )
                        if search_term:
                            df = df[df["Partner"].str.contains(search_term, case=False, na=False)]

                        st.dataframe(
                            df,
                            hide_index=True,
                            width='stretch',
                            column_config={
                                "Partner": st.column_config.TextColumn(width="small"),
                                f"Score ({protein_a})": st.column_config.NumberColumn(
                                    format="%.3f", width="small"
                                ),
                                f"Score ({protein_b})": st.column_config.NumberColumn(
                                    format="%.3f", width="small"
                                ),
                            },
                        )

        # Source citation for the tables above
        st.caption("Source: STRING protein interaction database")

        # Interpretation (styled as green AI summary box)
        interp = network.get('interpretation')
        if interp:
            st.markdown("#### Network Interpretation")
            st.caption("*AI-synthesised interpretation of interaction pattern*")
            st.markdown(
                f'<div style="background-color: #d4edda; border-left: 4px solid #28a745; '
                f'padding: 1rem 1rem 1rem 1.5rem; border-radius: 4px; line-height: 1.6;">'
                f'{interp}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No network analysis available. Run the pipeline to generate network data.")

    # ── Validation Results (bottom section) ───────────────────────────────────
    # Validation panel appears in its own section at the bottom, not nested
    # inside the interpretation card
    if network and network.get('interpretation'):
        st.divider()
        st.markdown("### Validation Results")
        render_cross_gene_validation_panel(
            "network_interpretation",
            state,
            display_name="Network Interpretation",
        )
