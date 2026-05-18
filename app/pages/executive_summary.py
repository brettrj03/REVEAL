"""
Executive Summary page for REVEAL.
"""

import streamlit as st
from typing import Dict, Any, Optional
import pandas as pd

from src.validation_config.validation_config import ACCURACY_THRESHOLD
from app.pages.individual_genes import (
    render_lineage_summary_line,
    render_claim_journey_cards,
)


def render_cross_gene_validation_panel(
    section_key: str,
    state: Optional[Dict[str, Any]],
    display_name: str = "Validation",
) -> None:
    """
    Render a validation panel for cross-gene sections.

    Reads from state.cross_gene_validation_results[section_key] and renders
    the same panel structure as per-gene validation on the individual genes page.

    Args:
        section_key: Key in cross_gene_validation_results ("go_interpretation" or "cross_gene_synthesis")
        state: The pipeline state dict
        display_name: Human-readable name for the section
    """
    if not state:
        return

    cross_gene_results = state.get("cross_gene_validation_results", {})
    if not cross_gene_results:
        return

    section_data = cross_gene_results.get(section_key)
    if not section_data:
        return

    # Extract key metrics
    status = section_data.get("status", "")
    if status == "no_claims":
        return  # Skip if no claims to validate

    final_accuracy = section_data.get("final_accuracy")
    initial_accuracy = section_data.get("initial_accuracy")
    iterations = section_data.get("iterations", [])
    claims_summary = section_data.get("claims_summary", {})

    if final_accuracy is None and not iterations:
        return  # No validation data

    # Badge styling helper
    def _badge_style(kind: str) -> tuple:
        palette = {
            "success": ("#edf7f1", "#2d7a4f"),
            "warning": ("#fdf6e3", "#8a6010"),
            "error": ("#fdf0f0", "#a03030"),
            "info": ("#eef4fb", "#2a5a8a"),
        }
        return palette.get(kind, ("#f7f6f2", "#555555"))

    # Determine icon and status
    if final_accuracy is None:
        acc_val = 0.0
        icon = "❓"
        badge_kind = "neutral"
        acc_display = "N/A"
    else:
        acc_val = float(final_accuracy)
        acc_display = f"{acc_val:.0f}%"
        if acc_val >= ACCURACY_THRESHOLD:
            icon = "✅"
            badge_kind = "success"
        elif acc_val >= 70:
            icon = "⚠️"
            badge_kind = "warning"
        else:
            icon = "❌"
            badge_kind = "error"

    bg, fg = _badge_style(badge_kind)

    # Render the validation panel
    with st.expander(
        f"{icon} {display_name} Validation — {acc_display}",
        expanded=(final_accuracy is not None and acc_val < 85),
    ):
        # Progress bar
        st.progress(max(0.0, min(acc_val / 100.0, 1.0)))

        # Show before/after if text was corrected
        original_summary = section_data.get("original_summary", "")
        final_summary = section_data.get("final_summary", "")
        if original_summary and final_summary and original_summary.strip() != final_summary.strip():
            with st.expander("📝 Before → After correction", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Original**")
                    st.caption(original_summary[:500] + ("..." if len(original_summary) > 500 else ""))
                with col_b:
                    st.markdown("**Corrected**")
                    st.caption(final_summary[:500] + ("..." if len(final_summary) > 500 else ""))

        # Lineage summary line (Phase 3) — rendered before the iteration table
        if iterations:
            render_lineage_summary_line(iterations)

        # Iteration history table (HTML, 8 columns — same as individual genes)
        if iterations:
            st.markdown("##### Validation Iterations")
            rows = []
            total_iters = len(iterations)
            for idx, it in enumerate(iterations):
                counts = it.get("counts", {})
                claims = it.get("claims_extracted", 0) or len(it.get("all_claims", []))
                accurate = counts.get("accurate", 0) or len(it.get("accurate_claims", []))
                partial_cnt = counts.get("partial", 0) or len(it.get("partially_accurate_claims", []))
                inaccurate_cnt = counts.get("inaccurate", 0) or len(it.get("inaccurate_claims", []))
                acc_raw = it.get("accuracy_percentage")
                acc_pct = float(acc_raw) if acc_raw is not None else None

                is_final = idx == total_iters - 1
                issues = inaccurate_cnt + partial_cnt
                iter_label = f"Iter {idx + 1}" + (" (final)" if is_final else "")

                acc_kind = (
                    ("success" if acc_pct >= ACCURACY_THRESHOLD else ("warning" if acc_pct >= 70 else "error"))
                    if acc_pct is not None else "neutral"
                )
                acc_bg, acc_fg = _badge_style(acc_kind)

                next_has_changes = False
                if idx < total_iters - 1:
                    next_has_changes = bool(iterations[idx + 1].get("refinement_changes"))
                partial_only = partial_cnt > 0 and inaccurate_cnt == 0

                if next_has_changes:
                    action = ("refined", "warning")
                elif is_final and issues == 0 and acc_pct is not None and acc_pct >= ACCURACY_THRESHOLD:
                    action = ("✓ passed", "success")
                elif is_final and partial_only and acc_pct is not None and acc_pct < ACCURACY_THRESHOLD:
                    action = ("needs review", "warning")
                elif partial_only:
                    action = ("softened", "info")
                elif issues == 0:
                    action = ("✓ passed", "warning")
                else:
                    action = ("needs review", "error")

                action_bg, action_fg = _badge_style(action[1])

                rows.append(
                    f"<tr>"
                    f"<td>{iter_label}</td>"
                    f"<td>{claims}</td>"
                    f"<td>{accurate}</td>"
                    f"<td>{partial_cnt}</td>"
                    f"<td>{inaccurate_cnt}</td>"
                    f'<td style="background:{acc_bg};color:{acc_fg};font-weight:600;">{"N/A" if acc_pct is None else f"{acc_pct:.0f}%"}</td>'
                    f'<td style="background:{action_bg};color:{action_fg};font-weight:600;">{action[0]}</td>'
                    f"</tr>"
                )

            # Final (resolved) row
            _last_it = iterations[-1]
            _lc = _last_it.get("counts", {})
            _l_accurate = _lc.get("accurate", 0) or len(_last_it.get("accurate_claims", []))
            _l_partial = _lc.get("partial", 0) or len(_last_it.get("partially_accurate_claims", []))
            _l_claims = _last_it.get("claims_extracted", 0) or len(_last_it.get("all_claims", []))
            _l_acc_raw = _last_it.get("accuracy_percentage")
            _l_acc_pct = float(_l_acc_raw) if _l_acc_raw is not None else None
            if _l_acc_pct is not None and _l_acc_pct >= ACCURACY_THRESHOLD:
                _r_bg, _r_fg = "#e8f5e9", "#1b5e20"
                rows.append(
                    f'<tr style="background:{_r_bg};color:{_r_fg};font-weight:600;">'
                    f"<td>Final (resolved)</td>"
                    f"<td>{_l_claims}</td>"
                    f"<td>{_l_accurate + _l_partial}</td>"
                    f"<td>0</td><td>0</td>"
                    f'<td style="background:{_r_bg};color:{_r_fg};font-weight:600;">100%</td>'
                    f'<td style="background:{_r_bg};color:{_r_fg};font-weight:600;">✓ all resolved</td>'
                    f"</tr>"
                )

            table_html = (
                '<div style="overflow-x:auto;margin:8px 0;">'
                '<table style="width:100%;border-collapse:collapse;font-size:0.75rem;">'
                "<thead><tr>"
                "<th>Iteration</th><th>Claims</th><th>Accurate</th>"
                "<th>Partial</th><th>Inaccurate</th><th>Accuracy</th><th>Action</th>"
                "</tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table>"
                "<style>table,th,td{border:1px solid #e4e2dd;padding:6px 10px;}"
                "th{background:#f7f6f2;text-transform:uppercase;letter-spacing:0.05em;}"
                "</style></div>"
            )
            st.markdown(table_html, unsafe_allow_html=True)

        # Claim journey (Phase 3)
        _final_iter = iterations[-1] if iterations else {}
        if iterations:
            st.markdown("##### Claim journey")
            render_claim_journey_cards(iterations, section_key)

        # Section footer
        _is_resolved = final_accuracy is not None and float(final_accuracy) >= ACCURACY_THRESHOLD
        if _is_resolved:
            _first_iter_acc_raw = iterations[0].get("accuracy_percentage") if iterations else None
            _first_passed = _first_iter_acc_raw is not None and float(_first_iter_acc_raw) >= ACCURACY_THRESHOLD
            if len(iterations) <= 1 or _first_passed:
                st.caption("✓ All claims passed on first check.")
            else:
                st.caption("✓ All issues resolved during refinement.")
        else:
            _final_all_c = _final_iter.get("all_claims", []) or []
            _n_unresolved = sum(
                1 for c in _final_all_c
                if isinstance(c, dict) and c.get("verdict") in ("partially_accurate", "inaccurate")
            )
            if _n_unresolved > 0:
                st.caption(f"⚠️ {_n_unresolved} claim(s) could not be fully verified.")
            else:
                st.caption("✓ All claims passed on first check.")


def render_executive_summary(state: Dict[str, Any]):
    """Render the Executive Summary tab.

    Sections:
      1. Query bar  — query text + run metadata pills
      2. Stats row  — 4 st.metric cards (genes, papers, shared GO, stages)
      3. Gene table — one row per gene with basic info + validation badge + NCBI attribution
      3b. Cross-Gene Synthesis — AI-generated key findings (green box with bullets)
      3b. Shared GO Terms — 3-column display (CC/BP/MF) with gene pills
      3b. GO Interpretation — AI-generated GO overlap analysis (green box)
      3c. Validation Quality — claims checked / passed / failed
      4. Network    — headline numbers + link to Network Analysis tab
    """
    st.markdown("## Executive Summary")

    # ── 0. Guard: nothing loaded yet ─────────────────────────────────────────
    if not state:
        st.info("No results loaded. Run the pipeline or load a previous run from the sidebar.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # 1. QUERY HEADER
    # ══════════════════════════════════════════════════════════════════════════
    user_query = state.get("user_query", "")
    exp_ctx = state.get("experiment_context", {})

    def _ctx(key: str, default: str = "—") -> str:
        if isinstance(exp_ctx, dict):
            return exp_ctx.get(key) or default
        return getattr(exp_ctx, key, None) or default

    organism    = _ctx("organism", "Human")
    output_mode = state.get("output_mode", "—")
    cell_type   = _ctx("cell_type", "Unspecified")

    st.markdown(
        f'<p style="font-size: 1.1rem; font-weight: 500; color: #374151; margin-bottom: 0.5rem;">'
        f'<strong>Query:</strong> {user_query or "No query loaded"}</p>',
        unsafe_allow_html=True
    )

    context_lines = []
    if cell_type and cell_type not in ("—", "Unspecified"):
        context_lines.append(f"**Cell type:** {cell_type}")
    if organism and organism != "—":
        context_lines.append(f"**Organism:** {organism.title()}")
    if output_mode and output_mode != "—":
        context_lines.append(f"**Mode:** {output_mode.title()}")
    if context_lines:
        st.markdown(" &nbsp;|&nbsp; ".join(context_lines))

    # ══════════════════════════════════════════════════════════════════════════
    # 2. STATS ROW
    # ══════════════════════════════════════════════════════════════════════════
    gene_mapping = state.get("gene_mapping", {})
    genes_found = [
        g for g, info in gene_mapping.items()
        if info and info.get("found", False)
    ]
    genes_missing = [
        g for g, info in gene_mapping.items()
        if not info or not info.get("found", False)
    ]

    total_papers = sum(
        len(papers.get("top_papers", papers.get("paper_candidates", [])))
        for papers in state.get("gene_top_papers", {}).values()
    )

    go_analysis = state.get("go_comparison_analysis") or {}
    shared_terms_list = go_analysis.get("shared_terms", go_analysis.get("shared_go_terms", []))
    shared_go_count = len(shared_terms_list) if isinstance(shared_terms_list, list) else 0

    nodes_executed = len(state.get("nodes_executed", []))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Genes Analysed", len(genes_found),
                  delta=f"{len(genes_missing)} unresolved" if genes_missing else None,
                  delta_color="inverse")
    with col2:
        st.metric("Papers Retrieved", total_papers)
    with col3:
        st.metric("Shared GO Terms", shared_go_count)
    with col4:
        st.metric("Pipeline Stages", nodes_executed)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. GENE TABLE
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown("### Genes")
    st.caption("To see the full gene summary go to the Individual Genes page.")

    gene_profiles   = state.get("gene_profiles", {})
    all_gene_data   = state.get("all_gene_data", {})
    accuracy_scores = state.get("accuracy_scores", {})
    lit_findings    = state.get("literature_findings_summary", {})
    gene_top_papers = state.get("gene_top_papers", {})

    rows = []
    for gene in sorted(genes_found):
        profile = gene_profiles.get(gene, {})
        raw     = all_gene_data.get(gene, {})

        def _get(*keys, default="—"):
            for k in keys:
                v = (profile.get(k) if isinstance(profile, dict) else getattr(profile, k, None)) \
                    or raw.get(k)
                if v:
                    return v
            return default

        chrom = _get("chromosome", "chrom", default="—")
        if chrom and chrom != "—":
            chrom = f"Chr {str(chrom).upper().replace('CHR', '').strip()}"

        biotype = _get("gene_type", "biotype", default="—")
        if biotype and biotype != "—":
            biotype = str(biotype).replace("_", " ").title()

        go_terms = raw.get("go_terms") or (
            profile.get("go_terms") if isinstance(profile, dict)
            else getattr(profile, "go_terms", None)
        ) or []
        go_count = len(go_terms) if isinstance(go_terms, list) else 0

        interactions = raw.get("interactions") or (
            profile.get("interactions") if isinstance(profile, dict)
            else getattr(profile, "interactions", None)
        ) or []
        inter_count = raw.get("total_interaction_count") or len(interactions)

        papers_for_gene = gene_top_papers.get(gene, {})
        paper_count = len(papers_for_gene.get("top_papers", []))

        acc = accuracy_scores.get(gene)
        if acc is not None:
            try:
                acc_val = float(acc)
                acc_str = f"{acc_val:.0f}%" if acc_val > 1 else f"{acc_val * 100:.0f}%"
            except (TypeError, ValueError):
                acc_str = str(acc)
        else:
            acc_str = "—"

        rows.append({
            "Gene":         gene,
            "Chromosome":   chrom,
            "Biotype":      biotype,
            "GO Terms":     go_count,
            "Interactions": inter_count,
            "Papers":       paper_count,
            "Validation":   acc_str,
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "Gene":         st.column_config.TextColumn(width="small"),
                "Chromosome":   st.column_config.TextColumn(width="small"),
                "Biotype":      st.column_config.TextColumn(width="medium"),
                "GO Terms":     st.column_config.NumberColumn(width="small"),
                "Interactions": st.column_config.NumberColumn(width="small", format="%d"),
                "Papers":       st.column_config.NumberColumn(width="small"),
                "Validation":   st.column_config.TextColumn(width="small"),
            },
        )
        st.caption("*Source: NCBI*")
    else:
        st.info("No gene data available. Run the pipeline or load a previous run.")


    # ══════════════════════════════════════════════════════════════════════════
    # 3b. CROSS-GENE ANALYSIS SECTIONS
    # Only shown for multi-gene queries where shared GO terms exist
    # ══════════════════════════════════════════════════════════════════════════
    n_genes_found = len(genes_found)

    if n_genes_found >= 2 and shared_terms_list:
        # Group shared terms by namespace
        # Note: analyze_go_comparison.py stores the namespace under the key "category",
        # not "namespace". We check both keys so the filter works regardless of which
        # key is present (older saved states may use "namespace").
        def _term_ns(t):
            return t.get("namespace") or t.get("category", "")

        bp_terms = [t for t in shared_terms_list if isinstance(t, dict) and _term_ns(t) == "biological_process"]
        mf_terms = [t for t in shared_terms_list if isinstance(t, dict) and _term_ns(t) == "molecular_function"]
        cc_terms = [t for t in shared_terms_list if isinstance(t, dict) and _term_ns(t) == "cellular_component"]

        # Pull key_findings from synthesis if available
        synthesis_raw = state.get("synthesis")
        key_findings = []
        if synthesis_raw is not None:
            if isinstance(synthesis_raw, dict):
                key_findings = synthesis_raw.get("key_findings") or []
            else:
                key_findings = getattr(synthesis_raw, "key_findings", None) or []

        # ── Cross-Gene Analysis Synthesis (green box with green dot bullets) ──
        if key_findings:
            st.markdown("### Cross-Gene Analysis Synthesis")
            st.caption("*AI-synthesised overview combining function, interactions, and cross-gene relationships*")

            # Helper to convert markdown bold to HTML strong tags
            def _convert_markdown_bold(text: str) -> str:
                """Convert **text** to <strong>text</strong>"""
                import re
                return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

            # Build bullet list with green dots and bold labels
            bullet_items = []
            for finding in key_findings:
                if not finding or not isinstance(finding, str):
                    continue

                # Convert markdown bold to HTML
                finding = _convert_markdown_bold(finding)

                # Check if finding contains a colon - if so, make label bold
                if ':' in finding:
                    label, text = finding.split(':', 1)
                    formatted = f'<strong>{label.strip()}</strong>:{text}'
                else:
                    formatted = finding

                # Create flex row with green dot bullet
                bullet_items.append(
                    f'<div style="display: flex; align-items: flex-start; margin-bottom: 8px;">'
                    f'<span style="display: inline-block; width: 6px; height: 6px; background-color: #1D9E75; '
                    f'border-radius: 50%; margin-top: 6px; margin-right: 10px; flex-shrink: 0;"></span>'
                    f'<span style="line-height: 1.6;">{formatted}</span>'
                    f'</div>'
                )

            bullet_html = ''.join(bullet_items)
            st.markdown(
                f'<div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 1rem 1rem 1rem 1.5rem; border-radius: 4px;">'
                f'{bullet_html}</div>',
                unsafe_allow_html=True
            )

        # ── Shared GO Terms 3-column display ──
        st.markdown("---")
        st.markdown("### Shared GO Terms")
        st.caption("To see individual gene GO terms, go to the Individual Genes page.")

        col_cc, col_bp, col_mf = st.columns(3)

        # Helper to render a single GO term row with gene pills
        def _render_go_term(term: dict) -> str:
            name = term.get("name") or term.get("term") or "Unknown"
            term_genes = term.get("genes", [])
            gene_pills = ''.join(
                f'<span style="display:inline-block; background:#dcfce7; color:#166534; '
                f'font-size:0.7rem; font-weight:600; padding:2px 6px; border-radius:10px; '
                f'margin-left:4px; font-family:monospace;">{g}</span>'
                for g in term_genes
            )
            return (
                f'<div style="display:flex; justify-content:space-between; align-items:center; '
                f'padding:6px 0; border-bottom:1px solid #f3f4f6; font-size:0.85rem;">'
                f'<span style="color:#374151;">{name}</span>'
                f'<span>{gene_pills}</span></div>'
            )

        # Cellular Component column (purple)
        with col_cc:
            cc_count = len(cc_terms)
            st.markdown(
                f'<div style="background:#ede9fe; color:#6b21a8; font-size:0.75rem; font-weight:700; '
                f'text-transform:uppercase; letter-spacing:0.08em; padding:8px 12px; '
                f'border-radius:6px 6px 0 0; margin-bottom:0;">'
                f'Cellular Component <span style="font-weight:400;">({cc_count})</span></div>',
                unsafe_allow_html=True
            )
            if cc_terms:
                # Render all terms in scrollable container
                all_terms_html = ''.join(_render_go_term(t) for t in cc_terms)
                st.markdown(
                    f'<div style="background:#fff; padding:4px 12px; overflow-y:auto; max-height:400px;">{all_terms_html}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="background:#fff; padding:12px; color:#9ca3af; font-style:italic; font-size:0.85rem;">No shared cellular components</div>',
                    unsafe_allow_html=True
                )

        # Biological Process column (blue)
        with col_bp:
            bp_count = len(bp_terms)
            st.markdown(
                f'<div style="background:#dbeafe; color:#1e40af; font-size:0.75rem; font-weight:700; '
                f'text-transform:uppercase; letter-spacing:0.08em; padding:8px 12px; '
                f'border-radius:6px 6px 0 0; margin-bottom:0;">'
                f'Biological Process <span style="font-weight:400;">({bp_count})</span></div>',
                unsafe_allow_html=True
            )
            if bp_terms:
                # Render all terms in scrollable container
                all_terms_html = ''.join(_render_go_term(t) for t in bp_terms)
                st.markdown(
                    f'<div style="background:#fff; padding:4px 12px; overflow-y:auto; max-height:400px;">{all_terms_html}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="background:#fff; padding:12px; color:#9ca3af; font-style:italic; font-size:0.85rem;">No shared biological processes</div>',
                    unsafe_allow_html=True
                )

        # Molecular Function column (amber)
        with col_mf:
            mf_count = len(mf_terms)
            st.markdown(
                f'<div style="background:#fef3c7; color:#92400e; font-size:0.75rem; font-weight:700; '
                f'text-transform:uppercase; letter-spacing:0.08em; padding:8px 12px; '
                f'border-radius:6px 6px 0 0; margin-bottom:0;">'
                f'Molecular Function <span style="font-weight:400;">({mf_count})</span></div>',
                unsafe_allow_html=True
            )
            if mf_terms:
                # Render all terms in scrollable container
                all_terms_html = ''.join(_render_go_term(t) for t in mf_terms)
                st.markdown(
                    f'<div style="background:#fff; padding:4px 12px; overflow-y:auto; max-height:400px;">{all_terms_html}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="background:#fff; padding:12px; color:#9ca3af; font-style:italic; font-size:0.85rem;">No shared molecular functions</div>',
                    unsafe_allow_html=True
                )

        st.caption("*Source: Gene Ontology (GO)*")
        st.markdown("---")

        # ── GO Term Overlap Interpretation (green box) ──
        go_interpretation = go_analysis.get("interpretation")
        if go_interpretation and isinstance(go_interpretation, str) and go_interpretation.strip():
            st.markdown("### GO Term Overlap Interpretation")
            st.caption("*AI-generated analysis of shared functional annotations across the gene set*")
            st.markdown(
                f'<div style="background-color: #d4edda; border-left: 4px solid #28a745; '
                f'padding: 1rem 1rem 1rem 1.5rem; border-radius: 4px; line-height: 1.6;">'
                f'{go_interpretation}</div>',
                unsafe_allow_html=True
            )

    # ══════════════════════════════════════════════════════════════════════════
    # 3c. VALIDATION QUALITY OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    claim_verdicts  = state.get("claim_verdicts", {})
    validation_results = state.get("validation_results", {})

    # Count totals across all genes
    total_claims   = 0
    passed_claims  = 0
    failed_claims  = 0
    genes_with_low_conf = []

    for gene, verdicts in claim_verdicts.items():
        if not isinstance(verdicts, list):
            continue
        for v in verdicts:
            total_claims += 1
            verdict_str = ""
            if isinstance(v, dict):
                verdict_str = str(v.get("verdict", "")).upper()
            else:
                verdict_str = str(getattr(v, "verdict", "")).upper()
            if verdict_str in ("SUPPORTED", "PASS", "TRUE", "CORRECT"):
                passed_claims += 1
            elif verdict_str in ("UNSUPPORTED", "FAIL", "FALSE", "INCORRECT", "CONTRADICTED"):
                failed_claims += 1

        # Flag genes whose accuracy score is below 100%
        acc = accuracy_scores.get(gene)
        if acc is not None:
            try:
                acc_val = float(acc)
                if acc_val < 100.0 if acc_val > 1 else acc_val < 1.0:
                    genes_with_low_conf.append(gene)
            except (TypeError, ValueError):
                pass

    if total_claims > 0:
        st.markdown("---")

        pass_pct = round(passed_claims / total_claims * 100)
        fail_pct  = round(failed_claims / total_claims * 100)

        if pass_pct == 100:
            badge_cls  = "val-badge-green"
            badge_icon = "✓"
            badge_text = "All claims verified"
        elif pass_pct >= 80:
            badge_cls  = "val-badge-amber"
            badge_icon = "⚠"
            badge_text = f"{100 - pass_pct}% of claims flagged"
        else:
            badge_cls  = "val-badge-red"
            badge_icon = "✗"
            badge_text = f"{100 - pass_pct}% of claims failed"

        low_conf_html = ""
        if genes_with_low_conf:
            gene_tags = "".join(
                f'<span class="val-gene-flag">{g}</span>'
                for g in sorted(genes_with_low_conf)
            )
            low_conf_html = (
                f'<div class="val-row" style="margin-top:0.5rem;">'
                f'<span class="val-row-label">Genes needing attention:</span> {gene_tags}'
                f'</div>'
            )

        st.markdown("### Validation Quality")
        st.markdown(
            f"""
            <div class="val-overview-box">
                <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                    <span class="val-badge {badge_cls}">{badge_icon} {badge_text}</span>
                    <div class="val-stat"><span class="val-stat-num">{total_claims}</span><span class="val-stat-label">claims checked</span></div>
                    <div class="val-stat"><span class="val-stat-num val-pass">{pass_pct}%</span><span class="val-stat-label">passed</span></div>
                    <div class="val-stat"><span class="val-stat-num val-fail">{fail_pct}%</span><span class="val-stat-label">failed/flagged</span></div>
                </div>
                {low_conf_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    # ══════════════════════════════════════════════════════════════════════════
    # 4. PROTEIN INTERACTION NETWORK SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()

    network_data = state.get("network_overlap_analysis")
    if not network_data:
        st.info("Run the pipeline to discover shared interaction partners.")
        return

    st.subheader("Protein Interaction Network")

    shared_raw = network_data.get("shared_partners")
    if isinstance(shared_raw, list):
        shared_value = len(shared_raw)
    elif isinstance(shared_raw, (int, float)):
        shared_value = int(shared_raw)
    else:
        shared_value = len(network_data.get("hub_proteins", []))

    direct_raw = network_data.get("direct_connections")
    if isinstance(direct_raw, (int, float)):
        direct_connections = int(direct_raw)
    else:
        direct_connections = len(network_data.get("direct_interactions", []))

    stats_block = (
        network_data.get("network_stats")
        or network_data.get("network_statistics")
        or network_data.get("statistics", {})
    )
    stats_shared_value = None
    if isinstance(stats_block, dict):
        stats_shared_value = (
            stats_block.get("shared_partners")
            or stats_block.get("shared_partner_count")
            or stats_block.get("shared_partners_count")
        )
    if not shared_value and isinstance(stats_shared_value, (int, float)):
        shared_value = int(stats_shared_value)

    total_interactions = 0
    if isinstance(stats_block, dict):
        total_interactions = (
            stats_block.get("total_interactions")
            or stats_block.get("all_interactions")
            or stats_block.get("total")
            or 0
        )
    if not total_interactions:
        total_interactions = (
            network_data.get("total_interactions")
            or network_data.get("all_interactions")
            or 0
        )
    try:
        total_interactions = int(total_interactions)
    except (TypeError, ValueError):
        total_interactions = 0

    other_value = max(total_interactions - direct_connections - shared_value, 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Shared protein partners",
            shared_value,
            help="No common interactors found between query genes",
        )
    with col2:
        st.metric(
            "Direct connection",
            direct_connections,
            help="Query genes interact directly with each other",
        )
    with col3:
        st.metric(
            "Other protein interactions",
            other_value,
            help="Total interactions across all query genes",
        )

    st.caption("To see the full protein interactions, go to the Network Analysis page.")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. CROSS-GENE VALIDATION RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    cross_gene_val = state.get("cross_gene_validation_results", {})
    if cross_gene_val:
        st.divider()
        st.markdown("## Validation Results")

        render_cross_gene_validation_panel(
            "cross_gene_synthesis",
            state,
            display_name="Cross-Gene Synthesis",
        )
        render_cross_gene_validation_panel(
            "go_interpretation",
            state,
            display_name="GO Interpretation",
        )
