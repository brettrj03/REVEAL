"""
Individual Genes page for REVEAL.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from collections import defaultdict

from src.validation_config.validation_config import ACCURACY_THRESHOLD
from src.validation.unified_validator import SECTION_DISPLAY_NAMES
from app.components.shared import (
    render_source,
    render_ai_interpretation,
    render_evidence_code_guide,
)


_TIER_STATUS_ICON = {
    "executed": "✅",
    "threshold_met_pool": "✅",
    "threshold_met_tier": "✅",
    "executed_zero": "⚠️",
    "skipped": "⏭️",
    "not_generated": "—",
    "error": "❌",
}

_TIER_STATUS_LABEL = {
    "executed": "ran",
    "threshold_met_pool": "ran",
    "threshold_met_tier": "ran",
    "executed_zero": "0 results",
    "skipped": "skipped",
    "not_generated": "not applicable",
    "error": "error",
}

_TIER_SUCCESS_STATUSES = {"executed", "threshold_met_pool", "threshold_met_tier"}



def render_gene_identity(gene_data: Dict[str, Any], selected_gene: str):
    """Render the Gene Identity factual section."""
    st.markdown("---")

    gene_data = gene_data or {}
    description_raw = gene_data.get('description', '')
    full_name_value = description_raw.split('[')[0].strip() if description_raw else ''
    if not full_name_value:
        fallback_name = gene_data.get('gene_name') or gene_data.get('long_description') or ''
        full_name_value = fallback_name.strip()
    full_name_display = full_name_value or 'N/A'

    header_text = "### Gene Identity"
    if full_name_value:
        header_text = f"### Gene Identity - **{full_name_value}**"
    st.markdown(header_text)

    if gene_data:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Official Symbol:** {gene_data.get('symbol', selected_gene)}")

            st.markdown(f"**Full Name:** {full_name_display}")
            st.markdown(f"**Ensembl ID:** {gene_data.get('gene_id', 'N/A')}")

        with col2:
            chrom = gene_data.get('chromosome', gene_data.get('chrom', 'N/A'))
            start = gene_data.get('start', '')
            end = gene_data.get('end', '')

            def _format_chromosome(chrom_value: str) -> str:
                if not chrom_value:
                    return 'N/A'
                chrom_str = str(chrom_value).strip()
                normalized = chrom_str.upper()
                if normalized.startswith('CHR'):
                    normalized = normalized[3:].strip()
                if normalized in {'X', 'Y', 'M'} or normalized.isdigit():
                    return f"Chr {normalized}"
                return chrom_str or 'N/A'

            chrom_display = _format_chromosome(str(chrom))
            if start and end:
                try:
                    start_val = int(start)
                    end_val = int(end)
                except (TypeError, ValueError):
                    location = chrom_display
                else:
                    location = f"{chrom_display} | {start_val:,} - {end_val:,}"
            else:
                location = chrom_display

            gene_type_raw = gene_data.get('gene_type', 'N/A')

            def _format_gene_type(gene_type_value: Any) -> str:
                if not gene_type_value:
                    return 'N/A'
                gene_type_str = str(gene_type_value)
                if '_' in gene_type_str:
                    return gene_type_str.replace('_', ' ').title()
                return gene_type_str

            strand_value = str(gene_data.get('strand', 'N/A'))
            if strand_value == '+':
                strand_display = "+ (positive)"
            elif strand_value == '-':
                strand_display = "- (negative)"
            else:
                strand_display = strand_value or 'N/A'

            st.markdown(f"**Chromosome Location:** {location}")
            st.markdown(f"**Gene Type:** {_format_gene_type(gene_type_raw)}")
            st.markdown(f"**Strand:** {strand_display}")

        aliases = gene_data.get('aliases') or []
        alias_values = []
        if isinstance(aliases, list):
            for entry in aliases:
                if isinstance(entry, dict):
                    alias = entry.get('symbol_alias') or entry.get('alias') or entry.get('symbol')
                else:
                    alias = entry
                if alias:
                    alias_values.append(alias.strip())

        unique_aliases = sorted({alias for alias in alias_values if alias}, key=lambda x: x.lower())

        st.markdown("#### Aliases")
        if unique_aliases:
            alias_text = " · ".join(unique_aliases)
            st.markdown(alias_text)
        else:
            st.caption("No aliases available in the local HGNC/NCBI records.")

        render_source(f"Source: {gene_data.get('source', 'GENCODE')}")
    else:
        st.warning("No basic gene information available")


def get_evidence_priority(code: str) -> int:
    """Sort evidence codes by reliability (best first)."""
    priority = {
        'EXP': 1,   # Experimental
        'IDA': 2,   # Direct Assay
        'IPI': 3,   # Physical Interaction
        'IMP': 4,   # Mutant Phenotype
        'IGI': 5,   # Genetic Interaction
        'IEP': 6,   # Expression Pattern
        'HDA': 7,   # High-throughput Direct Assay
        'HMP': 8,   # High-throughput Mutant Phenotype
        'HGI': 9,   # High-throughput Genetic Interaction
        'HEP': 10,  # High-throughput Expression Pattern
        'TAS': 11,  # Traceable Author Statement
        'NAS': 12,  # Non-traceable Author Statement
        'IC': 13,   # Inferred by Curator
        'IBA': 14,  # Biological Ancestor
        'IBD': 15,  # Biological Descendant
        'IKR': 16,  # Key Residues
        'IRD': 17,  # Rapid Divergence
        'ISS': 18,  # Sequence/Structural Similarity
        'ISO': 19,  # Sequence Orthology
        'ISA': 20,  # Sequence Alignment
        'ISM': 21,  # Sequence Model
        'IGC': 22,  # Genomic Context
        'RCA': 23,  # Reviewed Computational Analysis
        'IEA': 24,  # Electronic Annotation (lowest confidence)
        'ND': 25,   # No Data
    }
    return priority.get(code, 99)


def consolidate_go_terms(go_terms_list: list) -> list:
    """Group GO terms by ID and consolidate evidence codes.

    Sorting priority:
    1. Number of evidence codes (descending) - more evidence = higher priority
    2. Best evidence code quality (ascending) - stronger evidence = higher priority
    """
    go_dict = defaultdict(lambda: {
        'name': '',
        'namespace': '',
        'definition': '',
        'evidence_codes': set()
    })

    for term in go_terms_list:
        go_id = term.get('go_id', '')
        if not go_id:
            continue
        go_dict[go_id]['name'] = term.get('name', 'Unknown')
        go_dict[go_id]['go_id'] = go_id
        go_dict[go_id]['namespace'] = term.get('namespace', term.get('category', ''))
        definition_value = term.get('definition')
        if definition_value and not go_dict[go_id]['definition']:
            go_dict[go_id]['definition'] = definition_value
        evidence = term.get('evidence', '')
        if evidence:
            go_dict[go_id]['evidence_codes'].add(evidence)

    # Convert to list
    consolidated = []
    for go_id, data in go_dict.items():
        # Sort evidence codes by priority (best first)
        sorted_evidence = sorted(data['evidence_codes'], key=get_evidence_priority)
        # Best evidence code for sorting
        best_evidence_priority = get_evidence_priority(sorted_evidence[0]) if sorted_evidence else 99
        consolidated.append({
            'go_id': go_id,
            'name': data['name'],
            'namespace': data['namespace'],
            'definition': data.get('definition', ''),
            'evidence_codes': sorted_evidence,
            'num_evidence': len(sorted_evidence),
            'best_evidence_priority': best_evidence_priority
        })

    # Sort by: 1) number of evidence codes (descending), 2) best evidence quality (ascending)
    consolidated.sort(key=lambda x: (-x['num_evidence'], x['best_evidence_priority']))

    return consolidated


def render_go_terms(gene_data: Dict[str, Any]):
    """Render the GO Terms factual section with definitions and evidence codes."""
    st.markdown("---")
    st.markdown("### GO Term Annotations")

    go_terms = gene_data.get('go_terms', [])

    if go_terms:
        # Consolidate duplicates and preserve definitions already stored in the state
        consolidated = consolidate_go_terms(go_terms)

        # Group by namespace
        go_by_ns = {
            'molecular_function': [],
            'biological_process': [],
            'cellular_component': []
        }
        for term in consolidated:
            ns = term.get('namespace', 'unknown')
            if ns in go_by_ns:
                go_by_ns[ns].append(term)

        tab1, tab2, tab3 = st.tabs([
            f"Molecular Function ({len(go_by_ns['molecular_function'])})",
            f"Biological Process ({len(go_by_ns['biological_process'])})",
            f"Cellular Component ({len(go_by_ns['cellular_component'])})"
        ])

        for tab, (ns, ns_label) in zip([tab1, tab2, tab3], [
            ('molecular_function', 'Molecular Function'),
            ('biological_process', 'Biological Process'),
            ('cellular_component', 'Cellular Component')
        ]):
            with tab:
                terms = go_by_ns[ns]
                if terms:
                    # Create DataFrame for top 10
                    top_terms = terms[:10]
                    df = pd.DataFrame([
                        {
                            'GO Term': t['name'],
                            'GO ID': t['go_id'],
                            'Evidence': ', '.join(t['evidence_codes']),
                            'GO Definition': (t.get('definition') or '').strip() or 'Definition unavailable'
                        }
                        for t in top_terms
                    ])

                    st.dataframe(
                        df,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "GO Term": st.column_config.TextColumn(width="medium"),
                            "GO ID": st.column_config.TextColumn(width="small"),
                            "Evidence": st.column_config.TextColumn(width="small"),
                            "GO Definition": st.column_config.TextColumn(width="large")
                        }
                    )

                    # Expandable for all terms
                    if len(terms) > 10:
                        with st.expander(f"Show all {len(terms)} unique {ns_label} terms"):
                            df_all = pd.DataFrame([
                                {
                                    'GO Term': t['name'],
                                    'GO ID': t['go_id'],
                                    'Evidence': ', '.join(t['evidence_codes']),
                                    'GO Definition': (t.get('definition') or '').strip() or 'Definition unavailable'
                                }
                                for t in terms
                            ])
                            st.dataframe(
                                df_all,
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    "GO Term": st.column_config.TextColumn(width="medium"),
                                    "GO ID": st.column_config.TextColumn(width="small"),
                                    "Evidence": st.column_config.TextColumn(width="small"),
                                    "GO Definition": st.column_config.TextColumn(width="large")
                                },
                                height=400
                            )
                else:
                    st.info(f"No {ns_label} terms")

        # Evidence code reference sits below the tables per UX guidance
        render_evidence_code_guide()

        # Source attribution
        render_source("Source: Gene Ontology (GO)")
    else:
        st.write(
            "No GO term annotations are currently recorded in the "
            "Gene Ontology database for this gene. This may reflect "
            "limited experimental characterisation rather than an "
            "absence of function."
        )
        st.info("No GO term data available")


def render_expression_profile(gene_data: Dict[str, Any]):
    """Render the Expression Profile factual section."""
    st.markdown("---")
    st.markdown("### Expression Profile")
    st.caption("TPM = Transcripts Per Million, a measure of gene expression level")

    expression = gene_data.get('expression', [])

    if expression:
        sorted_expr = sorted(expression, key=lambda x: x.get('level', 0) or 0, reverse=True)
        top_expr = sorted_expr[:10]

        if top_expr:
            df_top = pd.DataFrame([
                {
                    'Tissue': e.get('tissue', 'Unknown').replace('_', ' '),
                    'TPM': e.get('level', 0) or 0
                }
                for e in top_expr
            ])

            st.caption(f"Showing top 10 of {len(expression)} tissues by expression level")
            st.bar_chart(df_top.set_index('Tissue'), horizontal=True)

            if len(expression) > 10:
                with st.expander(f"View all {len(expression)} tissues"):
                    df_all = pd.DataFrame([
                        {
                            'Tissue': e.get('tissue', 'Unknown').replace('_', ' '),
                            'TPM': e.get('level', 0) or 0
                        }
                        for e in sorted_expr
                    ])
                    st.dataframe(
                        df_all,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "TPM": st.column_config.NumberColumn(format="%.2f")
                        }
                    )

            dataset = top_expr[0].get('dataset', 'GTEx') if top_expr else 'GTEx'
            render_source(f"Source: {dataset}")
    else:
        st.write(
            "No expression data are currently recorded in GTEx v8 for this gene. "
            "This may reflect limited experimental characterisation rather than an "
            "absence of expression."
        )
        st.info("No expression data available")


def render_protein_interactions(gene_data: Dict[str, Any]):
    """Render the Protein Interactions factual section with search and scrollable table."""
    st.markdown("---")
    st.markdown("### Protein Interactions")

    interactions = gene_data.get('interactions', [])
    total_count = gene_data.get('total_interaction_count', len(interactions))

    # Display limit for UI performance
    DISPLAY_LIMIT = 500

    if interactions:
        def _raw_score(inter: Dict[str, Any]) -> float:
            score = inter.get('score')
            if score is None:
                score = inter.get('combined_score', 0)
            return float(score or 0)

        def _partner_name(inter: Dict[str, Any]) -> str:
            return (
                inter.get('partner_symbol')
                or inter.get('partner')
                or 'Unknown'
            )

        def _function(inter: Dict[str, Any]) -> str:
            """Get partner gene function description."""
            return inter.get('partner_function') or ''

        # Sort by confidence (highest first) and limit for performance
        sorted_inter = sorted(interactions, key=_raw_score, reverse=True)
        display_inter = sorted_inter[:DISPLAY_LIMIT]

        # Build full dataframe
        all_rows = []
        for inter in display_inter:
            partner = _partner_name(inter)
            score = _raw_score(inter)
            func = _function(inter)
            all_rows.append({
                'Partner': partner,
                'Function': func,
                'Score': score / 1000
            })

        df_full = pd.DataFrame(all_rows)

        if not df_full.empty:
            # Search box for filtering
            search_query = st.text_input(
                "Search partner genes",
                placeholder="🔍 Type to filter by partner name or function...",
                key="protein_interaction_search",
                label_visibility="collapsed"
            )

            # Filter dataframe based on search
            if search_query:
                search_lower = search_query.lower()
                mask = (
                    df_full['Partner'].str.lower().str.contains(search_lower, na=False) |
                    df_full['Function'].str.lower().str.contains(search_lower, na=False)
                )
                df_filtered = df_full[mask]
            else:
                df_filtered = df_full

            # Display scrollable table — height fits rows exactly, capped at 10
            _visible_rows = min(len(df_filtered), 10)
            _table_height = 38 + (_visible_rows * 35)  # 38px header + 35px per row
            st.dataframe(
                df_filtered,
                width='stretch',
                hide_index=True,
                height=_table_height,
                column_config={
                    "Partner": st.column_config.TextColumn(width="small"),
                    "Function": st.column_config.TextColumn(width="medium"),
                    "Score": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.3f",
                        min_value=0,
                        max_value=1,
                    )
                }
            )

            # Stats line
            avg_score = sum(_raw_score(i) for i in interactions) / len(interactions)
            showing_count = len(df_filtered)
            total_in_table = len(df_full)

            if search_query:
                stats_text = f"Showing {showing_count:,} of {total_in_table:,} interactions (filtered)"
            elif total_count > DISPLAY_LIMIT:
                stats_text = f"Showing top {total_in_table:,} of {total_count:,} interactions"
            else:
                stats_text = f"Total interactions: {total_count:,}"

            st.caption(stats_text)

            # Confidence score explanation
            st.caption(
                "*Confidence score (0-1) indicates interaction reliability "
                "based on experimental evidence, database annotations, "
                "text mining, and co-expression data from STRING database.*"
            )
            st.caption("*Higher scores = stronger evidence.*")

            # Source attribution
            if display_inter and isinstance(display_inter[0], dict):
                prov = display_inter[0].get('provenance', {})
                if isinstance(prov, dict):
                    source_rel = prov.get('source_release', '')
                    if source_rel:
                        render_source(f"Source: STRING {source_rel}")
                    else:
                        render_source("Source: STRING")
    else:
        st.write(
            "No protein interactions above the confidence threshold (score ≥ 400) are "
            "currently recorded in STRING v12.0 for this gene. This may reflect "
            "limited experimental characterisation rather than an absence of interactions."
        )
        st.info("No protein interaction data available")


def render_literature(gene_symbol: str, papers_data: Dict[str, Any], candidates_data: Dict[str, Any] = None):
    """
    Render literature evidence for a gene.

    Fixes:
      1. Search details moved to ONE collapsed expander at the bottom
      2. Deduplication math corrected
      3. Full queries shown (not truncated)
      4. BM25 terms shown if ranking_method == 'bm25'
      5. Clean filtering pipeline with correct counts at each stage

    Args:
        gene_symbol:    e.g. "MED12"
        papers_data:    from gene_top_papers state
        candidates_data: from gene_literature_candidates state
    """
    st.markdown("---")

    # ── Guard ──────────────────────────────────────────────────────────────
    if not papers_data:
        st.info("No literature data available for this gene.")
        return

    top_papers = papers_data.get("top_papers", [])
    total_candidates = papers_data.get("total_candidates", 0)
    fetch_strategy = papers_data.get("fetch_strategy", "unknown")
    ranking_method = papers_data.get("ranking_method", "unknown")

    tier_contributions = []
    if candidates_data:
        tier_contributions = candidates_data.get("tier_contributions", [])

    # ── Section header ─────────────────────────────────────────────────────
    st.markdown("### 📚 Literature Evidence")
    st.caption("★ Relevance score: 1 = brief mention → 5 = directly studies this gene in context")

    if not top_papers:
        st.warning("No papers found for this gene.")
        _render_search_details(
            gene_symbol, papers_data, candidates_data,
            tier_contributions, top_papers, total_candidates,
            fetch_strategy, ranking_method,
        )
        return

    # ── Paper cards ────────────────────────────────────────────────────────
    for i, paper in enumerate(top_papers, 1):
        _render_paper_card(i, paper, expanded=(i <= 3))

    st.markdown("---")

    # ── Search details expander (bottom, collapsed by default) ─────────────
    _render_search_details(
        gene_symbol, papers_data, candidates_data,
        tier_contributions, top_papers, total_candidates,
        fetch_strategy, ranking_method,
    )


# ─── Paper card ────────────────────────────────────────────────────────────────

def _render_paper_card(index: int, paper: dict, expanded: bool = False):
    """Render a single paper as an expandable card."""
    title = paper.get("title", "Untitled")
    pmid = paper.get("pmid", "")
    journal = paper.get("journal", "")
    year = paper.get("year", "")
    abstract = paper.get("abstract", "")
    finding = paper.get("key_finding", "")
    score = paper.get("relevance_score", 0)

    try:
        score_int = int(round(float(score))) if score else 0
    except (ValueError, TypeError):
        score_int = 0

    # Only show stars if score > 0
    if score_int > 0:
        stars = "★" * score_int + "☆" * (5 - score_int)
    else:
        stars = None

    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None

    label = f"**{index}. {title}**"
    with st.expander(label, expanded=expanded):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"{journal} ({year})")
            if pubmed_url:
                st.markdown(f"[View on PubMed]({pubmed_url})")
        with col2:
            # Only render stars if they exist (score > 0)
            if stars:
                st.markdown(
                    f'<span style="font-size:1.2em; color:#f0a500;">{stars}</span>',
                    unsafe_allow_html=True,
                )

        if abstract:
            st.markdown(abstract)

        if finding:
            st.info(f"**🔍 Key Finding (AI-extracted):** {finding}")


# ─── Search details expander ───────────────────────────────────────────────────

def _render_search_details(
    gene_symbol, papers_data, candidates_data,
    tier_contributions, top_papers, total_candidates,
    fetch_strategy, ranking_method,
):
    """Render search and ranking details in a collapsed expander."""
    with st.expander("🔍 Search & Ranking Details", expanded=False):

        # ── 1. Queries run ─────────────────────────────────────────────────
        st.markdown("#### Queries Run")

        # Strategy badge
        strategy_label = fetch_strategy.replace("_", " ").title()
        st.markdown(
            f'<span style="background:#1e3a5f; color:#7eb8f7; '
            f'padding:2px 10px; border-radius:12px; font-size:0.8em;">'
            f'Strategy: {strategy_label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        if tier_contributions:
            tier_numbers_present = set()
            for tier_entry in tier_contributions:
                try:
                    tier_numbers_present.add(int(tier_entry.get("tier")))
                except (TypeError, ValueError):
                    continue

            missing_early_tiers = [t for t in (1, 2) if t not in tier_numbers_present]
            if missing_early_tiers:
                missing_str = " and ".join(f"Tier {t}" for t in missing_early_tiers)
                st.caption(
                    "ℹ️ "
                    f"{missing_str} not generated — additional disease, phenotype, or tissue context "
                    "was not available for this query."
                )

            # Calculate total papers fetched from earlier tiers (for skipped tier message)
            papers_fetched_so_far = 0
            for tier in tier_contributions:
                tier_num = tier.get("tier", "?")
                query = tier.get("query", "N/A")
                count = tier.get("count")
                status = tier.get("status", "unknown")
                contributed = tier.get("contributed", 0) or 0

                try:
                    tier_num_for_label = int(tier_num)
                except (TypeError, ValueError):
                    tier_num_for_label = tier_num

                tier_label = _tier_label(tier_num_for_label)
                status_icon = _TIER_STATUS_ICON.get(status, "⚠️")

                # Custom status text for skipped tiers
                if status == "skipped":
                    status_text = f"skipped — {papers_fetched_so_far} papers already found in earlier tier"
                else:
                    status_text = _TIER_STATUS_LABEL.get(status, status.replace('_', ' '))

                if isinstance(count, (int, float)):
                    papers_display = f"{int(count):,}"
                elif count is None:
                    papers_display = ""  # Don't show "n/a papers" for skipped
                else:
                    papers_display = str(count)

                # Build the tier line
                papers_suffix = f"  ` {papers_display} papers `" if papers_display else ""
                st.markdown(
                    f"**Tier {tier_num}** — {tier_label}  {status_icon} "
                    f"*{status_text}*{papers_suffix}"
                )

                # Show query for all tiers
                st.code(query, language="text")

                # Add note for tiers that ran successfully
                if status in _TIER_SUCCESS_STATUSES and contributed > 0:
                    st.caption(f"↳ Top {contributed} fetched · sorted by date (newest first)")

                # Track papers fetched for skipped tier message
                if status in _TIER_SUCCESS_STATUSES:
                    papers_fetched_so_far += contributed
        else:
            # Fallback: show the single query from papers_data
            query_used = papers_data.get("query_used", "N/A")
            tier_used = papers_data.get("tier_used", "?")
            tier_label = _tier_label(tier_used)
            st.markdown(f"**Tier {tier_used}** — {tier_label}")
            st.code(query_used, language="text")

        # ── 2. BM25 terms (if applicable) ──────────────────────────────────
        if ranking_method == "bm25":
            bm25_terms = papers_data.get("bm25_terms") or papers_data.get("ranking_terms")
            if bm25_terms:
                st.markdown("#### BM25 Ranking Terms")
                if isinstance(bm25_terms, list):
                    terms_display = "  ·  ".join(bm25_terms)
                else:
                    terms_display = str(bm25_terms)
                st.markdown(
                    f'<div style="font-family:monospace; background:#0e1117; '
                    f'border:1px solid #333; border-radius:6px; padding:10px; '
                    f'font-size:0.85em; color:#a8d8a8;">{terms_display}</div>',
                    unsafe_allow_html=True,
                )

        # ── 3. Filtering pipeline ──────────────────────────────────────────
        st.markdown("#### Filtering Pipeline")

        # Pipeline stages:
        #   pubmed_matched   = sum of tier counts (how many papers exist in PubMed)
        #   papers_fetched   = papers_before_bm25 (actually downloaded, max ~200)
        #   after_bm25       = total_candidates (BM25 pre-ranked subset)
        #   after_llm        = len(top_papers) (final LLM-ranked selection)

        # Get PubMed match count from tier contributions
        if tier_contributions:
            pubmed_matched = sum(
                t.get("count", 0) or 0
                for t in tier_contributions
                if t.get("status") in _TIER_SUCCESS_STATUSES
            )
        else:
            pubmed_matched = total_candidates

        # Get actual papers fetched (before BM25)
        papers_fetched = (
            (candidates_data or {}).get("papers_before_bm25")
            or (candidates_data or {}).get("total_count")
            or total_candidates
        )

        after_bm25 = total_candidates  # post-BM25 pool
        after_llm = len(top_papers)    # final selection

        # Build pipeline rows with clearer labels
        pipeline_rows = [
            (
                "🔍 Matched in PubMed",
                f"{pubmed_matched:,}",
                f"{pubmed_matched:,} papers match the query — top {papers_fetched} fetched for ranking",
            ),
            (
                "⚡ After BM25 pre-ranking",
                after_bm25,
                f"BM25 keyword ranking selected top {after_bm25} from {papers_fetched} fetched papers",
            ),
            (
                "🏆 After LLM ranking",
                after_llm,
                f"LLM selected top {after_llm} most relevant from {after_bm25} BM25 candidates",
            ),
        ]

        for label, count, detail in pipeline_rows:
            col1, col2, col3 = st.columns([3, 1, 3])
            with col1:
                st.markdown(label)
            with col2:
                st.markdown(f"**{count}**")
            with col3:
                st.caption(detail)

        # Source note
        st.markdown("")
        st.caption(
            f"Source: PubMed · BM25 pre-ranking · LLM relevance ranking · "
            f"Displaying top {len(top_papers)} papers"
        )


def _tier_label(tier_num) -> str:
    """Human-readable label for each query tier."""
    labels = {
        1: "Gene AND disease AND variant",
        2: "Gene AND disease",
        3: "Gene AND disease (broader)",
        4: "Gene only (human filter)",
        "1": "Gene AND disease AND variant",
        "2": "Gene AND disease",
        "3": "Gene AND disease (broader)",
        "4": "Gene only (human filter)",
    }
    return labels.get(tier_num, f"Tier {tier_num} query")


# ============================================================================
# HELPER FUNCTIONS FOR AI SECTIONS
# ============================================================================

def render_ai_gene_summary(state: Dict[str, Any], selected_gene: str, interpretation: Dict[str, Any]):
    """Render the AI Gene Summary section - comprehensive synthesis of all data."""
    st.markdown("---")
    st.markdown("### Comprehensive Gene Summary")

    # Read directly from state — already the post-validation corrected version
    # (state.gene_summaries is updated in-place by validation nodes)
    gene_summaries = state.get('gene_summaries', {})
    gene_summary_text = None
    source = None

    if selected_gene in gene_summaries:
        summary_data = gene_summaries[selected_gene]
        if isinstance(summary_data, str) and summary_data.strip():
            gene_summary_text = summary_data
            source = "comprehensive"
        elif isinstance(summary_data, dict):
            gene_summary_text = (
                summary_data.get('gene_summary') or
                summary_data.get('summary') or
                summary_data.get('gene_description') or
                summary_data.get('functional_summary')
            )
            if gene_summary_text:
                source = "comprehensive"

    # Fallback to interpretation summary fields
    if not gene_summary_text and interpretation:
        gene_summary_text = (
            interpretation.get('gene_summary') or
            interpretation.get('summary') or
            interpretation.get('gene_description')
        )
        if gene_summary_text:
            source = "interpretation"

    # Check if validation refined this section (from snapshot)
    was_refined = False
    vr = state.get('validation_results', {})
    if isinstance(vr, dict):
        gene_snap = vr.get(selected_gene, {})
        gs_section = (gene_snap.get('sections', {}).get('gene_summary')
                      or gene_snap.get('gene_summary_validation'))
        if isinstance(gs_section, dict) and gs_section.get('status') == 'corrected':
            was_refined = True

    if gene_summary_text and source == "comprehensive":
        caption_parts = ["*AI-synthesized overview combining function, expression, interactions, and literature*"]
        if was_refined:
            caption_parts.append(" ✓ Validator refined")
        st.caption("".join(caption_parts))
        st.markdown(
            f'<div style="background-color: #d4edda; border-left: 4px solid #28a745; '
            f'padding: 1rem 1rem 1rem 1.5rem; border-radius: 4px; line-height: 1.6;">'
            f'{gene_summary_text}</div>',
            unsafe_allow_html=True
        )
    elif gene_summary_text:
        caption_parts = ["*AI-generated overview*"]
        if was_refined:
            caption_parts.append(" ✓ Validator refined")
        st.caption("".join(caption_parts))
        st.markdown(
            f'<div style="background-color: #d4edda; border-left: 4px solid #28a745; '
            f'padding: 1rem 1rem 1rem 1.5rem; border-radius: 4px; line-height: 1.6;">'
            f'{gene_summary_text}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("No comprehensive summary available. This is generated by the pipeline in 'interpreted' mode.")


def render_functional_interpretation(interpretation: Dict[str, Any], gene_symbol: str):
    """Render the Functional Interpretation AI section."""
    st.markdown("##### What this means:")
    # Read directly from state — state is already the post-validation corrected version
    # (ValidateInterpretations writes the corrected text back into state.gene_interpretations)
    func_interp = interpretation.get('functional_summary', interpretation.get('functional_interpretation', ''))
    if func_interp:
        render_ai_interpretation("Functional Interpretation", func_interp)
    else:
        st.caption("*No AI interpretation available. Run pipeline in 'interpreted' mode.*")


def render_expression_interpretation(interpretation: Dict[str, Any], gene_symbol: str):
    """Render the Expression Interpretation AI section."""
    st.markdown("##### What this means:")
    # Read directly from state — already the post-validation corrected version
    expr_interp = interpretation.get('expression_interpretation', '')
    if expr_interp:
        render_ai_interpretation("Expression Interpretation", expr_interp)
    else:
        st.caption("*No AI interpretation available. Run pipeline in 'interpreted' mode.*")


def render_network_interpretation(interpretation: Dict[str, Any], gene_symbol: str):
    """Render the Network Interpretation AI section."""
    st.markdown("##### What this means:")
    # Read directly from state — already the post-validation corrected version
    network_interp = interpretation.get('interaction_interpretation', interpretation.get('network_interpretation', ''))
    if network_interp:
        render_ai_interpretation("Network Interpretation", network_interp)
    else:
        st.caption("*No AI interpretation available. Run pipeline in 'interpreted' mode.*")


def render_experimental_relevance(interpretation: Dict[str, Any], gene_symbol: str):
    """Render the Experimental Relevance AI section."""
    st.markdown("---")
    st.markdown("### Experimental Relevance")

    # Read directly from state — already the post-validation corrected version
    exp_relevance = interpretation.get('experimental_relevance', '')
    if exp_relevance:
        render_ai_interpretation("Context-Specific Analysis", exp_relevance)
    else:
        st.info("No experimental relevance analysis available. Run pipeline in 'interpreted' mode.")


def _get_validation_snapshot(gene_symbol: str, state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the state-native validation snapshot for a gene (or None)."""
    if not state:
        return None
    vr = state.get("validation_results", {})
    if isinstance(vr, dict):
        return vr.get(gene_symbol)
    return None


def get_evidence_priority(code: str) -> int:
    """Sort evidence codes by reliability (best first)."""
    priority = {
        'EXP': 1,   # Experimental
        'IDA': 2,   # Direct Assay
        'IPI': 3,   # Physical Interaction
        'IMP': 4,   # Mutant Phenotype
        'IGI': 5,   # Genetic Interaction
        'IEP': 6,   # Expression Pattern
        'HDA': 7,   # High-throughput Direct Assay
        'HMP': 8,   # High-throughput Mutant Phenotype
        'HGI': 9,   # High-throughput Genetic Interaction
        'HEP': 10,  # High-throughput Expression Pattern
        'TAS': 11,  # Traceable Author Statement
        'NAS': 12,  # Non-traceable Author Statement
        'IC': 13,   # Inferred by Curator
        'IBA': 14,  # Biological Ancestor
        'IBD': 15,  # Biological Descendant
        'IKR': 16,  # Key Residues
        'IRD': 17,  # Rapid Divergence
        'ISS': 18,  # Sequence/Structural Similarity
        'ISO': 19,  # Sequence Orthology
        'ISA': 20,  # Sequence Alignment
        'ISM': 21,  # Sequence Model
        'IGC': 22,  # Genomic Context
        'RCA': 23,  # Reviewed Computational Analysis
        'IEA': 24,  # Electronic Annotation (lowest confidence)
        'ND': 25,   # No Data
    }
    return priority.get(code, 99)


def consolidate_go_terms(go_terms_list: list) -> list:
    """Group GO terms by ID and consolidate evidence codes.

    Sorting priority:
    1. Number of evidence codes (descending) - more evidence = higher priority
    2. Best evidence code quality (ascending) - stronger evidence = higher priority
    """
    go_dict = defaultdict(lambda: {
        'name': '',
        'namespace': '',
        'definition': '',
        'evidence_codes': set()
    })

    for term in go_terms_list:
        go_id = term.get('go_id', '')
        if not go_id:
            continue
        go_dict[go_id]['name'] = term.get('name', 'Unknown')
        go_dict[go_id]['go_id'] = go_id
        go_dict[go_id]['namespace'] = term.get('namespace', term.get('category', ''))
        definition_value = term.get('definition')
        if definition_value and not go_dict[go_id]['definition']:
            go_dict[go_id]['definition'] = definition_value
        evidence = term.get('evidence', '')
        if evidence:
            go_dict[go_id]['evidence_codes'].add(evidence)

    # Convert to list
    consolidated = []
    for go_id, data in go_dict.items():
        # Sort evidence codes by priority (best first)
        sorted_evidence = sorted(data['evidence_codes'], key=get_evidence_priority)
        # Best evidence code for sorting
        best_evidence_priority = get_evidence_priority(sorted_evidence[0]) if sorted_evidence else 99
        consolidated.append({
            'go_id': go_id,
            'name': data['name'],
            'namespace': data['namespace'],
            'definition': data.get('definition', ''),
            'evidence_codes': sorted_evidence,
            'num_evidence': len(sorted_evidence),
            'best_evidence_priority': best_evidence_priority
        })

    # Sort by: 1) number of evidence codes (descending), 2) best evidence quality (ascending)
    consolidated.sort(key=lambda x: (-x['num_evidence'], x['best_evidence_priority']))

    return consolidated


def render_go_terms(gene_data: Dict[str, Any]):
    """Render the GO Terms factual section with definitions and evidence codes."""
    st.markdown("---")
    st.markdown("### GO Term Annotations")

    go_terms = gene_data.get('go_terms', [])

    if go_terms:
        # Consolidate duplicates and preserve definitions already stored in the state
        consolidated = consolidate_go_terms(go_terms)

        # Group by namespace
        go_by_ns = {
            'molecular_function': [],
            'biological_process': [],
            'cellular_component': []
        }
        for term in consolidated:
            ns = term.get('namespace', 'unknown')
            if ns in go_by_ns:
                go_by_ns[ns].append(term)

        tab1, tab2, tab3 = st.tabs([
            f"Molecular Function ({len(go_by_ns['molecular_function'])})",
            f"Biological Process ({len(go_by_ns['biological_process'])})",
            f"Cellular Component ({len(go_by_ns['cellular_component'])})"
        ])

        for tab, (ns, ns_label) in zip([tab1, tab2, tab3], [
            ('molecular_function', 'Molecular Function'),
            ('biological_process', 'Biological Process'),
            ('cellular_component', 'Cellular Component')
        ]):
            with tab:
                terms = go_by_ns[ns]
                if terms:
                    # Create DataFrame for top 10
                    top_terms = terms[:10]
                    df = pd.DataFrame([
                        {
                            'GO Term': t['name'],
                            'GO ID': t['go_id'],
                            'Evidence': ', '.join(t['evidence_codes']),
                            'GO Definition': (t.get('definition') or '').strip() or 'Definition unavailable'
                        }
                        for t in top_terms
                    ])

                    st.dataframe(
                        df,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "GO Term": st.column_config.TextColumn(width="medium"),
                            "GO ID": st.column_config.TextColumn(width="small"),
                            "Evidence": st.column_config.TextColumn(width="small"),
                            "GO Definition": st.column_config.TextColumn(width="large")
                        }
                    )

                    # Expandable for all terms
                    if len(terms) > 10:
                        with st.expander(f"Show all {len(terms)} unique {ns_label} terms"):
                            df_all = pd.DataFrame([
                                {
                                    'GO Term': t['name'],
                                    'GO ID': t['go_id'],
                                    'Evidence': ', '.join(t['evidence_codes']),
                                    'GO Definition': (t.get('definition') or '').strip() or 'Definition unavailable'
                                }
                                for t in terms
                            ])
                            st.dataframe(
                                df_all,
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    "GO Term": st.column_config.TextColumn(width="medium"),
                                    "GO ID": st.column_config.TextColumn(width="small"),
                                    "Evidence": st.column_config.TextColumn(width="small"),
                                    "GO Definition": st.column_config.TextColumn(width="large")
                                },
                                height=400
                            )
                else:
                    st.info(f"No {ns_label} terms")

        # Evidence code reference sits below the tables per UX guidance
        render_evidence_code_guide()

        # Source attribution
        render_source("Source: Gene Ontology (GO)")
    else:
        st.write(
            "No GO term annotations are currently recorded in the "
            "Gene Ontology database for this gene. This may reflect "
            "limited experimental characterisation rather than an "
            "absence of function."
        )
        st.info("No GO term data available")


def render_expression_profile(gene_data: Dict[str, Any]):
    """Render the Expression Profile factual section."""
    st.markdown("---")
    st.markdown("### Expression Profile")
    st.caption("TPM = Transcripts Per Million, a measure of gene expression level")

    expression = gene_data.get('expression', [])

    if expression:
        sorted_expr = sorted(expression, key=lambda x: x.get('level', 0) or 0, reverse=True)
        top_expr = sorted_expr[:10]

        if top_expr:
            df_top = pd.DataFrame([
                {
                    'Tissue': e.get('tissue', 'Unknown').replace('_', ' '),
                    'TPM': e.get('level', 0) or 0
                }
                for e in top_expr
            ])

            st.caption(f"Showing top 10 of {len(expression)} tissues by expression level")
            st.bar_chart(df_top.set_index('Tissue'), horizontal=True)

            if len(expression) > 10:
                with st.expander(f"View all {len(expression)} tissues"):
                    df_all = pd.DataFrame([
                        {
                            'Tissue': e.get('tissue', 'Unknown').replace('_', ' '),
                            'TPM': e.get('level', 0) or 0
                        }
                        for e in sorted_expr
                    ])
                    st.dataframe(
                        df_all,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "TPM": st.column_config.NumberColumn(format="%.2f")
                        }
                    )

            dataset = top_expr[0].get('dataset', 'GTEx') if top_expr else 'GTEx'
            render_source(f"Source: {dataset}")
    else:
        st.write(
            "No expression data are currently recorded in GTEx v8 for this gene. "
            "This may reflect limited experimental characterisation rather than an "
            "absence of expression."
        )
        st.info("No expression data available")


def render_protein_interactions(gene_data: Dict[str, Any]):
    """Render the Protein Interactions factual section with search and scrollable table."""
    st.markdown("---")
    st.markdown("### Protein Interactions")

    interactions = gene_data.get('interactions', [])
    total_count = gene_data.get('total_interaction_count', len(interactions))

    # Display limit for UI performance
    DISPLAY_LIMIT = 500

    if interactions:
        def _raw_score(inter: Dict[str, Any]) -> float:
            score = inter.get('score')
            if score is None:
                score = inter.get('combined_score', 0)
            return float(score or 0)

        def _partner_name(inter: Dict[str, Any]) -> str:
            return (
                inter.get('partner_symbol')
                or inter.get('partner')
                or 'Unknown'
            )

        def _function(inter: Dict[str, Any]) -> str:
            """Get partner gene function description."""
            return inter.get('partner_function') or ''

        # Sort by confidence (highest first) and limit for performance
        sorted_inter = sorted(interactions, key=_raw_score, reverse=True)
        display_inter = sorted_inter[:DISPLAY_LIMIT]

        # Build full dataframe
        all_rows = []
        for inter in display_inter:
            partner = _partner_name(inter)
            score = _raw_score(inter)
            func = _function(inter)
            all_rows.append({
                'Partner': partner,
                'Function': func,
                'Score': score / 1000
            })

        df_full = pd.DataFrame(all_rows)

        if not df_full.empty:
            # Search box for filtering
            search_query = st.text_input(
                "Search partner genes",
                placeholder="🔍 Type to filter by partner name or function...",
                key="protein_interaction_search",
                label_visibility="collapsed"
            )

            # Filter dataframe based on search
            if search_query:
                search_lower = search_query.lower()
                mask = (
                    df_full['Partner'].str.lower().str.contains(search_lower, na=False) |
                    df_full['Function'].str.lower().str.contains(search_lower, na=False)
                )
                df_filtered = df_full[mask]
            else:
                df_filtered = df_full

            # Display scrollable table — height fits rows exactly, capped at 10
            _visible_rows = min(len(df_filtered), 10)
            _table_height = 38 + (_visible_rows * 35)  # 38px header + 35px per row
            st.dataframe(
                df_filtered,
                width='stretch',
                hide_index=True,
                height=_table_height,
                column_config={
                    "Partner": st.column_config.TextColumn(width="small"),
                    "Function": st.column_config.TextColumn(width="medium"),
                    "Score": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.3f",
                        min_value=0,
                        max_value=1,
                    )
                }
            )

            # Stats line
            avg_score = sum(_raw_score(i) for i in interactions) / len(interactions)
            showing_count = len(df_filtered)
            total_in_table = len(df_full)

            if search_query:
                stats_text = f"Showing {showing_count:,} of {total_in_table:,} interactions (filtered)"
            elif total_count > DISPLAY_LIMIT:
                stats_text = f"Showing top {total_in_table:,} of {total_count:,} interactions"
            else:
                stats_text = f"Total interactions: {total_count:,}"

            st.caption(stats_text)

            # Confidence score explanation
            st.caption(
                "*Confidence score (0-1) indicates interaction reliability "
                "based on experimental evidence, database annotations, "
                "text mining, and co-expression data from STRING database.*"
            )
            st.caption("*Higher scores = stronger evidence.*")

            # Source attribution
            if display_inter and isinstance(display_inter[0], dict):
                prov = display_inter[0].get('provenance', {})
                if isinstance(prov, dict):
                    source_rel = prov.get('source_release', '')
                    if source_rel:
                        render_source(f"Source: STRING {source_rel}")
                    else:
                        render_source("Source: STRING")
    else:
        st.write(
            "No protein interactions above the confidence threshold (score ≥ 400) are "
            "currently recorded in STRING v12.0 for this gene. This may reflect "
            "limited experimental characterisation rather than an absence of interactions."
        )
        st.info("No protein interaction data available")


def render_literature(gene_symbol: str, papers_data: Dict[str, Any], candidates_data: Dict[str, Any] = None):
    """
    Render literature evidence for a gene.

    Fixes:
      1. Search details moved to ONE collapsed expander at the bottom
      2. Deduplication math corrected
      3. Full queries shown (not truncated)
      4. BM25 terms shown if ranking_method == 'bm25'
      5. Clean filtering pipeline with correct counts at each stage

    Args:
        gene_symbol:    e.g. "MED12"
        papers_data:    from gene_top_papers state
        candidates_data: from gene_literature_candidates state
    """
    st.markdown("---")

    # ── Guard ──────────────────────────────────────────────────────────────
    if not papers_data:
        st.info("No literature data available for this gene.")
        return

    top_papers = papers_data.get("top_papers", [])
    total_candidates = papers_data.get("total_candidates", 0)
    fetch_strategy = papers_data.get("fetch_strategy", "unknown")
    ranking_method = papers_data.get("ranking_method", "unknown")

    tier_contributions = []
    if candidates_data:
        tier_contributions = candidates_data.get("tier_contributions", [])

    # ── Section header ─────────────────────────────────────────────────────
    st.markdown("### 📚 Literature Evidence")
    st.caption("★ Relevance score: 1 = brief mention → 5 = directly studies this gene in context")

    if not top_papers:
        st.warning("No papers found for this gene.")
        _render_search_details(
            gene_symbol, papers_data, candidates_data,
            tier_contributions, top_papers, total_candidates,
            fetch_strategy, ranking_method,
        )
        return

    # ── Paper cards ────────────────────────────────────────────────────────
    for i, paper in enumerate(top_papers, 1):
        _render_paper_card(i, paper, expanded=(i <= 3))

    st.markdown("---")

    # ── Search details expander (bottom, collapsed by default) ─────────────
    _render_search_details(
        gene_symbol, papers_data, candidates_data,
        tier_contributions, top_papers, total_candidates,
        fetch_strategy, ranking_method,
    )


# ─── Paper card ────────────────────────────────────────────────────────────────

def _render_paper_card(index: int, paper: dict, expanded: bool = False):
    """Render a single paper as an expandable card."""
    title = paper.get("title", "Untitled")
    pmid = paper.get("pmid", "")
    journal = paper.get("journal", "")
    year = paper.get("year", "")
    abstract = paper.get("abstract", "")
    finding = paper.get("key_finding", "")
    score = paper.get("relevance_score", 0)

    try:
        score_int = int(round(float(score))) if score else 0
    except (ValueError, TypeError):
        score_int = 0

    # Only show stars if score > 0
    if score_int > 0:
        stars = "★" * score_int + "☆" * (5 - score_int)
    else:
        stars = None

    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None

    label = f"**{index}. {title}**"
    with st.expander(label, expanded=expanded):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"{journal} ({year})")
            if pubmed_url:
                st.markdown(f"[View on PubMed]({pubmed_url})")
        with col2:
            # Only render stars if they exist (score > 0)
            if stars:
                st.markdown(
                    f'<span style="font-size:1.2em; color:#f0a500;">{stars}</span>',
                    unsafe_allow_html=True,
                )

        if abstract:
            st.markdown(abstract)

        if finding:
            st.info(f"**🔍 Key Finding (AI-extracted):** {finding}")


# ─── Search details expander ───────────────────────────────────────────────────

def _render_search_details(
    gene_symbol, papers_data, candidates_data,
    tier_contributions, top_papers, total_candidates,
    fetch_strategy, ranking_method,
):
    """Render search and ranking details in a collapsed expander."""
    with st.expander("🔍 Search & Ranking Details", expanded=False):

        # ── 1. Queries run ─────────────────────────────────────────────────
        st.markdown("#### Queries Run")

        # Strategy badge
        strategy_label = fetch_strategy.replace("_", " ").title()
        st.markdown(
            f'<span style="background:#1e3a5f; color:#7eb8f7; '
            f'padding:2px 10px; border-radius:12px; font-size:0.8em;">'
            f'Strategy: {strategy_label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        if tier_contributions:
            tier_numbers_present = set()
            for tier_entry in tier_contributions:
                try:
                    tier_numbers_present.add(int(tier_entry.get("tier")))
                except (TypeError, ValueError):
                    continue

            missing_early_tiers = [t for t in (1, 2) if t not in tier_numbers_present]
            if missing_early_tiers:
                missing_str = " and ".join(f"Tier {t}" for t in missing_early_tiers)
                st.caption(
                    "ℹ️ "
                    f"{missing_str} not generated — additional disease, phenotype, or tissue context "
                    "was not available for this query."
                )

            # Calculate total papers fetched from earlier tiers (for skipped tier message)
            papers_fetched_so_far = 0
            for tier in tier_contributions:
                tier_num = tier.get("tier", "?")
                query = tier.get("query", "N/A")
                count = tier.get("count")
                status = tier.get("status", "unknown")
                contributed = tier.get("contributed", 0) or 0

                try:
                    tier_num_for_label = int(tier_num)
                except (TypeError, ValueError):
                    tier_num_for_label = tier_num

                tier_label = _tier_label(tier_num_for_label)
                status_icon = _TIER_STATUS_ICON.get(status, "⚠️")

                # Custom status text for skipped tiers
                if status == "skipped":
                    status_text = f"skipped — {papers_fetched_so_far} papers already found in earlier tier"
                else:
                    status_text = _TIER_STATUS_LABEL.get(status, status.replace('_', ' '))

                if isinstance(count, (int, float)):
                    papers_display = f"{int(count):,}"
                elif count is None:
                    papers_display = ""  # Don't show "n/a papers" for skipped
                else:
                    papers_display = str(count)

                # Build the tier line
                papers_suffix = f"  ` {papers_display} papers `" if papers_display else ""
                st.markdown(
                    f"**Tier {tier_num}** — {tier_label}  {status_icon} "
                    f"*{status_text}*{papers_suffix}"
                )

                # Show query for all tiers
                st.code(query, language="text")

                # Add note for tiers that ran successfully
                if status in _TIER_SUCCESS_STATUSES and contributed > 0:
                    st.caption(f"↳ Top {contributed} fetched · sorted by date (newest first)")

                # Track papers fetched for skipped tier message
                if status in _TIER_SUCCESS_STATUSES:
                    papers_fetched_so_far += contributed
        else:
            # Fallback: show the single query from papers_data
            query_used = papers_data.get("query_used", "N/A")
            tier_used = papers_data.get("tier_used", "?")
            tier_label = _tier_label(tier_used)
            st.markdown(f"**Tier {tier_used}** — {tier_label}")
            st.code(query_used, language="text")

        # ── 2. BM25 terms (if applicable) ──────────────────────────────────
        if ranking_method == "bm25":
            bm25_terms = papers_data.get("bm25_terms") or papers_data.get("ranking_terms")
            if bm25_terms:
                st.markdown("#### BM25 Ranking Terms")
                if isinstance(bm25_terms, list):
                    terms_display = "  ·  ".join(bm25_terms)
                else:
                    terms_display = str(bm25_terms)
                st.markdown(
                    f'<div style="font-family:monospace; background:#0e1117; '
                    f'border:1px solid #333; border-radius:6px; padding:10px; '
                    f'font-size:0.85em; color:#a8d8a8;">{terms_display}</div>',
                    unsafe_allow_html=True,
                )

        # ── 3. Filtering pipeline ──────────────────────────────────────────
        st.markdown("#### Filtering Pipeline")

        # Pipeline stages:
        #   pubmed_matched   = sum of tier counts (how many papers exist in PubMed)
        #   papers_fetched   = papers_before_bm25 (actually downloaded, max ~200)
        #   after_bm25       = total_candidates (BM25 pre-ranked subset)
        #   after_llm        = len(top_papers) (final LLM-ranked selection)

        # Get PubMed match count from tier contributions
        if tier_contributions:
            pubmed_matched = sum(
                t.get("count", 0) or 0
                for t in tier_contributions
                if t.get("status") in _TIER_SUCCESS_STATUSES
            )
        else:
            pubmed_matched = total_candidates

        # Get actual papers fetched (before BM25)
        papers_fetched = (
            (candidates_data or {}).get("papers_before_bm25")
            or (candidates_data or {}).get("total_count")
            or total_candidates
        )

        after_bm25 = total_candidates  # post-BM25 pool
        after_llm = len(top_papers)    # final selection

        # Build pipeline rows with clearer labels
        pipeline_rows = [
            (
                "🔍 Matched in PubMed",
                f"{pubmed_matched:,}",
                f"{pubmed_matched:,} papers match the query — top {papers_fetched} fetched for ranking",
            ),
            (
                "⚡ After BM25 pre-ranking",
                after_bm25,
                f"BM25 keyword ranking selected top {after_bm25} from {papers_fetched} fetched papers",
            ),
            (
                "🏆 After LLM ranking",
                after_llm,
                f"LLM selected top {after_llm} most relevant from {after_bm25} BM25 candidates",
            ),
        ]

        for label, count, detail in pipeline_rows:
            col1, col2, col3 = st.columns([3, 1, 3])
            with col1:
                st.markdown(label)
            with col2:
                st.markdown(f"**{count}**")
            with col3:
                st.caption(detail)

        # Source note
        st.markdown("")
        st.caption(
            f"Source: PubMed · BM25 pre-ranking · LLM relevance ranking · "
            f"Displaying top {len(top_papers)} papers"
        )


def _tier_label(tier_num) -> str:
    """Human-readable label for each query tier."""
    labels = {
        1: "Gene AND disease AND variant",
        2: "Gene AND disease",
        3: "Gene AND disease (broader)",
        4: "Gene only (human filter)",
        "1": "Gene AND disease AND variant",
        "2": "Gene AND disease",
        "3": "Gene AND disease (broader)",
        "4": "Gene only (human filter)",
    }
    return labels.get(tier_num, f"Tier {tier_num} query")


# ============================================================================
# HELPER FUNCTIONS FOR AI SECTIONS
# ============================================================================

def render_ai_gene_summary(state: Dict[str, Any], selected_gene: str, interpretation: Dict[str, Any]):
    """Render the AI Gene Summary section - comprehensive synthesis of all data."""
    st.markdown("---")
    st.markdown("### Comprehensive Gene Summary")

    # Read directly from state — already the post-validation corrected version
    # (state.gene_summaries is updated in-place by validation nodes)
    gene_summaries = state.get('gene_summaries', {})
    gene_summary_text = None
    source = None

    if selected_gene in gene_summaries:
        summary_data = gene_summaries[selected_gene]
        if isinstance(summary_data, str) and summary_data.strip():
            gene_summary_text = summary_data
            source = "comprehensive"
        elif isinstance(summary_data, dict):
            gene_summary_text = (
                summary_data.get('gene_summary') or
                summary_data.get('summary') or
                summary_data.get('gene_description') or
                summary_data.get('functional_summary')
            )
            if gene_summary_text:
                source = "comprehensive"

    # Fallback to interpretation summary fields
    if not gene_summary_text and interpretation:
        gene_summary_text = (
            interpretation.get('gene_summary') or
            interpretation.get('summary') or
            interpretation.get('gene_description')
        )
        if gene_summary_text:
            source = "interpretation"

    # Check if validation refined this section (from snapshot)
    was_refined = False
    vr = state.get('validation_results', {})
    if isinstance(vr, dict):
        gene_snap = vr.get(selected_gene, {})
        gs_section = (gene_snap.get('sections', {}).get('gene_summary')
                      or gene_snap.get('gene_summary_validation'))
        if isinstance(gs_section, dict) and gs_section.get('status') == 'corrected':
            was_refined = True

    if gene_summary_text and source == "comprehensive":
        caption_parts = ["*AI-synthesized overview combining function, expression, interactions, and literature*"]
        if was_refined:
            caption_parts.append(" ✓ Validator refined")
        st.caption("".join(caption_parts))
        st.markdown(
            f'<div style="background-color: #d4edda; border-left: 4px solid #28a745; '
            f'padding: 1rem 1rem 1rem 1.5rem; border-radius: 4px; line-height: 1.6;">'
            f'{gene_summary_text}</div>',
            unsafe_allow_html=True
        )
    elif gene_summary_text:
        caption_parts = ["*AI-generated overview*"]
        if was_refined:
            caption_parts.append(" ✓ Validator refined")
        st.caption("".join(caption_parts))
        st.markdown(
            f'<div style="background-color: #d4edda; border-left: 4px solid #28a745; '
            f'padding: 1rem 1rem 1rem 1.5rem; border-radius: 4px; line-height: 1.6;">'
            f'{gene_summary_text}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("No comprehensive summary available. This is generated by the pipeline in 'interpreted' mode.")


def render_functional_interpretation(interpretation: Dict[str, Any], gene_symbol: str):
    """Render the Functional Interpretation AI section."""
    st.markdown("##### What this means:")
    # Read directly from state — state is already the post-validation corrected version
    # (ValidateInterpretations writes the corrected text back into state.gene_interpretations)
    func_interp = interpretation.get('functional_summary', interpretation.get('functional_interpretation', ''))
    if func_interp:
        render_ai_interpretation("Functional Interpretation", func_interp)
    else:
        st.caption("*No AI interpretation available. Run pipeline in 'interpreted' mode.*")


def render_expression_interpretation(interpretation: Dict[str, Any], gene_symbol: str):
    """Render the Expression Interpretation AI section."""
    st.markdown("##### What this means:")
    # Read directly from state — already the post-validation corrected version
    expr_interp = interpretation.get('expression_interpretation', '')
    if expr_interp:
        render_ai_interpretation("Expression Interpretation", expr_interp)
    else:
        st.caption("*No AI interpretation available. Run pipeline in 'interpreted' mode.*")


def render_network_interpretation(interpretation: Dict[str, Any], gene_symbol: str):
    """Render the Network Interpretation AI section."""
    st.markdown("##### What this means:")
    # Read directly from state — already the post-validation corrected version
    network_interp = interpretation.get('interaction_interpretation', interpretation.get('network_interpretation', ''))
    if network_interp:
        render_ai_interpretation("Network Interpretation", network_interp)
    else:
        st.caption("*No AI interpretation available. Run pipeline in 'interpreted' mode.*")


def render_experimental_relevance(interpretation: Dict[str, Any], gene_symbol: str):
    """Render the Experimental Relevance AI section."""
    st.markdown("---")
    st.markdown("### Experimental Relevance")

    # Read directly from state — already the post-validation corrected version
    exp_relevance = interpretation.get('experimental_relevance', '')
    if exp_relevance:
        render_ai_interpretation("Context-Specific Analysis", exp_relevance)
    else:
        st.info("No experimental relevance analysis available. Run pipeline in 'interpreted' mode.")


# ──────────────────────────────────────────────────────────────────────────
# Phase 3: claim-lineage UI helpers
# Operate on the iteration data emitted by src/validation/unified_validator.py.
# Iter N>0 claim dicts carry original_claim_id, parent_claim_id, final_status.
# For pre-Phase-2 snapshots, fall back to claim_id as the stable identifier.
# ──────────────────────────────────────────────────────────────────────────

_VERDICT_ICON = {"accurate": "✅", "partially_accurate": "🟡", "inaccurate": "❌"}

_VERDICT_BORDER = {
    "accurate": ("#e8f5e9", "#2e7d32"),
    "partially_accurate": ("#fff3e0", "#e65100"),
    "inaccurate": ("#ffebee", "#c62828"),
}

_GREY_BORDER = ("#f5f5f5", "#9e9e9e")

_WHY_LABEL = {
    "accurate": "Why accurate",
    "partially_accurate": "Why partial",
    "inaccurate": "Why inaccurate",
}


def _original_cid(c: Dict[str, Any]) -> Optional[str]:
    """Return lineage root id with a backward-compat fallback to claim_id."""
    return c.get("original_claim_id") or c.get("claim_id")


def _iter0_descendants(iter0_cid: str, iteration: Dict[str, Any]) -> list:
    """Non-orphan claims in `iteration` whose lineage points back to iter0_cid."""
    return [
        c for c in (iteration.get("all_claims") or [])
        if isinstance(c, dict)
        and _original_cid(c) == iter0_cid
        and c.get("final_status") != "orphan_introduced"
    ]


def classify_iter0_claim(iter0_claim: Dict[str, Any], iterations: list) -> tuple:
    """Return (kind, final_descendants) for an iter-0 claim.

    kind ∈ {"kept", "softened", "dropped"}:
      - kept: one final descendant with text equal (whitespace-trimmed)
      - softened: descendants exist but text differs or multiple descendants
      - dropped: no non-orphan descendant in the final iteration
    """
    iter0_cid = iter0_claim.get("claim_id")
    iter0_text = (iter0_claim.get("claim_text") or "").strip()
    final_iter = iterations[-1] if iterations else {}
    descendants = _iter0_descendants(iter0_cid, final_iter)
    if not descendants:
        return "dropped", []
    if (
        len(descendants) == 1
        and (descendants[0].get("claim_text") or "").strip() == iter0_text
    ):
        return "kept", descendants
    return "softened", descendants


def _final_orphans(iterations: list) -> list:
    """Final-iteration claims flagged as orphans or whose lineage never traces back to iter 0.

    Two conditions, because positional claim_ids can collide across iterations:
      (a) final_status == "orphan_introduced" — freshly introduced at the final iter.
      (b) original_claim_id not in iter-0 cids — persisted orphan from an earlier iter.
    """
    if not iterations:
        return []
    iter0 = iterations[0].get("all_claims") or []
    iter0_cids = {c.get("claim_id") for c in iter0 if isinstance(c, dict)}
    final_claims = iterations[-1].get("all_claims") or []
    return [
        c for c in final_claims
        if isinstance(c, dict)
        and (
            c.get("final_status") == "orphan_introduced"
            or _original_cid(c) not in iter0_cids
        )
    ]


def compute_lineage_summary(iterations: list) -> Dict[str, int]:
    """Counts for the header summary line across original/final/kept/softened/dropped/orphans."""
    if not iterations:
        return {"original": 0, "final": 0, "kept": 0, "softened": 0, "dropped": 0, "orphans": 0}
    iter0 = iterations[0].get("all_claims") or []
    final_claims = iterations[-1].get("all_claims") or []
    kept = softened = dropped = 0
    for c0 in iter0:
        if not isinstance(c0, dict):
            continue
        kind, _ = classify_iter0_claim(c0, iterations)
        if kind == "kept":
            kept += 1
        elif kind == "softened":
            softened += 1
        else:
            dropped += 1
    orphans = len(_final_orphans(iterations))
    return {
        "original": len(iter0),
        "final": len(final_claims),
        "kept": kept,
        "softened": softened,
        "dropped": dropped,
        "orphans": orphans,
    }


def _iter_header(iter_idx: int, is_final: bool) -> str:
    """1-based iteration label matching the iteration table."""
    label = f"Iter {iter_idx + 1}"
    if is_final:
        label += " (final)"
    return label


def _reasoning_html(reasoning: Optional[str], verdict: str) -> str:
    """Inline HTML block for reasoning — always rendered, never suppressed."""
    text = (reasoning or "").strip()
    label = _WHY_LABEL.get(verdict, "Why")
    if not text:
        return (
            '<div style="color:#9ca3af; font-style:italic; font-size:0.82rem; margin-top:6px;">'
            'No reasoning recorded.</div>'
        )
    return (
        f'<div style="color:#6b7280; font-style:italic; font-size:0.82rem; margin-top:6px;">'
        f'<strong style="color:#4b5563;">{label}:</strong> {text}</div>'
    )


def _status_badge_html(kind: str, iter0_verdict: str, final_verdict: Optional[str]) -> str:
    """Top-right pill describing the final outcome of a card."""
    icon0 = _VERDICT_ICON.get(iter0_verdict, "❓")
    iconF = _VERDICT_ICON.get(final_verdict or iter0_verdict, "❓")
    v0_label = (iter0_verdict or "unknown").replace("_", " ").upper()
    if kind == "kept":
        label = f"{icon0} KEPT · {v0_label}"
        bg, fg = "#edf7f1", "#2d7a4f"
    elif kind == "softened":
        label = f"↻ SOFTENED · {icon0} → {iconF}"
        bg, fg = "#fef3c7", "#92400e"
    elif kind == "dropped":
        label = "⚪ DROPPED"
        bg, fg = "#f5f5f5", "#6b7280"
    elif kind == "orphan":
        label = "⚠️ ORPHAN"
        bg, fg = "#fef3c7", "#92400e"
    else:
        label, bg, fg = "—", "#f5f5f5", "#6b7280"
    return (
        f'<span style="background:{bg}; color:{fg}; padding:3px 10px; border-radius:12px; '
        f'font-size:0.72rem; font-weight:700; letter-spacing:0.04em;">{label}</span>'
    )


def _render_iter_detail_line(c: Dict[str, Any], iter_idx: int, is_final: bool) -> None:
    """Verdict line + reasoning for a single claim at a single iteration."""
    verdict = (c.get("verdict") or "").lower()
    icon = _VERDICT_ICON.get(verdict, "❓")
    verdict_label = verdict.replace("_", " ").title() or "Unknown"
    claim_type = c.get("claim_type") or "unknown"
    conf = c.get("confidence")
    conf_str = f" · Confidence: {float(conf):.2f}" if isinstance(conf, (int, float)) else ""
    st.markdown(
        f'<div style="font-size:0.85rem; margin:6px 0 2px 0; color:#1f2937;">'
        f'<strong>{_iter_header(iter_idx, is_final)} verdict:</strong> '
        f'{icon} {verdict_label} &nbsp;·&nbsp; Type: {claim_type}{conf_str}</div>'
        f'{_reasoning_html(c.get("reasoning") or c.get("issue"), verdict)}',
        unsafe_allow_html=True,
    )


def _render_descendant_card(label: str, c: Dict[str, Any], iter_idx: int, is_final: bool) -> None:
    """Nested indented card inside a softened iter-0 container."""
    verdict = (c.get("verdict") or "").lower()
    icon = _VERDICT_ICON.get(verdict, "❓")
    bg, border = _VERDICT_BORDER.get(verdict, _GREY_BORDER)
    text = c.get("claim_text") or ""
    claim_type = c.get("claim_type") or "unknown"
    conf = c.get("confidence")
    conf_str = f" · Confidence: {float(conf):.2f}" if isinstance(conf, (int, float)) else ""
    verdict_label = verdict.replace("_", " ").title() or "Unknown"
    iter_label = _iter_header(iter_idx, is_final)
    st.markdown(
        f'<div style="margin:8px 0 8px 24px; padding:10px 14px; '
        f'background:{bg}; border-left:3px solid {border}; border-radius:0 6px 6px 0;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
        f'<span style="font-size:0.72rem; font-weight:700; text-transform:uppercase; '
        f'letter-spacing:0.05em; color:{border};">{label}</span>'
        f'<span style="font-size:0.72rem; color:{border};">{iter_label}</span>'
        f'</div>'
        f'<div style="font-size:0.9rem; color:#1f2937; margin-top:4px; line-height:1.45;">{text}</div>'
        f'<div style="font-size:0.82rem; margin-top:6px; color:#374151;">'
        f'{icon} <strong>{verdict_label}</strong> &nbsp;·&nbsp; Type: {claim_type}{conf_str}</div>'
        f'{_reasoning_html(c.get("reasoning") or c.get("issue"), verdict)}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_iter0_body(idx: int, iter0_claim: Dict[str, Any], iterations: list) -> None:
    """Inner body for a top-level iter-0 card (called inside container or expander)."""
    iter0_cid = iter0_claim.get("claim_id")
    iter0_verdict = (iter0_claim.get("verdict") or "").lower()
    kind, descendants = classify_iter0_claim(iter0_claim, iterations)
    final_verdict = (descendants[0].get("verdict") or "").lower() if descendants else None
    bg, border = _VERDICT_BORDER.get(iter0_verdict, _GREY_BORDER)

    # Header row: "Original Claim #N" + status badge
    st.markdown(
        f'<div style="display:flex; justify-content:space-between; align-items:center; '
        f'margin-bottom:4px;">'
        f'<div style="font-weight:600; font-size:0.95rem; color:#111827;">Original Claim #{idx}</div>'
        f'{_status_badge_html(kind, iter0_verdict, final_verdict)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Original text block with verdict-coloured left border
    st.markdown(
        f'<div style="background:{bg}; border-left:4px solid {border}; '
        f'padding:8px 12px; border-radius:0 6px 6px 0; margin:4px 0 6px 0;">'
        f'<div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; '
        f'letter-spacing:0.05em; color:{border};">Original claim</div>'
        f'<div style="font-size:0.9rem; color:#111827; margin-top:3px; line-height:1.45;">'
        f'{iter0_claim.get("claim_text","")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Iter-0 verdict + reasoning (always visible)
    _render_iter_detail_line(iter0_claim, 0, is_final=(len(iterations) == 1))

    if kind == "softened":
        # Gather intermediate + final descendants (iter>0)
        chain = []
        for n in range(1, len(iterations)):
            for c in _iter0_descendants(iter0_cid, iterations[n]):
                chain.append((n, c))
        last_idx = len(iterations) - 1
        n_final = sum(1 for n, _ in chain if n == last_idx)
        plural = f"{n_final} claims" if n_final != 1 else "1 claim"
        st.markdown(
            f'<div style="margin:10px 0 4px 0; color:#6b7280; font-size:0.88rem;">'
            f'↓ Softened and re-validated as {plural}:</div>',
            unsafe_allow_html=True,
        )
        final_letter_counter = 0
        for n, c in chain:
            is_final = n == last_idx
            if is_final:
                if n_final > 1:
                    suffix = chr(ord('a') + final_letter_counter)
                    label = f"Final #{idx}{suffix}"
                    final_letter_counter += 1
                else:
                    label = f"Final #{idx}"
            else:
                label = f"Iter {n + 1}"
            _render_descendant_card(label, c, n, is_final=is_final)
    elif kind == "kept":
        st.markdown(
            '<div style="margin-top:8px; padding:6px 10px; background:#edf7f1; '
            'color:#2d7a4f; font-size:0.82rem; border-radius:4px;">'
            '<strong>Final status:</strong> KEPT UNCHANGED</div>',
            unsafe_allow_html=True,
        )
    else:  # dropped
        st.markdown(
            '<div style="margin-top:8px; padding:6px 10px; background:#f5f5f5; '
            'color:#6b7280; font-size:0.82rem; border-radius:4px;">'
            '<strong>Final status:</strong> DROPPED — no descendant in final iteration</div>',
            unsafe_allow_html=True,
        )


def _render_iter0_card(idx: int, iter0_claim: Dict[str, Any], iterations: list) -> None:
    """Top-level card for an iter-0 claim. Uses expander when the chain spans 3+ iterations."""
    iter0_cid = iter0_claim.get("claim_id")
    kind, _ = classify_iter0_claim(iter0_claim, iterations)
    # Chain-length counter: count iterations that have a non-orphan descendant of this iter-0
    chain_hops = 1  # iter 0
    for n in range(1, len(iterations)):
        if _iter0_descendants(iter0_cid, iterations[n]):
            chain_hops += 1
    use_expander = chain_hops >= 3 and kind == "softened"

    if use_expander:
        with st.expander(
            f"Original Claim #{idx} · {kind.upper()} · {chain_hops} iterations",
            expanded=False,
        ):
            _render_iter0_body(idx, iter0_claim, iterations)
    else:
        with st.container(border=True):
            _render_iter0_body(idx, iter0_claim, iterations)


def _render_orphan_card(idx: int, orphan: Dict[str, Any], iter_idx: int) -> None:
    """Bottom-section card for a final-iter claim whose lineage doesn't trace to iter 0."""
    verdict = (orphan.get("verdict") or "").lower()
    bg, border = _GREY_BORDER
    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; align-items:center; '
            f'margin-bottom:4px;">'
            f'<div style="font-weight:600; font-size:0.95rem; color:#374151;">Orphan Claim #{idx}</div>'
            f'{_status_badge_html("orphan", verdict, verdict)}'
            f'</div>'
            f'<div style="background:{bg}; border-left:4px solid {border}; '
            f'padding:8px 12px; border-radius:0 6px 6px 0; margin:4px 0 6px 0;">'
            f'<div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; '
            f'letter-spacing:0.05em; color:{border};">Introduced during rewriting</div>'
            f'<div style="font-size:0.9rem; color:#1f2937; margin-top:3px; line-height:1.45;">'
            f'{orphan.get("claim_text","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _render_iter_detail_line(orphan, iter_idx, is_final=True)


def render_lineage_summary_line(iterations: list) -> None:
    """Single header summary line rendered above the iteration table."""
    if not iterations:
        return
    summary = compute_lineage_summary(iterations)
    st.markdown(
        f'<div style="font-size:0.88rem; color:#374151; margin:8px 0 10px 0; font-weight:500;">'
        f'<strong>{summary["original"]} original claims → {summary["final"]} final claims</strong>'
        f' &nbsp;·&nbsp; <span style="color:#2d7a4f;">{summary["kept"]} kept unchanged</span>'
        f' &nbsp;·&nbsp; <span style="color:#e65100;">{summary["softened"]} softened</span>'
        f' &nbsp;·&nbsp; <span style="color:#6b7280;">{summary["dropped"]} dropped</span>'
        f' &nbsp;·&nbsp; <span style="color:#92400e;">{summary["orphans"]} orphans introduced</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_claim_journey_cards(iterations: list, section_name: str) -> None:
    """Phase 3 claim-journey cards: iter-0 cards + orphan section. Summary line is separate."""
    if not iterations:
        return

    iter0_claims = iterations[0].get("all_claims") or []
    if not iter0_claims and not _final_orphans(iterations):
        st.caption("No claim data recorded for this section.")
        return

    for idx, c0 in enumerate(iter0_claims, 1):
        if isinstance(c0, dict):
            _render_iter0_card(idx, c0, iterations)

    orphans = _final_orphans(iterations)
    if orphans:
        st.markdown(
            '<div style="margin:16px 0 8px 0; font-weight:600; font-size:0.95rem; color:#92400e;">'
            'Orphaned claims — introduced during rewriting</div>',
            unsafe_allow_html=True,
        )
        last_iter_idx = len(iterations) - 1
        for idx, oc in enumerate(orphans, 1):
            _render_orphan_card(idx, oc, last_iter_idx)


def _get_validation_snapshot(gene_symbol: str, state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the state-native validation snapshot for a gene (or None)."""
    if not state:
        return None
    vr = state.get("validation_results", {})
    if isinstance(vr, dict):
        return vr.get(gene_symbol)
    return None


def render_validation_results(gene_symbol: str, state: Optional[Dict[str, Any]] = None):
    """
    Render validation status from state.validation_results (state-native).

    Shows per-section expanders with iteration tables, claim details,
    and a bottom "changes table" summarising all refinement changes.
    """
    snapshot = _get_validation_snapshot(gene_symbol, state)

    st.markdown("---")
    st.subheader("🔍 Validation Results")

    low_confidence = False
    if state:
        low_confidence = (state.get("low_confidence_flags") or {}).get(gene_symbol, False)

    if not snapshot:
        st.info("No validation data available for this gene.")
        st.markdown("---")
        return

    # ── Top-level metrics ──────────────────────────────────────────────
    overall_accuracy = snapshot.get("overall_accuracy")
    hallucinations = snapshot.get("total_hallucinations_detected", 0)
    corrections = snapshot.get("total_corrections_made", 0)
    overall_status = snapshot.get("status", "")

    sections = snapshot.get("sections", {})

    # Count partial claims across all sections
    partial_total = sum(
        s.get("claims_summary", {}).get("partial_count", 0)
        for s in sections.values()
        if isinstance(s, dict)
    )

    metric_cols = 3 if partial_total == 0 else 4
    cols = st.columns(metric_cols)
    cols[0].metric(
        "Overall Accuracy",
        f"{overall_accuracy:.1f}%" if overall_accuracy is not None else "N/A",
    )
    cols[1].metric("Hallucinations Detected", hallucinations)
    cols[2].metric("Total Corrections", corrections)
    if partial_total > 0:
        cols[3].metric("Partially Accurate Claims", partial_total)

    # Section count
    validated_count = sum(
        1 for s in sections.values()
        if isinstance(s, dict) and s.get("status") not in ("no_claims", None)
    )
    if validated_count > 0:
        st.caption(f"*Final accuracy across {validated_count} validated section(s)*")

    if low_confidence and overall_accuracy is not None and overall_accuracy < 85.0:
        st.warning(
            "Accuracy remained below threshold after maximum refinement passes "
            "— treat interpretations with caution."
        )

    if not sections:
        st.markdown("---")
        return

    # ── Per-section expanders ──────────────────────────────────────────
    st.markdown("### Validation by Section")

    def _badge_style(kind: str) -> tuple:
        palette = {
            "success": ("#edf7f1", "#2d7a4f"),
            "warning": ("#fdf6e3", "#8a6010"),
            "error": ("#fdf0f0", "#a03030"),
            "info": ("#eef4fb", "#2a5a8a"),
        }
        return palette.get(kind, ("#f7f6f2", "#555555"))

    for section_name, section_data in sections.items():
        if not isinstance(section_data, dict):
            continue

        display_name = section_data.get("display_name", section_name.replace("_", " ").title())
        status = section_data.get("status", "")
        iterations = section_data.get("iterations", [])
        final_acc = section_data.get("final_accuracy")
        error_message = section_data.get("error_message")

        # ── No-claims shortcut ─────────────────────────────────────
        if status == "no_claims" or (not iterations and error_message == "no_claims_extracted"):
            with st.expander(f"— {display_name} — no claims extracted", expanded=False):
                st.caption("This section had no substantive claims to validate (empty or too brief).")
            continue

        if not iterations:
            with st.expander(f"— {display_name} — not validated", expanded=False):
                if error_message:
                    st.caption(f"Error: {error_message}")
                else:
                    st.caption("No iteration data available.")
            continue

        # ── Determine icon / label ─────────────────────────────────
        if final_acc is None:
            acc_val = 0.0
            icon = "❓"
            label = f"{icon} {display_name} — N/A"
            expanded = False
        else:
            acc_val = float(final_acc)
            icon = "✅" if acc_val >= ACCURACY_THRESHOLD else ("⚠️" if acc_val >= 70 else "❌")
            label = f"{icon} {display_name} — {acc_val:.0f}%"
            expanded = acc_val < 85 and bool(iterations)

        final_iteration = iterations[-1] if iterations else {}

        with st.expander(label, expanded=expanded):
            st.progress(max(0.0, min(acc_val / 100.0, 1.0)))

            # ── Existence check shortcut ───────────────────────────
            if final_iteration.get("is_existence_check"):
                st.caption("✓ Auto-validated via database existence check.")
                all_claims = final_iteration.get("all_claims", [])
                claim_text = (
                    all_claims[0].get("claim_text", "") if all_claims
                    else final_iteration.get("summary_text", "")
                )
                if claim_text:
                    st.info(claim_text)
                continue

            # ── Summary before / after ─────────────────────────────
            original_summary = section_data.get("original_summary", "")
            final_summary = section_data.get("final_summary", "")
            if original_summary and final_summary and original_summary.strip() != final_summary.strip():
                with st.expander("📝 Summary before → after", expanded=False):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Original**")
                        st.caption(original_summary[:500] + ("..." if len(original_summary) > 500 else ""))
                    with col_b:
                        st.markdown("**Final (corrected)**")
                        st.caption(final_summary[:500] + ("..." if len(final_summary) > 500 else ""))

            # ── Lineage summary line (Phase 3) ─────────────────────
            render_lineage_summary_line(iterations)

            # ── Iteration history table ────────────────────────────
            rows = []
            total_iters = len(iterations)
            for idx, it in enumerate(iterations):
                counts = it.get("counts", {})
                # Handle both formats: unified validator uses "counts" dict and "claims_extracted",
                # while literature findings uses "all_claims", "accurate_claims", etc. lists
                claims = it.get("claims_extracted", 0) or len(it.get("all_claims", []))
                accurate = counts.get("accurate", 0) or len(it.get("accurate_claims", []))
                partial = counts.get("partial", 0) or len(it.get("partially_accurate_claims", []))
                inaccurate = counts.get("inaccurate", 0) or len(it.get("inaccurate_claims", []))
                acc_raw = it.get("accuracy_percentage")
                acc_pct = float(acc_raw) if acc_raw is not None else None

                is_final = idx == total_iters - 1
                issues = inaccurate + partial
                if is_final and acc_pct is not None and acc_pct == 0 and accurate == claims and issues == 0 and claims > 0:
                    acc_pct = float(final_acc) if final_acc is not None else None

                acc_kind = (
                    ("success" if acc_pct >= ACCURACY_THRESHOLD else ("warning" if acc_pct >= 70 else "error"))
                    if acc_pct is not None else "neutral"
                )
                acc_bg, acc_fg = _badge_style(acc_kind)

                # Determine action badge
                next_has_changes = False
                if idx < total_iters - 1:
                    next_has_changes = bool(iterations[idx + 1].get("refinement_changes"))
                partial_only = partial > 0 and inaccurate == 0

                if next_has_changes:
                    action = ("refined", "warning")
                elif is_final and issues == 0 and acc_pct >= ACCURACY_THRESHOLD:
                    action = ("✓ passed", "success")
                elif is_final and partial_only and acc_pct < ACCURACY_THRESHOLD:
                    action = ("needs review", "warning")
                elif partial_only:
                    action = ("softened", "info")
                elif issues == 0:
                    action = ("✓ passed", "warning")
                else:
                    action = ("needs review", "error")

                action_bg, action_fg = _badge_style(action[1])
                iter_label = f"Iter {idx + 1}"
                if is_final:
                    iter_label += " (final)"

                rows.append(
                    f"<tr>"
                    f"<td>{iter_label}</td>"
                    f"<td>{claims}</td>"
                    f"<td>{accurate}</td>"
                    f"<td>{partial}</td>"
                    f"<td>{inaccurate}</td>"
                    f'<td style="background:{acc_bg};color:{acc_fg};font-weight:600;">{"N/A" if acc_pct is None else f"{acc_pct:.0f}%"}</td>'
                    f'<td style="background:{action_bg};color:{action_fg};font-weight:600;">{action[0]}</td>'
                    f"</tr>"
                )

            if iterations:
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
                        f"<td>0</td>"
                        f"<td>0</td>"
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

            # ── Claim journey (Phase 3) ────────────────────────────
            st.markdown("#### Claim journey")
            render_claim_journey_cards(iterations, section_name)

            # ── Section footer ─────────────────────────────────────
            _is_resolved = final_acc is not None and float(final_acc) >= ACCURACY_THRESHOLD
            if _is_resolved:
                _first_iter_acc_raw = iterations[0].get("accuracy_percentage") if iterations else None
                _first_passed = _first_iter_acc_raw is not None and float(_first_iter_acc_raw) >= ACCURACY_THRESHOLD
                if len(iterations) <= 1 or _first_passed:
                    st.caption("✓ All claims passed on first check.")
                else:
                    st.caption("✓ All issues resolved during refinement.")
            else:
                _final_all_c = final_iteration.get("all_claims", []) or []
                _n_unresolved = sum(
                    1 for c in _final_all_c
                    if isinstance(c, dict) and c.get("verdict") in ("partially_accurate", "inaccurate")
                )
                if _n_unresolved > 0:
                    st.caption(f"⚠️ {_n_unresolved} claim(s) could not be fully verified.")
                else:
                    st.caption("✓ All claims passed on first check.")

    st.markdown("---")


# ============================================================================
# MAIN GENE RENDERING FUNCTION WITH PROGRESSIVE LOADING
# ============================================================================

def render_individual_genes(state: Dict[str, Any]):
    """
    Render the Individual Genes tab with progressive loading.

    Phase 1: All factual data renders immediately
    Phase 2: AI interpretations fill into placeholders
    """
    st.markdown("## Individual Gene Analysis")

    # Get available genes
    all_gene_data = state.get('all_gene_data', {})
    gene_interpretations_data = state.get('gene_interpretations', {})
    gene_top_papers = state.get('gene_top_papers', {})

    available_genes = list(set(
        list(all_gene_data.keys()) +
        list(gene_interpretations_data.keys())
    ))

    if not available_genes:
        st.info("No gene data available. Load a completed pipeline run or execute the pipeline.")
        return

    # Gene selector
    selected_gene = st.selectbox(
        "Select a gene to explore",
        options=sorted(available_genes),
        key="gene_selector"
    )

    if not selected_gene:
        return

    # Get data for selected gene
    gene_data = all_gene_data.get(selected_gene, {})
    interpretation = gene_interpretations_data.get(selected_gene, {})
    if isinstance(interpretation, str):
        interpretation = {}  # Handle legacy string interpretations
    papers_data = gene_top_papers.get(selected_gene, {})
    candidates_data = state.get('gene_literature_candidates', {}).get(selected_gene, {})

    # ========================================================================
    # PHASE 1: FACTUAL DATA (RENDERS IMMEDIATELY)
    # ========================================================================

    # 1. Gene Identity (factual)
    render_gene_identity(gene_data, selected_gene)

    # Placeholder for AI Gene Summary
    ai_summary_placeholder = st.empty()

    # 2. GO Terms (factual)
    render_go_terms(gene_data)

    # Placeholder for Functional Interpretation
    functional_interp_placeholder = st.empty()

    # 4. Expression Profile (factual)
    render_expression_profile(gene_data)

    # Placeholder for Expression Interpretation
    expression_interp_placeholder = st.empty()

    # 5. Protein Interactions (factual)
    render_protein_interactions(gene_data)

    # Placeholder for Network Interpretation
    network_interp_placeholder = st.empty()

    # 6. Literature (factual)
    render_literature(selected_gene, papers_data, candidates_data)

    # ========================================================================
    # PHASE 2: AI CONTENT (FILLS INTO PLACEHOLDERS)
    # ========================================================================

    # Check if there's any AI content to load
    has_ai_content = bool(
        state.get('gene_summaries', {}).get(selected_gene) or
        interpretation.get('functional_summary') or
        interpretation.get('expression_interpretation') or
        interpretation.get('interaction_interpretation')
    )

    # Fill in AI content (appears after factual data is visible)
    # Gene Summary
    with ai_summary_placeholder.container():
        render_ai_gene_summary(state, selected_gene, interpretation)

    # Functional Interpretation (after GO Terms)
    with functional_interp_placeholder.container():
        if gene_data.get('go_terms'):  # Only show if GO terms exist
            render_functional_interpretation(interpretation, selected_gene)

    # Expression Interpretation (after Expression)
    with expression_interp_placeholder.container():
        if gene_data.get('expression'):  # Only show if expression data exists
            render_expression_interpretation(interpretation, selected_gene)

    # Network Interpretation (after Interactions)
    with network_interp_placeholder.container():
        if gene_data.get('interactions'):  # Only show if interactions exist
            render_network_interpretation(interpretation, selected_gene)

    render_validation_results(selected_gene, state)
