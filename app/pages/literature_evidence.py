"""
Literature Evidence page for REVEAL.
"""

import streamlit as st
from typing import Dict, Any


def render_literature_evidence(state: Dict[str, Any]):
    """Render the Literature Evidence tab with top papers and expandable candidates."""
    st.markdown("## Literature Evidence")

    gene_papers = state.get('gene_top_papers', {})
    gene_candidates = state.get('gene_literature_candidates', {})

    if not gene_papers:
        st.info("No literature data available. Run the pipeline to fetch papers from PubMed.")
        return

    # Gene selector for papers
    selected_gene = st.selectbox(
        "Select gene to view papers",
        options=list(gene_papers.keys()),
        key="lit_gene_selector"
    )

    if selected_gene:
        papers_data = gene_papers.get(selected_gene, {})
        candidates_data = gene_candidates.get(selected_gene, {})

        # Get top papers
        top_papers = papers_data.get('top_papers', [])
        total_candidates = papers_data.get('total_candidates', candidates_data.get('total_count', len(top_papers)))
        fetch_strategy = papers_data.get('fetch_strategy', candidates_data.get('fetch_strategy', 'standard'))
        tier_used = papers_data.get('tier_used', candidates_data.get('tier_used', ''))
        query_used = papers_data.get('query_used', candidates_data.get('query_used', 'N/A'))

        if top_papers:
            st.subheader("Top Ranked Papers")

            for i, paper in enumerate(top_papers, 1):
                title = paper.get('title', paper.get('Title', 'Untitled'))
                pmid = paper.get('pmid', paper.get('PMID'))
                relevance = paper.get('relevance_score', 0)
                stars = "★" * int(relevance) + "☆" * (5 - int(relevance)) if relevance else ""

                with st.expander(f"{i}. {title} {stars}", expanded=(i <= 3)):
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        # Authors
                        authors = paper.get('authors', paper.get('Authors', []))
                        if authors:
                            if isinstance(authors, list):
                                author_str = ", ".join(authors[:3])
                                if len(authors) > 3:
                                    author_str += " et al."
                            else:
                                author_str = str(authors)
                            st.markdown(f"**Authors:** {author_str}")

                        # Journal and year
                        journal = paper.get('journal', paper.get('Journal', 'Unknown'))
                        year = paper.get('year', paper.get('Year', 'N/A'))
                        st.markdown(f"**Journal:** {journal} ({year})")

                        # Key finding
                        finding = paper.get('key_finding', '')
                        if finding:
                            st.info(f"**Key Finding:** {finding}")

                        # Abstract
                        abstract = paper.get('abstract', paper.get('Abstract', ''))
                        if abstract:
                            st.markdown("**Abstract:**")
                            st.markdown(abstract[:500] + "..." if len(str(abstract)) > 500 else abstract)

                    with col2:
                        if relevance:
                            st.metric("Relevance", f"{relevance}/5")

                        if pmid:
                            st.markdown(f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")

            # Show all candidates in expander
            all_candidates = candidates_data.get('candidate_papers', [])
            if all_candidates and len(all_candidates) > len(top_papers):
                top_pmids = set(p.get('pmid') for p in top_papers)
                remaining = [p for p in all_candidates if p.get('pmid') not in top_pmids]

                if remaining:
                    with st.expander(f"Show all {len(remaining)} additional candidates"):
                        st.caption("These papers were considered but ranked lower in relevance.")
                        for i, paper in enumerate(remaining, len(top_papers) + 1):
                            pmid = paper.get('pmid')
                            title = paper.get('title', 'Untitled')
                            year = paper.get('year', 'N/A')
                            relevance = paper.get('relevance_score', 0)
                            stars = "★" * int(relevance) + "☆" * (5 - int(relevance)) if relevance else ""

                            if pmid:
                                st.markdown(f"{i}. [{title}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) ({year}) {stars}")
                            else:
                                st.markdown(f"{i}. {title} ({year}) {stars}")

            # Search details
            with st.expander("Search details"):
                if query_used and query_used != 'N/A':
                    st.markdown("**Query used:**")
                    st.code(query_used, language="text")
                if tier_used:
                    st.markdown(f"**Query tier:** {tier_used}")
                st.markdown(f"**Ranking method:** {papers_data.get('ranking_method', 'LLM')}")

        else:
            st.info("No papers retrieved for this gene")
