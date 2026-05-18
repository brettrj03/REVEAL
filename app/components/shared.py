"""
Shared UI components for REVEAL.
"""

import streamlit as st
from typing import Any

from src.integrations.pubmed_client import check_ncbi_credentials


def get_source_text(data_with_provenance: Any, default_source: str = "Database") -> str:
    """Extract source attribution text from data with provenance."""
    if isinstance(data_with_provenance, dict):
        prov = data_with_provenance.get('provenance', {})
        if isinstance(prov, dict):
            source = prov.get('source_release', default_source)
            retrieved = prov.get('retrieved_at', '')
            if retrieved:
                date = retrieved.split('T')[0]
                return f"Source: {source} (retrieved {date})"
            return f"Source: {source}"
    return f"Source: {default_source}"


def render_source(source_text: str):
    """Render source attribution in small gray text."""
    st.caption(f"*{source_text}*")


def render_ai_interpretation(title: str, content: str):
    """Render an AI-generated interpretation with clear labeling."""
    if content:
        st.markdown(f"**{title}** *(AI-generated)*")
        st.markdown(
            f'<div style="background-color: #d4edda; border-left: 4px solid #28a745; '
            f'padding: 1rem 1rem 1rem 1.5rem; border-radius: 4px; line-height: 1.6;">'
            f'{content}</div>',
            unsafe_allow_html=True
        )


def render_evidence_code_guide():
    """Render collapsible evidence code guide."""
    with st.expander("ℹ️ Evidence Code Guide", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Strong Evidence (Experimental)**")
            st.markdown("- **EXP**: Inferred from Experiment")
            st.markdown("- **IDA**: Inferred from Direct Assay")
            st.markdown("- **IPI**: Inferred from Physical Interaction")
            st.markdown("- **IMP**: Inferred from Mutant Phenotype")
            st.markdown("- **IGI**: Inferred from Genetic Interaction")
            st.markdown("- **IEP**: Inferred from Expression Pattern")

        with col2:
            st.markdown("**Moderate Evidence (Computational)**")
            st.markdown("- **ISS**: Inferred from Sequence Similarity")
            st.markdown("- **ISO**: Inferred from Sequence Orthology")
            st.markdown("- **IBA**: Inferred from Biological Aspect of Ancestor")
            st.markdown("- **TAS**: Traceable Author Statement")
            st.markdown("- **IC**: Inferred by Curator")
            st.markdown("- **IEA**: Inferred from Electronic Annotation")

        st.markdown("[Full evidence code reference →](https://geneontology.org/docs/guide-go-evidence-codes/)")


def render_ncbi_credentials_warning():
    """Display a warning banner if NCBI credentials are misconfigured."""
    creds = check_ncbi_credentials()
    if creds["warning_message"]:
        st.warning(
            "⚠️ " + creds["warning_message"] + " "
            "Please update your .env file with your real email address. "
            "If you don't have an NCBI API key, remove or comment out the "
            "NCBI_API_KEY line — don't leave it blank."
        )
