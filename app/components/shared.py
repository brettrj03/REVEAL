"""
Shared UI components for REVEAL.
"""

import streamlit as st
import os
from typing import Any
from openai import OpenAI

from src.integrations.pubmed_client import check_ncbi_credentials
from src.utils.env import load_project_env

# Ensure .env is loaded
load_project_env()


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


def check_openai_api_key():
    """Check if OpenAI API key is properly configured.

    Returns:
        dict with keys:
        - has_key: bool - True if API key is configured
        - warning_message: str - Warning message if key is missing or empty
    """
    api_key = os.getenv("OPENAI_API_KEY", "")

    # Check if API key is set but empty
    key_set_but_empty = "OPENAI_API_KEY" in os.environ and not os.environ["OPENAI_API_KEY"].strip()

    # Check if API key is missing entirely
    key_missing = not api_key or not api_key.strip()

    # Check if API key is a placeholder value
    is_placeholder = False
    if api_key and api_key.strip():
        key_lower = api_key.lower()
        is_placeholder = (
            key_lower.startswith('sk-your') or
            key_lower == 'sk-your-openai-api-key-here'
        )

    warning_message = ""
    if key_set_but_empty:
        warning_message = (
            "OPENAI_API_KEY is set but empty. "
            "Please add your OpenAI API key to the .env file."
        )
    elif key_missing:
        warning_message = (
            "OPENAI_API_KEY is not set. "
            "Please add your OpenAI API key to the .env file."
        )
    elif is_placeholder:
        warning_message = (
            "OPENAI_API_KEY is still set to the placeholder value. "
            "Please replace it with your real OpenAI API key."
        )

    return {
        "has_key": bool(api_key and api_key.strip() and not is_placeholder),
        "warning_message": warning_message
    }


def validate_openai_api_key():
    """Test the OpenAI API key with a real API call.

    Returns:
        dict with keys:
        - valid: bool - True if the API key works
        - error: str or None - Error message if validation failed
    """
    api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key or not api_key.strip():
        return {
            "valid": False,
            "error": "No API key configured."
        }

    try:
        # Make a lightweight test call with short timeout
        client = OpenAI(api_key=api_key, timeout=5.0)
        client.models.list()
        return {
            "valid": True,
            "error": None
        }
    except Exception as e:
        # Check for authentication errors (401)
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str or "invalid" in error_str:
            return {
                "valid": False,
                "error": "OpenAI API key is invalid or has been revoked."
            }
        else:
            # Other errors (network, timeout, etc.)
            return {
                "valid": False,
                "error": f"Could not verify OpenAI API key: {str(e)}"
            }


def render_openai_api_key_warning():
    """Display a warning banner if OpenAI API key is not configured."""
    key_check = check_openai_api_key()
    if key_check["warning_message"]:
        st.warning(
            "⚠️ " + key_check["warning_message"] + " "
            "Get your API key from https://platform.openai.com/api-keys and add it to your .env file as: "
            "OPENAI_API_KEY=sk-your-actual-key-here. "
            "After adding the key, restart Streamlit (Ctrl+C, then run `streamlit run streamlit_app.py` again)."
        )
