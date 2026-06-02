"""
REVEAL - Streamlit Interface

REVEAL: Retrieval and Evidence-based Validated Interpretation Analysis for gene Lists
A researcher-friendly interface for exploring gene annotation results.
"""

import warnings

# Suppress pydantic_graph library-level serialisation warnings (library issue, not application code)
warnings.filterwarnings("ignore", message=".*Expected.*NodeSnapshot.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Expected.*none.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*", category=UserWarning)

import streamlit as st

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="REVEAL",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app.styles import apply_styles
from app.sidebar import render_sidebar, run_pipeline_with_progress
from app.components.shared import render_ncbi_credentials_warning, check_openai_api_key, validate_openai_api_key
from app.pages.executive_summary import render_executive_summary
from app.pages.individual_genes import render_individual_genes
from app.pages.network_analysis import render_network_analysis
from app.pages.literature_evidence import render_literature_evidence
from app.pages.methods_diagnostics import render_methods_diagnostics


def main():
    """Main application."""
    # Apply custom CSS styles
    apply_styles()

    # Initialise session state
    if 'status' not in st.session_state:
        st.session_state['status'] = 'Ready'
    if 'pipeline_state' not in st.session_state:
        st.session_state['pipeline_state'] = {}
    if 'selected_model' not in st.session_state:
        st.session_state['selected_model'] = 'gpt-4.1-mini'
    if 'running' not in st.session_state:
        st.session_state['running'] = False

    # Check NCBI credentials and show warning if needed
    render_ncbi_credentials_warning()

    # Render sidebar (inputs)
    render_sidebar()

    # Check if we need to run the pipeline
    if st.session_state.get('status') == 'Running' and 'run_query' in st.session_state:
        query = st.session_state['run_query']
        mode = st.session_state.get('run_mode', 'factual')
        model = st.session_state.get('run_model', 'gpt-4.1-mini')

        # Defensive check: verify API key for interpreted mode
        # Guards against stale session state bypassing button handler validation
        if mode == 'interpreted':
            api_key_check = check_openai_api_key()
            if not api_key_check["has_key"]:
                # API key missing - cancel the run
                st.session_state['status'] = 'Error'
                st.session_state['running'] = False

                # Clean up stale run parameters
                if 'run_query' in st.session_state:
                    del st.session_state['run_query']
                if 'run_mode' in st.session_state:
                    del st.session_state['run_mode']
                if 'run_model' in st.session_state:
                    del st.session_state['run_model']

                # Show error to user
                st.error(
                    "⚠️ **Run cancelled: OpenAI API key required**\n\n"
                    "Interpreted mode requires an OpenAI API key, but none was found. "
                    "This can happen if the API key was removed after a run was started.\n\n"
                    "Please add your API key to the .env file:\n\n"
                    "```\nOPENAI_API_KEY=sk-your-actual-key-here\n```\n\n"
                    "Get your API key from: https://platform.openai.com/api-keys\n\n"
                    "**After adding the key, restart Streamlit** (Ctrl+C, then run `streamlit run streamlit_app.py` again)."
                )
                return

            # Validate API key with a test call
            with st.spinner("Verifying API key..."):
                validation_result = validate_openai_api_key()

            if not validation_result["valid"]:
                # API key validation failed - cancel the run
                st.session_state['status'] = 'Error'
                st.session_state['running'] = False

                # Clean up stale run parameters
                if 'run_query' in st.session_state:
                    del st.session_state['run_query']
                if 'run_mode' in st.session_state:
                    del st.session_state['run_mode']
                if 'run_model' in st.session_state:
                    del st.session_state['run_model']

                # Show error to user with specific validation error
                st.error(
                    f"⚠️ **Run cancelled: API key validation failed**\n\n"
                    f"{validation_result['error']}\n\n"
                    f"Please check your .env file and ensure OPENAI_API_KEY contains a valid API key.\n\n"
                    f"Get your API key from: https://platform.openai.com/api-keys\n\n"
                    f"**After updating the key, restart Streamlit** (Ctrl+C, then run `streamlit run streamlit_app.py` again)."
                )
                return

        # Main header while running
        st.markdown('<p class="main-header">REVEAL</p>', unsafe_allow_html=True)

        # Create progress container in main area
        progress_container = st.container()

        try:
            # Run pipeline with progress tracking
            final_report, state_dict, persistence_dir = run_pipeline_with_progress(
                query, mode, progress_container, model
            )

            # Store results
            st.session_state['pipeline_state'] = state_dict
            st.session_state['final_report'] = final_report
            st.session_state['persistence_dir'] = persistence_dir
            st.session_state['status'] = 'Complete'
            st.session_state['running'] = False

            # Clean up run params
            del st.session_state['run_query']
            for key in ('run_mode', 'run_model'):
                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

        except Exception as e:
            st.session_state['status'] = 'Error'
            st.session_state['running'] = False
            st.session_state['error'] = str(e)

            # Clean up run params (same as success path)
            if 'run_query' in st.session_state:
                del st.session_state['run_query']
            for key in ('run_mode', 'run_model'):
                if key in st.session_state:
                    del st.session_state[key]

            st.error(f"Pipeline failed: {e}")
            import traceback
            st.code(traceback.format_exc())
        return

    # Load state data (use 'or {}' to handle None values)
    state = st.session_state.get('pipeline_state') or {}

    # Main header
    st.markdown('<p class="main-header">REVEAL</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Retrieval and Evidence-based Validated Interpretation Analysis for gene Lists</p>', unsafe_allow_html=True)

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Summary",
        "Individual Genes",
        "Network Analysis",
        "Literature Evidence",
        "Methods & Diagnostics"
    ])

    with tab1:
        render_executive_summary(state)

    with tab2:
        render_individual_genes(state)

    with tab3:
        render_network_analysis(state)

    with tab4:
        render_literature_evidence(state)

    with tab5:
        render_methods_diagnostics(state)


if __name__ == "__main__":
    main()
