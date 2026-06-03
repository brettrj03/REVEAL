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
import json
import tempfile
import os
from datetime import datetime

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


class _StateEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime and other non-serializable types."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle other non-serializable types by converting to string
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def _save_state_to_disk(state_dict):
    """Serialize pipeline state to a temporary JSON file and return the file path."""
    # Clean up old state file if it exists
    old_path = st.session_state.get('pipeline_state_path')
    if old_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
        except Exception:
            pass

    # Create new temporary file
    fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='reveal_state_')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(state_dict, f, cls=_StateEncoder)
        return temp_path
    except Exception as e:
        # Note: fd is already closed by os.fdopen context manager
        # Clean up the temp file if serialization failed
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise e


def _load_state_from_disk():
    """Load pipeline state from disk using the stored file path."""
    state_path = st.session_state.get('pipeline_state_path')
    if not state_path or not os.path.exists(state_path):
        return {}

    try:
        with open(state_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_run_permanently(state_dict):
    """Save a completed pipeline run to the permanent saved_runs directory.

    Handles files of any size, including large runs (100MB+).

    Args:
        state_dict: The pipeline state dictionary to save

    Returns:
        str: Path to the saved file

    Raises:
        Exception: If save fails, with detailed error message
    """
    print("\n[SAVE] Starting permanent save of pipeline run...")

    # Create saved_runs directory if it doesn't exist
    saved_runs_dir = os.path.expanduser("~/Desktop/REVEAL-backup_2/saved_runs")
    os.makedirs(saved_runs_dir, exist_ok=True)
    print(f"[SAVE] Save directory: {saved_runs_dir}")

    # Extract gene names from state
    genes = []
    if 'genes_to_process' in state_dict and state_dict['genes_to_process']:
        genes = state_dict['genes_to_process'][:3]  # Take first 3 genes for filename
    elif 'gene_mapping' in state_dict and state_dict['gene_mapping']:
        genes = list(state_dict['gene_mapping'].keys())[:3]

    # Create filename with genes and timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if genes:
        gene_str = "_".join(genes)
        # Clean gene string for filename (remove special characters)
        gene_str = "".join(c for c in gene_str if c.isalnum() or c in ('_', '-'))
        filename = f"{gene_str}_{timestamp}.json"
    else:
        filename = f"run_{timestamp}.json"

    print(f"[SAVE] Filename: {filename}")
    print(f"[SAVE] Genes in run: {genes if genes else 'unknown'}")

    # Save the state - no file size limit
    filepath = os.path.join(saved_runs_dir, filename)
    try:
        print(f"[SAVE] Writing JSON to {filepath}...")
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, cls=_StateEncoder, indent=2)

        # Get file size for logging
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[SAVE] ✓ Successfully saved {file_size_mb:.1f}MB to {filename}")

        return filepath
    except Exception as e:
        print(f"[SAVE] ✗ FAILED to save run: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main application."""
    # Apply custom CSS styles
    apply_styles()

    # Initialise session state
    if 'status' not in st.session_state:
        st.session_state['status'] = 'Ready'
    if 'pipeline_state_path' not in st.session_state:
        st.session_state['pipeline_state_path'] = None
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

            # Store results - serialize state to disk to prevent memory overload
            state_path = _save_state_to_disk(state_dict)
            st.session_state['pipeline_state_path'] = state_path
            st.session_state['final_report'] = final_report
            st.session_state['persistence_dir'] = persistence_dir

            # Save run permanently for future access
            # This is called for ALL completed runs regardless of size or gene count
            try:
                saved_path = _save_run_permanently(state_dict)
                st.session_state['last_saved_run'] = saved_path
                print(f"[SAVE] Run saved to: {saved_path}")
            except Exception as e:
                # Don't fail the whole run if save fails, but log prominently
                error_msg = f"Could not save run permanently: {type(e).__name__}: {e}"
                print(f"\n{'='*80}")
                print(f"[SAVE ERROR] {error_msg}")
                print(f"{'='*80}\n")
                st.warning(error_msg)

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

    # Load state data from disk (prevents memory overload with large gene sets)
    state = _load_state_from_disk()

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
