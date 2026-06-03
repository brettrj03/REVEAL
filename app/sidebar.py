"""
Sidebar and pipeline execution functions for REVEAL.
"""

import streamlit as st
import json
import asyncio
import hashlib
import re
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import fields, is_dataclass

from pydantic import BaseModel

from src.graph.gene_graph import run_stateful_pipeline
from src.models.models import ExperimentContext
from app.components.shared import check_openai_api_key


def load_state_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load and parse a pipeline state file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # State files are arrays of node execution snapshots
        # Get the final state from the last entry
        if isinstance(data, list) and len(data) > 0:
            return data[-1].get('state', {})
        elif isinstance(data, dict):
            return data.get('state', data)
        return None
    except Exception as e:
        st.error(f"Error loading state file: {e}")
        return None


def get_available_runs() -> list:
    """Get list of available pipeline runs."""
    results_dir = Path("results/stateful_pipeline")
    if not results_dir.exists():
        return []

    runs = []
    for run_dir in sorted(results_dir.iterdir(), reverse=True):
        if run_dir.is_dir():
            state_file = run_dir / "state.json"
            if state_file.exists():
                runs.append({
                    'name': run_dir.name,
                    'path': str(state_file)
                })
    return runs


def get_saved_runs() -> list:
    """Get list of permanently saved pipeline runs.

    Returns:
        List of dicts with 'display_name', 'genes', 'date', 'path'
    """
    saved_runs_dir = Path(os.path.expanduser("~/Desktop/REVEAL-backup_2/saved_runs"))
    if not saved_runs_dir.exists():
        return []

    runs = []
    for file_path in sorted(saved_runs_dir.glob("*.json"), reverse=True):
        filename = file_path.stem  # filename without .json extension

        # Try to parse filename format: GENE1_GENE2_YYYY-MM-DD_HH-MM-SS
        # or: run_YYYY-MM-DD_HH-MM-SS
        parts = filename.split('_')

        # Extract date and time (last 3 parts: YYYY-MM-DD, HH-MM-SS)
        if len(parts) >= 3:
            try:
                date_str = parts[-2]  # YYYY-MM-DD
                time_str = parts[-1]  # HH-MM-SS
                datetime_str = f"{date_str} {time_str.replace('-', ':')}"

                # Extract gene names (everything before the date)
                gene_parts = parts[:-2]
                if gene_parts and gene_parts[0] != 'run':
                    genes = ', '.join(gene_parts)
                    display_name = f"{genes} ({date_str})"
                else:
                    genes = "Unknown"
                    display_name = f"Run {date_str} {time_str}"

                runs.append({
                    'display_name': display_name,
                    'genes': genes,
                    'date': date_str,
                    'path': str(file_path)
                })
            except Exception:
                # If parsing fails, just use filename
                runs.append({
                    'display_name': filename,
                    'genes': 'Unknown',
                    'date': 'Unknown',
                    'path': str(file_path)
                })
        else:
            # Fallback for unexpected format
            runs.append({
                'display_name': filename,
                'genes': 'Unknown',
                'date': 'Unknown',
                'path': str(file_path)
            })

    return runs


def dataclass_to_dict(obj) -> Dict[str, Any]:
    """Convert a dataclass to a dict, handling nested Pydantic models and dataclasses."""

    def convert_value(value):
        """Recursively convert a value to a JSON-serializable format."""
        if isinstance(value, BaseModel):
            return value.model_dump(mode='json')
        elif is_dataclass(value) and not isinstance(value, type):
            return dataclass_to_dict(value)
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_value(item) for item in value]
        else:
            return value

    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = convert_value(value)
        return result
    elif isinstance(obj, BaseModel):
        return obj.model_dump(mode='json')
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return {}


async def run_pipeline_async(
    query: str,
    output_mode: str,
    model: str = "gpt-4.1-mini",
    progress_callback=None
) -> tuple[str, Dict[str, Any]]:
    """Run the pipeline asynchronously."""

    # Apply selected model before any pipeline calls
    from src.config import set_active_model
    set_active_model(model)

    # Create experiment context
    experiment_context = ExperimentContext(
        organism="human",
        comparison="Gene function analysis"
    )

    # Create persistence directory
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    persistence_dir = f"results/stateful_pipeline/run_{timestamp}_query_{query_hash}"

    # Run pipeline
    final_report, pipeline_state = await run_stateful_pipeline(
        user_query=query,
        experiment_context=experiment_context,
        output_mode=output_mode,
        persistence_dir=persistence_dir
    )

    # Convert state to dict for storage
    state_dict = dataclass_to_dict(pipeline_state)

    return final_report, state_dict, persistence_dir


def run_pipeline_sync(query: str, output_mode: str, model: str = "gpt-4.1-mini") -> tuple[str, Dict[str, Any], str]:
    """Synchronous wrapper to run the async pipeline."""
    return asyncio.run(run_pipeline_async(query, output_mode, model))


# Pipeline node display names (in execution order)
# Format: (node_id, user_friendly_status_message)
PIPELINE_NODES = [
    ("ExtractGenesStateful", "🔍 Extracting genes from your query..."),
    ("FetchAllGeneData", "🧬 Fetching gene information from database..."),
    ("AnalyzeNetworkOverlap", "🔗 Mapping shared protein interactions..."),
    ("AnalyzeGoComparison", "🧠 Comparing GO functions across genes..."),
    ("PopulateReportMetadata", "📋 Organising report details..."),
    ("BuildLiteratureQueryPlan", "🗺️ Planning the literature search strategy..."),
    ("FetchAdaptiveLiterature", "📚 Searching scientific literature..."),
    ("RankPapersByRelevance", "🧠 Prioritising the most relevant papers..."),
    ("AnalyzeLiteratureFindings", "📰 Summarising literature insights..."),
    ("InterpretAllGenes", "🤖 Interpreting each gene's role..."),
    ("ValidateInterpretations", "🔎 Validating AI interpretations..."),
    ("InterpretNetworkOverlap", "🔬 Interpreting network patterns..."),
    ("ValidateNetworkInterpretation", "🔎 Validating network interpretation..."),
    ("InterpretGoPatterns", "🧪 Interpreting GO term overlaps..."),
    ("ValidateGoInterpretation", "🔎 Validating GO term interpretation..."),
    ("GenerateCrossGeneSynthesis", "🌐 Connecting cross-gene insights..."),
    ("ValidateCrossGeneSynthesis", "🔎 Validating cross-gene synthesis..."),
    ("ValidateLiteratureFindings", "🔎 Validating literature findings..."),
    ("GenerateGeneSummaries", "✨ Generating final gene summaries..."),
    ("ValidateGeneSummaries", "🔎 Validating gene summaries..."),
    ("FinalSummary", "🧾 Assembling the final report..."),
]

# Map actual node IDs recorded in state snapshots to canonical pipeline IDs
PIPELINE_NODE_ALIASES = {
    'extract_genes': 'ExtractGenesStateful',
    'fetch_gene_data': 'FetchAllGeneData',
    'analyze_network_overlap': 'AnalyzeNetworkOverlap',
    'analyze_go_comparison': 'AnalyzeGoComparison',
    'populate_report_metadata': 'PopulateReportMetadata',
    'build_literature_query_plan': 'BuildLiteratureQueryPlan',
    'fetch_adaptive_literature': 'FetchAdaptiveLiterature',
    'rank_papers_by_relevance': 'RankPapersByRelevance',
    'analyze_literature_findings': 'AnalyzeLiteratureFindings',
    'interpret_all_genes': 'InterpretAllGenes',
    'interpret_network_overlap': 'InterpretNetworkOverlap',
    'interpret_go_patterns': 'InterpretGoPatterns',
    'generate_cross_gene_synthesis': 'GenerateCrossGeneSynthesis',
    'generate_gene_summaries': 'GenerateGeneSummaries',
    'final_summary': 'FinalSummary',
    'validate_network_interpretation': 'ValidateNetworkInterpretation',
    'validate_go_interpretation': 'ValidateGoInterpretation',
    'validate_cross_gene_synthesis': 'ValidateCrossGeneSynthesis',
    'validate_literature_findings': 'ValidateLiteratureFindings',
    'validate_gene_summaries': 'ValidateGeneSummaries',
}


def get_node_status_message(node_id: str) -> str:
    """Get user-friendly status message for a pipeline node."""
    canonical_id = PIPELINE_NODE_ALIASES.get(node_id) or PIPELINE_NODE_ALIASES.get(str(node_id).lower())
    if canonical_id:
        node_id = canonical_id

    for nid, message in PIPELINE_NODES:
        if nid == node_id:
            return message
    # Fallback: convert CamelCase to readable format
    readable = re.sub(r'([A-Z])', r' \1', node_id).strip()
    return f"Running {readable}..."


def run_pipeline_with_progress(query: str, output_mode: str, progress_container, model: str = "gpt-4.1-mini"):
    """Run pipeline with progress tracking via state file polling."""
    import threading
    import time

    # Create persistence directory
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    persistence_dir = f"results/stateful_pipeline/run_{timestamp}_query_{query_hash}"
    state_file = Path(persistence_dir) / "state.json"

    # Result container for thread
    result = {"report": None, "state": None, "error": None}

    def pipeline_thread():
        try:
            report, state_dict, _ = run_pipeline_sync(query, output_mode, model)
            result["report"] = report
            result["state"] = state_dict
        except Exception as e:
            result["error"] = str(e)
            import traceback
            result["traceback"] = traceback.format_exc()

    # Start pipeline in background thread
    thread = threading.Thread(target=pipeline_thread)
    thread.start()

    # Progress display elements
    with progress_container:
        st.subheader("Analysing Genes...")
        st.caption(f"Query: {query} | Mode: {output_mode}")

        extracted_genes_placeholder = st.empty()
        status_text = st.empty()

    last_node_count = 0
    genes_displayed = False
    displayed_gene_list: list[str] = []

    # Poll state file while thread runs
    while thread.is_alive():
        time.sleep(0.5)

        try:
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_data = json.load(f)

                # Get latest state (last entry in array)
                if isinstance(state_data, list) and len(state_data) > 0:
                    latest_state = state_data[-1].get('state', {})
                    nodes_executed = latest_state.get('nodes_executed', [])

                    # Combine all gene sources (queue + already processed)
                    # genes_to_process is consumed as genes are fetched, so we need to combine sources
                    genes_in_queue = latest_state.get('genes_to_process') or []
                    genes_in_mapping = list(latest_state.get('gene_mapping', {}).keys())
                    genes_with_data = list(latest_state.get('all_gene_data', {}).keys())

                    # Combine and deduplicate while preserving order
                    seen = set()
                    genes_list = []
                    for g in genes_in_mapping + genes_with_data + genes_in_queue:
                        if g and g not in seen:
                            genes_list.append(g)
                            seen.add(g)

                    if (not genes_displayed) and genes_list:
                        displayed_gene_list = genes_list[:]
                        gene_count = len(displayed_gene_list)
                        gene_text = ', '.join(displayed_gene_list)
                        extracted_genes_placeholder.success(
                            f"Found {gene_count} gene{'s' if gene_count != 1 else ''}: {gene_text}"
                        )
                        genes_displayed = True

                    if len(nodes_executed) > last_node_count:
                        last_node_count = len(nodes_executed)

                        # Show message for currently running node (one ahead of last completed)
                        node_ids = [nid for nid, _ in PIPELINE_NODES]
                        if nodes_executed:
                            last_completed = nodes_executed[-1]
                            # Resolve alias if needed
                            canonical = PIPELINE_NODE_ALIASES.get(last_completed) or PIPELINE_NODE_ALIASES.get(str(last_completed).lower()) or last_completed
                            if canonical in node_ids:
                                next_index = node_ids.index(canonical) + 1
                                if next_index < len(node_ids):
                                    current_node = node_ids[next_index]
                                else:
                                    current_node = canonical  # last node just finished
                            else:
                                current_node = canonical
                        else:
                            current_node = node_ids[0] if node_ids else "Initialising"
                        status_message = get_node_status_message(current_node)
                        status_text.info(status_message)

        except (json.JSONDecodeError, FileNotFoundError):
            # State file not ready yet
            status_text.info("Initialising pipeline...")

    # Wait for thread to complete
    thread.join()

    # Final update
    status_text.success("✨ Analysis complete! Loading detailed results...")
    time.sleep(1)

    if result["error"]:
        raise Exception(f"{result['error']}\n\n{result.get('traceback', '')}")

    return result["report"], result["state"], persistence_dir


def render_sidebar():
    """Render the right sidebar with inputs."""
    with st.sidebar:
        st.markdown("### Analysis Controls")

        # Query input
        st.markdown("#### Query")
        query = st.text_area(
            "Enter your research question",
            placeholder="e.g., What are the roles of BRCA1 and TP53 in breast cancer?",
            height=100,
            label_visibility="collapsed",
            key="query_input"
        )

        # File uploader (alternative to typing a query)
        st.caption("Or upload a results file")
        uploaded_file = st.file_uploader(
            "Upload a results file",
            type=['json'],
            label_visibility="collapsed"
        )

        if uploaded_file:
            data = json.load(uploaded_file)
            if isinstance(data, list) and len(data) > 0:
                st.session_state['pipeline_state'] = data[-1].get('state', {})
            elif isinstance(data, dict):
                st.session_state['pipeline_state'] = data.get('state', data)
            st.session_state['status'] = 'Complete'
            st.rerun()

        # Output mode selector
        st.markdown("#### Analysis Mode")
        output_mode = st.radio(
            "Select analysis depth",
            options=["interpreted", "factual"],
            format_func=str.capitalize,
            captions=["Full LLM analysis", "Fast, database only"],
            index=0,
            label_visibility="collapsed"
        )

        # Model selector
        st.markdown("#### Model")
        is_running = st.session_state.get("running", False)
        _valid_models = ["gpt-4.1-mini", "gpt-4.1-nano"]
        if st.session_state.get("selected_model") not in _valid_models:
            st.session_state["selected_model"] = _valid_models[0]
        st.radio(
            "Select model",
            options=_valid_models,
            captions=["Highest quality", "Fast & cost-efficient"],
            key="selected_model",
            disabled=is_running,
            label_visibility="collapsed",
        )

        # Run button - always clickable (only disabled when already running)
        button_clicked = st.button("Run Analysis", type="primary", width='stretch', disabled=is_running)

        if button_clicked:
            # Check if query is provided
            if not query:
                st.warning("⚠️ Please enter a query before running analysis")
            # For interpreted mode, check API key
            elif output_mode == "interpreted":
                api_key_check = check_openai_api_key()
                if not api_key_check["has_key"]:
                    # Show clear error message about missing API key
                    st.error(
                        "⚠️ **OpenAI API key required**\n\n"
                        "Interpreted mode requires an OpenAI API key. "
                        "Please add your API key to the .env file:\n\n"
                        "```\nOPENAI_API_KEY=sk-your-actual-key-here\n```\n\n"
                        "Get your API key from: https://platform.openai.com/api-keys\n\n"
                        "**After adding the key, restart Streamlit** (Ctrl+C, then run `streamlit run streamlit_app.py` again)."
                    )
                else:
                    # API key is set - proceed with the run
                    st.session_state['run_query'] = query
                    st.session_state['run_mode'] = output_mode
                    st.session_state['run_model'] = st.session_state.get("selected_model", "gpt-4.1-mini")
                    st.session_state['status'] = 'Running'
                    st.session_state['running'] = True
                    st.rerun()
            else:
                # Factual mode - no API key required, proceed directly
                st.session_state['run_query'] = query
                st.session_state['run_mode'] = output_mode
                st.session_state['run_model'] = st.session_state.get("selected_model", "gpt-4.1-mini")
                st.session_state['status'] = 'Running'
                st.session_state['running'] = True
                st.rerun()

        st.markdown("---")

        # Load saved runs
        st.markdown("#### Previous Runs")
        st.caption("Load a previously saved analysis")

        saved_runs = get_saved_runs()

        if saved_runs:
            run_options = ["Select a run..."] + [r['display_name'] for r in saved_runs]
            selected_run = st.selectbox(
                "Select from saved runs",
                options=run_options,
                label_visibility="collapsed",
                key="saved_run_selector"
            )

            if selected_run != "Select a run...":
                run_info = next((r for r in saved_runs if r['display_name'] == selected_run), None)
                if run_info and st.button("Load Selected Run", width='stretch'):
                    # Point pipeline_state_path to the saved run file
                    st.session_state['pipeline_state_path'] = run_info['path']
                    st.session_state['status'] = 'Complete'
                    st.rerun()
        else:
            st.info("No saved runs found")

        # Status display
        st.markdown("---")
        st.markdown("#### Status")
        status = st.session_state.get('status', 'Ready')

        if status == "Ready":
            st.success("Ready")
        elif status == "Running":
            st.info("Running...")
        elif status == "Complete":
            st.success("Complete")
        elif status == "Error":
            st.error("Error occurred")
        else:
            st.info(status)

        # Show persistence path if available
        if 'persistence_dir' in st.session_state:
            st.caption(f"State saved to: {st.session_state['persistence_dir']}")
