"""
Methods & Diagnostics page for REVEAL.
"""

import streamlit as st
import json
from typing import Dict, Any

from pydantic import BaseModel


def render_methods_diagnostics(state: Dict[str, Any]):
    """Render the Methods & Diagnostics tab."""
    st.markdown("## Methods & Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Pipeline Execution")

        nodes_executed = state.get('nodes_executed', [])
        execution_times = state.get('execution_times', {})

        if nodes_executed:
            st.markdown(f"**Nodes executed:** {len(nodes_executed)}")

            for node in nodes_executed:
                time_taken = execution_times.get(node, None)
                if time_taken:
                    st.markdown(f"- {node}: {time_taken:.2f}s")
                else:
                    st.markdown(f"- {node}")

            # Total time
            if execution_times:
                total_time = sum(execution_times.values())
                st.markdown(f"**Total execution time:** {total_time:.2f}s")
        else:
            st.info("No execution data available")

        # Pipeline metadata
        st.markdown("### Run Metadata")

        # Format start time nicely if it's an ISO string
        raw_start = state.get('pipeline_start_time', '')
        if raw_start:
            try:
                from datetime import datetime as _dt
                parsed = _dt.fromisoformat(raw_start)
                start_display = parsed.strftime("%d %b %Y, %H:%M:%S")
            except Exception:
                start_display = raw_start
        else:
            start_display = 'N/A'

        active_model = state.get('active_model', 'N/A')
        output_mode  = state.get('output_mode', 'N/A')
        db_path      = state.get('db_path', 'N/A')

        def _chip(label: str, value: str, icon: str = "") -> str:
            return (
                f'<div style="display:flex; align-items:center; gap:10px; '
                f'padding:10px 14px; margin-bottom:8px; background:#f9f8f6; '
                f'border:1px solid #e8e6e1; border-radius:8px;">'
                f'<span style="font-size:1.1rem;">{icon}</span>'
                f'<div>'
                f'<div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; '
                f'letter-spacing:0.06em; color:#9ca3af;">{label}</div>'
                f'<div style="font-size:0.9rem; font-weight:500; color:#111827;">{value}</div>'
                f'</div></div>'
            )

        st.markdown(
            _chip("Started", start_display, "🕐") +
            _chip("Model", active_model, "🤖") +
            _chip("Output mode", output_mode.title(), "⚙️") +
            _chip("Database", db_path.split("/")[-1], "🗄️"),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### Quality Metrics")

        # Extraction confidence
        extraction_conf = state.get('extraction_confidence', 0)
        st.metric("Extraction Confidence", f"{extraction_conf:.0%}")

        # Literature quality
        lit_quality = state.get('literature_quality', {})
        if lit_quality:
            errors = sum(1 for g, q in lit_quality.items() if q.get('errors'))
            st.metric("Literature Errors", errors)

        # Validation
        validation = state.get('validation_results', {})
        if isinstance(validation, dict) and validation:
            passed = sum(1 for v in validation.values() if isinstance(v, dict) and v.get('status') == 'passed')
            st.metric("Validations Passed", f"{passed}/{len(validation)}")

        # Phase completion
        st.markdown("### Phase Completion")
        phases = {
            'Factual Data': state.get('factual_data_complete', False),
            'Cross-Gene Analysis': state.get('cross_gene_analysis_complete', False),
            'Interpretations': state.get('interpretations_complete', False)
        }

        for phase, complete in phases.items():
            status = "Complete" if complete else "Pending"
            st.markdown(f"- **{phase}:** {status}")

    # Experiment context
    st.markdown("### Experiment Context")
    exp_context = state.get('experiment_context', {})
    if exp_context:
        # Handle both dict and Pydantic model
        def get_ctx(key, default='N/A'):
            if isinstance(exp_context, dict):
                return exp_context.get(key, default)
            else:
                return getattr(exp_context, key, default) or default

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Organism:** {get_ctx('organism')}")
            st.markdown(f"**Cell Type:** {get_ctx('cell_type')}")
        with col2:
            st.markdown(f"**Treatment:** {get_ctx('treatment')}")
            st.markdown(f"**Timepoint:** {get_ctx('timepoint')}")
        with col3:
            st.markdown(f"**Comparison:** {get_ctx('comparison')}")
            st.markdown(f"**Hypothesis:** {get_ctx('hypothesis')}")

    # Literature context
    lit_context = state.get('literature_context_terms_used', [])
    if lit_context:
        st.markdown("### Literature Context Terms")
        st.markdown(", ".join(lit_context))

    disease_terms = state.get('literature_disease_terms_detected', [])
    if disease_terms:
        st.markdown("### Disease Terms Detected")
        st.markdown(", ".join(disease_terms))

    # Raw state viewer
    with st.expander("Raw State Data (for debugging)"):
        try:
            # Try to serialize, handling Pydantic models
            def serialize(obj):
                if isinstance(obj, BaseModel):
                    return obj.model_dump()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            st.json(json.loads(json.dumps(state, default=serialize)))
        except Exception:
            st.write(state)
