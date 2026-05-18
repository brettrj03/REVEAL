"""
Streamlit CSS styles for REVEAL.
"""

import streamlit as st


def apply_styles():
    """Apply all custom CSS styles to the Streamlit app."""
    st.markdown("""
<style>
    /* Clean header styling */
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1f2937;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    /* Card-like sections */
    .info-card {
        background-color: #f9fafb;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }

    /* Gene chip styling */
    .gene-chip {
        display: inline-block;
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }

    /* Score indicator */
    .score-high { color: #059669; font-weight: 600; }
    .score-medium { color: #d97706; font-weight: 600; }
    .score-low { color: #dc2626; font-weight: 600; }

    /* Metric boxes */
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        color: white;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
    }

    /* ── Executive Summary ── */
    .exec-query-bar {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1.25rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem 1.5rem;
        align-items: flex-start;
    }
    .exec-query-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #3b82f6;
        margin-bottom: 0.2rem;
    }
    .exec-query-text {
        font-family: monospace;
        font-size: 0.82rem;
        color: #374151;
    }
    .exec-meta-pill {
        display: inline-block;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 0.15rem 0.6rem;
        font-size: 0.72rem;
        color: #6b7280;
        white-space: nowrap;
    }
    .exec-meta-pill span { color: #374151; font-weight: 500; }
    .exec-gene-chip {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: monospace;
        letter-spacing: 0.03em;
        margin: 0.2rem;
        cursor: default;
    }
    .go-namespace-header {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 0.5rem 0.75rem;
        border-radius: 6px 6px 0 0;
        margin-bottom: 0;
    }
    .go-bp-header { background: #eff6ff; color: #1d4ed8; }
    .go-mf-header { background: #f0fdf4; color: #15803d; }
    .go-cc-header { background: #faf5ff; color: #7e22ce; }
    .go-term-row {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #f3f4f6;
        font-size: 0.82rem;
    }
    .go-term-row:last-child { border-bottom: none; }
    .go-term-name { color: #374151; margin-bottom: 0.25rem; }
    .go-gene-tag {
        display: inline-block;
        font-size: 0.68rem;
        font-family: monospace;
        font-weight: 600;
        padding: 0.1rem 0.4rem;
        border-radius: 3px;
        margin: 0.1rem;
    }
    .go-bp-tag { background: #dbeafe; color: #1e40af; }
    .go-mf-tag { background: #dcfce7; color: #166534; }
    .go-cc-tag { background: #f3e8ff; color: #6b21a8; }
    .network-callout-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .network-stat-num {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        line-height: 1;
    }
    .network-stat-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-top: 0.2rem;
    }

    /* ── Gene table context bar ── */
    .exec-ctx-bar {
        font-size: 0.8rem;
        color: #6b7280;
        margin-bottom: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.25rem;
    }
    .exec-ctx-query {
        color: #1f2937;
        font-family: monospace;
        font-size: 0.82rem;
    }
    .exec-ctx-pill {
        color: #6b7280;
    }
    .exec-ctx-pill strong {
        color: #374151;
    }

    /* ── AI Research Synthesis ── */
    .ai-synthesis-box {
        background: linear-gradient(135deg, #f5f3ff 0%, #eff6ff 100%);
        border: 1px solid #c4b5fd;
        border-left: 4px solid #7c3aed;
        border-radius: 0 8px 8px 0;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1.25rem;
    }
    .ai-synthesis-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7c3aed;
        margin-bottom: 0.55rem;
    }
    .ai-synthesis-text {
        font-size: 0.92rem;
        color: #1f2937;
        line-height: 1.75;
    }
    .ai-synthesis-footer {
        font-size: 0.68rem;
        color: #9ca3af;
        margin-top: 0.75rem;
        font-style: italic;
    }

    /* ── Validation Quality Overview ── */
    .val-overview-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
    }
    .val-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 6px;
        white-space: nowrap;
    }
    .val-badge-green { background: #dcfce7; color: #15803d; }
    .val-badge-amber { background: #fef3c7; color: #92400e; }
    .val-badge-red   { background: #fee2e2; color: #991b1b; }
    .val-stat {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        padding: 0 0.75rem;
        border-left: 1px solid #e5e7eb;
    }
    .val-stat-num {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1f2937;
        line-height: 1.1;
    }
    .val-pass { color: #15803d; }
    .val-fail { color: #dc2626; }
    .val-stat-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-top: 0.15rem;
    }
    .val-row { font-size: 0.8rem; color: #374151; }
    .val-row-label { color: #6b7280; margin-right: 0.4rem; }
    .val-gene-flag {
        display: inline-block;
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        font-size: 0.72rem;
        font-family: monospace;
        font-weight: 600;
        padding: 0.1rem 0.45rem;
        border-radius: 4px;
        margin: 0.1rem;
    }
</style>
""", unsafe_allow_html=True)
