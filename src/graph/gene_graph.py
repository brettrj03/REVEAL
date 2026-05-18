"""
Stateful REVEAL Graph

Processes genes one-at-a-time with persistence.

NOTE: This is a simplified initial implementation.
For full validation and retry logic, additional nodes need to be integrated.
"""

import os
import sys
import time
import warnings

# Apply Python 3.13+ compatibility patch before importing pydantic_graph
# This handles stricter forward reference evaluation in Python 3.13+
if sys.version_info >= (3, 13):
    from src.utils.python314_compat import apply_python314_patch
    apply_python314_patch()

from openai import OpenAI
from pydantic_graph import Graph
from pydantic_graph.nodes import BaseNode, GraphRunContext, End

# Patch pydantic_graph modules after import (they cache get_type_hints)
if sys.version_info >= (3, 13):
    from src.utils.python314_compat import patch_pydantic_graph_modules
    patch_pydantic_graph_modules()
from pathlib import Path

from src.graph.state import GeneState
from src.nodes.extract_genes_stateful import ExtractGenesStateful
from src.nodes.fetch_gene_data import FetchAllGeneData
from src.nodes.populate_report_metadata import PopulateReportMetadata
from src.nodes.analyze_network_overlap import AnalyzeNetworkOverlap
from src.nodes.analyze_go_comparison import AnalyzeGoComparison
from src.nodes.build_literature_query_plan import BuildLiteratureQueryPlan
from src.nodes.fetch_adaptive_literature import FetchAdaptiveLiterature
from src.nodes.rank_papers_by_relevance import RankPapersByRelevance
from src.nodes.analyze_literature_findings import AnalyzeLiteratureFindings
from src.nodes.interpret_all_genes import InterpretAllGenes
from src.nodes.interpret_network_overlap import InterpretNetworkOverlap
from src.nodes.validate_network_interpretation import ValidateNetworkInterpretation
from src.nodes.interpret_go_patterns import InterpretGoPatterns
from src.nodes.validate_go_interpretation import ValidateGoInterpretation
from src.nodes.generate_cross_gene_synthesis import GenerateCrossGeneSynthesis
from src.nodes.validate_cross_gene_synthesis import ValidateCrossGeneSynthesis
from src.nodes.generate_gene_summaries import GenerateGeneSummaries
from src.nodes.validate_interpretations import ValidateInterpretations
from src.nodes.validate_gene_summaries import ValidateGeneSummaries
from src.nodes.validate_literature_findings import ValidateLiteratureFindings
from src.nodes.final_summary import FinalSummary
from src.integrations.pubmed_client import PubMedClient
from src.utils.persistence import SafeFileStatePersistence


def create_reveal_graph():
    """
    Create the complete REVEAL pipeline graph with 21 nodes.

    Pipeline phases:
    - Phase 1: Extraction & Fetching (2 nodes)
    - Phase 2: Cross-Gene Analysis (3 nodes)
    - Phase 3: Literature Mining (4 nodes) - Adaptive tiered search + analysis
    - Phase 4: Interpretation & Validation (11 nodes)
    - Phase 5: Report Generation (1 node)

    Returns:
        Configured Graph instance
    """

    nodes = [
        # Phase 1: Extraction & Fetching
        ExtractGenesStateful(),
        FetchAllGeneData(),

        # Phase 2: Cross-Gene Analysis
        AnalyzeNetworkOverlap(),
        AnalyzeGoComparison(),
        PopulateReportMetadata(),

        # Phase 3: Literature Mining (Adaptive Tiered System)
        BuildLiteratureQueryPlan(),      # Builds tiered queries
        FetchAdaptiveLiterature(),       # Smart fetching with fallback
        RankPapersByRelevance(),          # LLM ranking (top_n=10 by default)
        AnalyzeLiteratureFindings(),     # Extract research trends & themes

        # Phase 4: Interpretation & Validation
        InterpretAllGenes(),
        ValidateInterpretations(),
        InterpretNetworkOverlap(),
        ValidateNetworkInterpretation(),      # Validate network interpretation
        InterpretGoPatterns(),
        ValidateGoInterpretation(),           # Validate GO interpretation
        GenerateCrossGeneSynthesis(),
        ValidateCrossGeneSynthesis(),         # Validate cross-gene synthesis
        ValidateLiteratureFindings(),         # Validate paper key findings BEFORE summaries
        GenerateGeneSummaries(),              # Build summaries from validated findings
        ValidateGeneSummaries(),

        # Phase 5: Report Generation
        FinalSummary(),
    ]

    return Graph(
        nodes=nodes,
        state_type=GeneState
    )


# Lazy initialization to avoid forward reference issues at module import time
def get_reveal_graph():
    """Get the default REVEAL pipeline graph instance (lazy initialization)"""
    global _reveal_graph
    try:
        return _reveal_graph
    except NameError:
        pass
    _reveal_graph = create_reveal_graph()
    return _reveal_graph


def get_resume_info(state_file: Path) -> dict | None:
    """
    Read a state.json file and return resume metadata.

    Returns a dict with:
      - can_resume: bool — True if interrupted mid-run
      - is_complete: bool — True if pipeline reached End node
      - last_completed_node: str — last node_id that finished
      - completed_count: int — number of nodes completed
    Returns None if the file can't be read.
    """
    try:
        import json
        snapshots = json.loads(state_file.read_text())
        if not isinstance(snapshots, list) or not snapshots:
            return None

        completed_nodes = []
        is_complete = False
        for snap in snapshots:
            node = snap.get("node", {})
            node_id = node.get("node_id")
            if not node or node == {}:
                is_complete = True  # End node
            elif node_id:
                completed_nodes.append(node_id)

        return {
            "can_resume": not is_complete and len(completed_nodes) > 0,
            "is_complete": is_complete,
            "last_completed_node": completed_nodes[-1] if completed_nodes else None,
            "completed_count": len(completed_nodes),
        }
    except Exception:
        return None


def get_checkpoint_status(run_dir: str) -> None:
    """
    Print a human-readable checkpoint status for a pipeline run directory.
    Called by: python run_stateful_pipeline.py --status <run_dir>
    """
    import json
    run_path = Path(run_dir)
    state_file = run_path / "state.json"

    print(f"\n{'='*60}")
    print(f"CHECKPOINT STATUS: {run_path.name}")
    print(f"{'='*60}")

    if not state_file.exists():
        print("  ❌ No state.json found — run has not started or state was not saved.")
        print(f"{'='*60}\n")
        return

    info = get_resume_info(state_file)
    if info is None:
        print("  ❌ Could not read state file — may be corrupt.")
        print(f"{'='*60}\n")
        return

    if info["is_complete"]:
        print(f"  ✅ Run COMPLETE ({info['completed_count']} nodes)")
        print(f"  Last node: {info['last_completed_node']}")
        print(f"\n  Use --resume-from NODE to re-run from a specific node.")
    elif info["can_resume"]:
        print(f"  ⏸️  Run INTERRUPTED — can resume")
        print(f"  Completed: {info['completed_count']} nodes")
        print(f"  Last completed: {info['last_completed_node']}")
        print(f"\n  Resume with: python run_stateful_pipeline.py \"your query\" --resume")
    else:
        print("  ⚠️  Run started but no nodes completed yet.")

    # Print full node list
    try:
        snapshots = json.loads(state_file.read_text())
        print(f"\n  Completed nodes:")
        for snap in snapshots:
            node_id = snap.get("node", {}).get("node_id")
            if node_id:
                duration = snap.get("duration", "?")
                print(f"    ✓ {node_id} ({duration}s)")
    except Exception:
        pass

    print(f"{'='*60}\n")


async def run_stateful_pipeline(
    user_query: str,
    experiment_context,
    db_path: str = None,
    output_mode: str = "interpreted",
    persistence_dir: str | None = None
):
    """
    Run the stateful REVEAL pipeline.

    Args:
        user_query: Natural language query
        experiment_context: ExperimentContext object
        db_path: Path to gene database
        output_mode: "factual" (fast, no LLM) or "interpreted" (LLM analysis)
        persistence_dir: Optional directory to save state (enables resume)

    Returns:
        Tuple containing the final report string and the final GeneState
    """

    # Resolve database path
    if db_path is None:
        from src.config import DEFAULT_DB_PATH
        db_path = DEFAULT_DB_PATH

    # Create initial state
    from src.config import get_active_model
    state = GeneState(
        user_query=user_query,
        experiment_context=experiment_context,
        db_path=db_path,
        output_mode=output_mode,
        active_model=get_active_model(),
    )

    # Create persistence if requested
    persistence = None
    if persistence_dir:
        persistence_path = Path(persistence_dir)
        persistence_path.mkdir(parents=True, exist_ok=True)
        state_file = persistence_path / "state.json"
        # Always start fresh — delete any stale/corrupted file from a previous
        # failed run that may have ended up in the same directory
        if state_file.exists():
            state_file.unlink()
        persistence = SafeFileStatePersistence(state_file)

    # Run the graph
    print(f"\n{'='*70}")
    print(f"🧬 REVEAL: Retrieval and Evidence-based Validated Interpretation Analysis")
    print(f"{'='*70}")
    print(f"Query: {user_query}")
    print(f"Persistence: {persistence_dir or 'None (in-memory only)'}")
    print(f"{'='*70}\n")

    # Get the graph instance (lazy initialization)
    graph = get_reveal_graph()

    _pipeline_start = time.perf_counter()
    # Suppress Pydantic serialisation warnings from pydantic_graph's internal SafeFileStatePersistence
    # These warnings occur when serializing EndSnapshot with NodeSnapshot schema (library issue, not our code)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*PydanticSerializationUnexpectedValue.*",
            category=UserWarning
        )
        result = await graph.run(
            ExtractGenesStateful(),
            state=state,
            persistence=persistence
        )
    state.execution_times["__total__"] = round(time.perf_counter() - _pipeline_start, 3)

    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Genes processed: {len(result.state.all_gene_data)}")
    print(f"Interpretations: {len(result.state.gene_interpretations)}")
    print(f"Nodes executed: {len(result.state.nodes_executed)}")

    if result.state.execution_times:
        print(f"\n{'='*70}")
        print("EXECUTION TIMES")
        print(f"{'='*70}")
        for node, exec_time in sorted(result.state.execution_times.items()):
            if node == "__total__":
                continue
            print(f"  {node:<40} {exec_time:.3f}s")
        total = result.state.execution_times.get("__total__", "not tracked")
        print(f"  {'TOTAL (wall clock)':<40} {total}")

    if result.state.token_usage:
        print(f"\n{'='*70}")
        print("TOKEN USAGE")
        print(f"{'='*70}")
        grand = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for node, usage in sorted(result.state.token_usage.items()):
            prompt = usage.get('prompt_tokens', 0)
            completion = usage.get('completion_tokens', 0)
            print(f"  {node:<40} {prompt} in / {completion} out")
            for key in grand:
                grand[key] += usage.get(key, 0)
        print(f"  {'TOTAL':<40} {grand['prompt_tokens']} in / {grand['completion_tokens']} out / {grand['total_tokens']} total")

    print(f"{'='*70}\n")

    # Write pipeline log
    run_dir = Path(persistence_dir) if persistence_dir else Path('results/stateful_pipeline')
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'pipeline.log'
    lines = []
    lines.append(f"Run directory: {run_dir}")
    lines.append(f"Query: {state.user_query}")
    lines.append(f"Genes: {getattr(state, 'genes_to_process', [])}")
    lines.append("")
    lines.append("=== Execution Times ===")
    for node, t in sorted(state.execution_times.items()):
        lines.append(f"  {node}: {t:.3f}s")
    lines.append("")
    lines.append("=== Token Usage ===")
    for node, usage in sorted(state.token_usage.items()):
        lines.append(
            f"  {node}: {usage.get('prompt_tokens', 0)} prompt / "
            f"{usage.get('completion_tokens', 0)} completion / "
            f"{usage.get('total_tokens', 0)} total"
        )
    lines.append("")
    lines.append("=== Validation Iterations ===")
    for gene, vdata in (state.validation_results or {}).items():
        for section, sdata in (vdata.get('sections') or {}).items():
            iters = len(sdata.get('iterations', []))
            lines.append(f"  {gene} / {section}: {iters} iteration(s)")
    log_path.write_text("\n".join(lines))
    print(f"📋 Pipeline log written to: {log_path}")

    return result.output, result.state
