#!/usr/bin/env python3
"""
REVEAL Pipeline - CLI
Processes genes one-at-a-time with state persistence.
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from src.graph.gene_graph import run_stateful_pipeline
from src.models.models import ExperimentContext
from src.utils.phoenix_tracing import setup_phoenix

PIPELINE_NODE_IDS = {
    # Phase 1: Extraction & Fetching
    "ExtractGenesStateful",
    "FetchAllGeneData",
    # Phase 2: Cross-Gene Analysis
    "AnalyzeNetworkOverlap",
    "AnalyzeGoComparison",
    "PopulateReportMetadata",
    # Phase 3: Literature Mining
    "BuildLiteratureQueryPlan",
    "FetchAdaptiveLiterature",
    "RankPapersByRelevance",
    "AnalyzeLiteratureFindings",
    # Phase 4: Interpretation & Validation
    "InterpretAllGenes",
    "ValidateInterpretations",
    "InterpretNetworkOverlap",
    "ValidateNetworkInterpretation",
    "InterpretGoPatterns",
    "ValidateGoInterpretation",
    "GenerateCrossGeneSynthesis",
    "ValidateCrossGeneSynthesis",
    "GenerateGeneSummaries",
    "ValidateGeneSummaries",
    "ValidateLiteratureFindings",
    # Phase 5: Report Generation
    "FinalSummary",
}


async def main():
    if len(sys.argv) < 2:
        print("Usage: python run_stateful_pipeline.py \"YOUR QUERY\" [OPTIONS]")
        print("\nExamples:")
        print('  python run_stateful_pipeline.py "What does TP53 do?"')
        print('  python run_stateful_pipeline.py "Analyse BRCA1, BRCA2, PALB2" --factual-only')
        print('  python run_stateful_pipeline.py "Endothelial markers" --no-persist')
        print('  python run_stateful_pipeline.py "BRCA1" --resume')
        print('  python run_stateful_pipeline.py "BRCA1" --resume --resume-from InterpretAllGenes')
        print('  python run_stateful_pipeline.py --status results/stateful_pipeline/run_xxx')
        print('\nOptions:')
        print('  --factual-only: Disable AI interpretations, return database facts only (for debugging)')
        print('  --no-persist: Disable state persistence (default: ON)')
        print('  --resume: Resume from last successful checkpoint (requires matching query)')
        print('  --resume-from NODE: Resume from specific node (e.g., InterpretAllGenes)')
        print('  --status DIR: Show checkpoint status for a run directory')
        print('\nNote: AI interpretations and validation are ON by default')
        sys.exit(1)

    # Handle --status command (doesn't need a query)
    if '--status' in sys.argv:
        status_idx = sys.argv.index('--status')
        if status_idx + 1 < len(sys.argv):
            from src.graph.gene_graph import get_checkpoint_status
            get_checkpoint_status(sys.argv[status_idx + 1])
            sys.exit(0)
        else:
            print("Error: --status requires a directory path")
            sys.exit(1)

    # Parse arguments
    # Persistence is now ON by default (use --no-persist to disable)
    persist = '--no-persist' not in sys.argv
    resume = '--resume' in sys.argv

    # Parse --resume-from argument
    resume_from_node = None
    if '--resume-from' in sys.argv:
        resume_idx = sys.argv.index('--resume-from')
        if resume_idx + 1 < len(sys.argv):
            resume_from_node = sys.argv[resume_idx + 1]
            resume = True  # --resume-from implies --resume
        else:
            print("Error: --resume-from requires a node name")
            sys.exit(1)

    # Parse mode argument (interpretations ON by default)
    output_mode = "factual" if '--factual-only' in sys.argv else "interpreted"

    # Get query (everything that's not a flag or flag value)
    flag_values = set()
    for flag in ['--resume-from', '--status']:
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 < len(sys.argv):
                flag_values.add(sys.argv[idx + 1])

    args = [
        arg for arg in sys.argv[1:]
        if not arg.startswith('--') and arg not in flag_values
    ]
    user_query = " ".join(args)

    if not user_query:
        print("Error: No query provided")
        sys.exit(1)

    # Create experiment context
    experiment_context = ExperimentContext(
        organism="human",
        comparison="Gene function analysis"
    )

    # Create persistence directory if requested
    persistence_dir = None
    if persist:
        # Check if there's an existing state for this query
        import hashlib
        query_hash = hashlib.md5(user_query.encode()).hexdigest()[:8]
        results_dir = Path("results/stateful_pipeline")

        # Look for existing runs with same query
        existing_runs = list(results_dir.glob(f"run_*_query_{query_hash}")) if results_dir.exists() else []
        compatible_run = None
        if existing_runs:
            for candidate in sorted(existing_runs):
                if _is_run_compatible(candidate):
                    compatible_run = candidate

            if compatible_run:
                persistence_dir = str(compatible_run)
                if resume:
                    print(f"\n♻️  Found compatible state for resume: {persistence_dir}")
                    # Show what we're resuming from
                    from src.graph.gene_graph import get_resume_info
                    info = get_resume_info(Path(persistence_dir) / "state.json")
                    if info and info['can_resume']:
                        print(f"   Will resume after: {info['last_completed_node']}")
                        print(f"   Completed nodes: {info['completed_count']}")
                    elif info and info['is_complete']:
                        print(f"   ⚠️  Previous run completed successfully.")
                        if resume_from_node:
                            print(f"   Will re-run from: {resume_from_node}")
                        else:
                            print(f"   Use --resume-from NODE to re-run specific nodes.")
                else:
                    print(f"\n♻️  Found existing state for this query: {persistence_dir}")
                    print(f"   Use --resume to continue from last checkpoint")

        if not persistence_dir:
            # Create new run directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            persistence_dir = f"results/stateful_pipeline/run_{timestamp}_query_{query_hash}"
            print(f"\n🆕 Creating new state directory: {persistence_dir}")

    # Start Phoenix tracing (no-op if dependencies missing)
    setup_phoenix()

    # Run pipeline
    try:
        final_report, pipeline_state = await run_stateful_pipeline(
            user_query=user_query,
            experiment_context=experiment_context,
            output_mode=output_mode,
            persistence_dir=persistence_dir
        )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results/stateful_pipeline")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write(final_report)

        print(f"\n✅ Report saved to: {report_file}")
        if persist and persistence_dir:
            print(f"✅ State saved to: {persistence_dir}")
            print("\n💡 Benefits of persistence:")
            print("   - Can resume if interrupted")
            print("   - Can load state later to ask new questions")
            print("   - All gene data accumulated and queryable")
            run_dir = Path(persistence_dir)
            num_genes = len(pipeline_state.all_gene_data)
            print_usage_instructions(run_dir, user_query, num_genes)
        elif not persist:
            print("\n⚠️  Persistence disabled (--no-persist). State instructions skipped.")

    except KeyboardInterrupt:
        print("\n\n⏸️  Pipeline interrupted!")
        if persist:
            print(f"   State is saved in: {persistence_dir}")
            print("   You can resume or query this data later")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_usage_instructions(run_dir: Path, query: str, num_genes: int):
    """Print instructions for how to use the state after pipeline completes."""
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nYour state has been saved to:")
    print(f"   {run_dir}")
    print(f"\nThis state contains:")
    print(f"   - Query: {query}")
    print(f"   - {num_genes} genes with complete data from database")
    print(f"   - LLM-generated interpretations for each gene")
    print(f"   - Final synthesis report")
    print(f"\nYour state file:")
    print(f"   Location: {run_dir / 'state.json'}")
    print(f"   This JSON file contains all your data")
    print(f"\nTo explore your results:")
    print(f"   - Launch the Streamlit interface: streamlit run streamlit_app.py")
    print(f"   - Load the state file from the sidebar to view results")
    print(f"   - Or open {run_dir / 'state.json'} directly in Python")
    print("\n" + "=" * 80 + "\n")


def _is_run_compatible(run_dir: Path | str) -> bool:
    """Return True only if stored snapshots can be safely loaded by pydantic_graph.

    Checks:
    1. All node IDs must be in the current pipeline
    2. No empty node dicts (End nodes) - these cause pydantic validation errors
       when FileStatePersistence tries to load them
    """
    run_path = Path(run_dir)
    state_file = run_path / "state.json"

    if not state_file.exists():
        return False

    try:
        snapshots = json.loads(state_file.read_text())
    except Exception:
        return False

    for snapshot in snapshots:
        node = snapshot.get('node', {})
        node_id = node.get('node_id')

        # Empty node dict (End node) causes pydantic validation errors
        # Mark these states as incompatible - they can't be loaded safely
        if not node or node == {}:
            return False

        # Check if this node ID is in the current pipeline
        if node_id and node_id not in PIPELINE_NODE_IDS:
            return False

    return True


if __name__ == "__main__":
    asyncio.run(main())
