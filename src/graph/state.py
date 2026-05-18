"""
Stateful graph state for REVEAL pipeline.

This replaces GeneQueryContext for the new stateful architecture.
Key difference: genes are processed ONE AT A TIME and accumulated in dictionaries.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from src.models.models import ExperimentContext, GeneNameMapping
from src.models.report_components import GeneProfile, ReportMetadata, BiologicalSynthesis


class ClaimObject(BaseModel):
    """Structured representation of a validated claim.

    Note: Changed from dataclass to Pydantic BaseModel to fix serialization warnings
    when LangGraph saves state snapshots to JSON.
    """

    claim_id: str
    section: str
    claim_text: str
    category: str
    tier: int
    verdict: str
    clarify_decision: Optional[str] = None
    final_text: Optional[str] = None

@dataclass
class GeneState:
    """
    State that accumulates as genes are processed one-by-one.

    Flow:
    1. ExtractGenes → populates genes_to_process queue
    2. FetchAllGeneData → iterates through queue, storing factual profiles
    3. Cross-gene deterministic analyses (network/GO/metadata)
    4. Interpretation phase (skipped in factual mode)
    5. FinalSummary → synthesizes everything into unified report
    """

    # ========================================================================
    # INPUT
    # ========================================================================
    user_query: str
    experiment_context: ExperimentContext
    db_path: str = field(default_factory=lambda: __import__("src.config", fromlist=["DEFAULT_DB_PATH"]).DEFAULT_DB_PATH)
    output_mode: Literal["factual", "interpreted"] = "interpreted"

    # ========================================================================
    # GENE PROCESSING QUEUE (managed by FetchAllGeneData)
    # ========================================================================
    genes_to_process: List[str] = field(default_factory=list)  # Queue of genes
    current_gene: str | None = None  # Gene being worked on right now
    current_gene_data: Dict[str, Any] | None = None  # Working memory

    # ========================================================================
    # ACCUMULATED DATA (builds up as genes are processed)
    # ========================================================================
    # These are the KEY fields - all genes go into SAME dictionaries
    all_gene_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Changed to Any to support both string summaries and structured dict interpretations
    gene_interpretations: Dict[str, Any] = field(default_factory=dict)

    # ========================================================================
    # MODULAR REPORT COMPONENTS (NEW)
    # ========================================================================
    report_metadata: Optional[ReportMetadata] = None
    gene_profiles: Dict[str, GeneProfile] = field(default_factory=dict)  # key = gene_symbol
    synthesis: Optional[BiologicalSynthesis] = None

    # Store generated gene summaries
    # Can be either plain strings (from GenerateGeneSummaries) or dicts (from academic summarizer)
    # Structure: {'BRCA1': 'comprehensive summary...'} or {'BRCA1': {'gene_description': '...', ...}}
    gene_summaries: Dict[str, Any] = field(default_factory=dict)

    # ========================================================================
    # LITERATURE MINING (PubMed) - Adaptive Tiered System
    # ========================================================================

    # Versioning
    literature_version: str = "v4.0-adaptive"

    # NEW: Tiered query system - stores QueryTier objects per gene
    # Structure: {'BRCA1': [QueryTier(...), QueryTier(...), ...]}
    literature_query_tiers: Dict[str, List] = field(default_factory=dict)

    # NEW: Extracted context terms from LLM
    # Structure: {'diseases': [...], 'processes': [...], 'cell_types': [...]}
    literature_context_terms: Dict[str, Any] = field(default_factory=dict)

    # NEW: All candidate papers considered (before ranking)
    # Structure: {'BRCA1': {'tier_used': str, 'total_count': int, 'candidate_papers': [...], ...}}
    gene_literature_candidates: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Context terms resolved and used in queries (for audit trail)
    literature_context_terms_used: List[str] = field(default_factory=list)

    # Disease terms detected and used in queries
    literature_disease_terms_detected: List[str] = field(default_factory=list)
    literature_disease_filter_applied: str = ""

    # Per gene query strings (backward compat)
    literature_query_plan: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Per gene, per bucket hit counts (backward compat)
    pubmed_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Per gene, per bucket list of PMIDs returned by search
    pubmed_pmids: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    # PMID -> record (metadata/abstract) cache
    pubmed_records: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Top papers per gene for display (ranked, top 10)
    gene_top_papers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Reverse mappings for provenance (PMID -> buckets/genes)
    pubmed_pmid_buckets: Dict[str, List[str]] = field(default_factory=dict)  # pmid -> buckets
    pubmed_pmid_genes: Dict[str, List[str]] = field(default_factory=dict)    # pmid -> genes

    # Retrieval parameters for reproducibility
    pubmed_retrieval_params: Dict[str, Any] = field(default_factory=dict)

    # Derived metrics per gene for scoring
    pubmed_context_fraction: Dict[str, float] = field(default_factory=dict)  # strict/global ratio
    pubmed_context_fraction_in_disease: Dict[str, float] = field(default_factory=dict)  # strict/disease ratio
    pubmed_publication_type_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)  # gene -> type counts

    # Quality tracking with structured error reporting
    literature_quality: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # supports: errors, flags, retry_suggested, retry_queries
    literature_retry_count: int = 0

    # Literature findings analysis - research trends and themes per gene
    # Structure: {'BRCA1': {'total_papers': int, 'recent_papers_count': int, 'research_themes': [...], ...}}
    literature_findings_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Literature retrieval statistics for thesis benchmarking (April 2026)
    # Structure: {'BRCA1': {'candidates_before_bm25': int, 'candidates_after_bm25': int,
    #                        'bm25_applied': bool, 'final_papers_count': int,
    #                        'source_paper_found': bool, 'source_paper_pmid': str|None}}
    literature_retrieval_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ========================================================================
    # EXTRACTION OUTPUT
    # ========================================================================
    extraction_confidence: float = 0.0
    gene_name_mapping: Optional[GeneNameMapping] = None
    # Identifiers that could not be resolved to database genes
    unresolved_identifiers: List[Dict[str, Any]] = field(default_factory=list)

    # ========================================================================
    # RETRIEVAL METADATA
    # ========================================================================
    gene_mapping: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gene_alias_map: Dict[str, str] = field(default_factory=dict)  # Maps user input → official symbol
    protein_network: Optional[Dict[str, Any]] = None
    retrieval_stats: Dict[str, int] = field(default_factory=dict)

    # ========================================================================
    # VALIDATION OUTPUT (Phase 2)
    # ========================================================================
    factual_statements: List[Dict[str, Any]] = field(default_factory=list)
    # State-native validation snapshots: gene_symbol -> UI-ready validation dict.
    # This is the SINGLE SOURCE OF TRUTH for Streamlit validation display.
    # Structure documented in src/validation/validation_snapshot.py
    validation_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hallucinations_detected: List[Dict[str, Any]] = field(default_factory=list)
    verification_reports: List[Dict[str, Any]] = field(default_factory=list)
    verification_summary: Dict[str, Any] = field(default_factory=dict)
    validated_claims: Dict[str, List[ClaimObject]] = field(default_factory=dict)
    claim_verdicts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accuracy_scores: Dict[str, float] = field(default_factory=dict)
    validation_summary: Dict[str, str] = field(default_factory=dict)
    low_confidence_flags: Dict[str, bool] = field(default_factory=dict)

    # Summary validation logs (from ValidateInterpretations node)
    # Structure: {'GENE': {'accuracy': float, 'improvement': float, 'hallucinations': int, ...}}
    validation_logs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Data source tracking
    data_sources: List[Dict[str, Any]] = field(default_factory=list)

    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================
    final_interpretation: Optional[str] = None  # Comprehensive final report
    biological_insights: Dict[str, Any] = field(default_factory=dict)
    key_findings: List[str] = field(default_factory=list)

    # ========================================================================
    # ANALYSIS OUTPUTS
    # ========================================================================
    network_overlap_analysis: Optional[Dict[str, Any]] = None
    go_comparison_analysis: Optional[Dict[str, Any]] = None

    # ========================================================================
    # CROSS-GENE VALIDATION (keyed by section name, not gene symbol)
    # ========================================================================
    # Keys: "go_interpretation", "cross_gene_synthesis"
    cross_gene_validation_results: Dict[str, Any] = field(default_factory=dict)
    cross_gene_accuracy_scores: Dict[str, float] = field(default_factory=dict)

    # ========================================================================
    # METADATA
    # ========================================================================
    error: Optional[str] = None
    pipeline_start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    active_model: str = "gpt-4.1-mini"
    nodes_executed: List[str] = field(default_factory=list)
    execution_times: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Event logger (optional, for compatibility)
    event_logger: Optional[Any] = None

    # Phase tracking
    factual_data_complete: bool = False
    cross_gene_analysis_complete: bool = False
    interpretations_complete: bool = False

    def log_node_execution(self, node_name: str, execution_time: float = None):
        """Track which nodes have been executed"""
        self.nodes_executed.append(node_name)
        if execution_time is not None:
            self.execution_times[node_name] = execution_time

    def has_error(self) -> bool:
        return self.error is not None

    def add_data_source(self, gene: str, source_type: str, source_info: Dict[str, Any]):
        """
        Track data source for Phase 2 validation

        Args:
            gene: Gene symbol
            source_type: Type of data (e.g., 'function', 'go_term', 'interaction')
            source_info: Details about the source (table, record_id, etc.)
        """
        self.data_sources.append({
            'gene': gene,
            'source_type': source_type,
            'source_info': source_info,
            'timestamp': datetime.now().isoformat()
        })

    def get_genes_found(self) -> List[str]:
        """Get list of genes that were successfully found"""
        return [
            symbol for symbol, info in self.gene_mapping.items()
            if info and info.get('found', False)
        ]

    def get_genes_missing(self) -> List[str]:
        """Get list of genes that were not found"""
        return [
            symbol for symbol, info in self.gene_mapping.items()
            if not info or not info.get('found', False)
        ]


def _accumulate_tokens(state, node_name: str, usage) -> None:
    """Accumulate token usage into state.token_usage[node_name].

    Handles both OpenAI and pydantic_ai usage object formats:
    - OpenAI: prompt_tokens, completion_tokens, total_tokens
    - pydantic_ai RunUsage: input_tokens, output_tokens
    """
    if state is None or usage is None:
        return

    current = state.token_usage.get(
        node_name,
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )

    # Try OpenAI format first
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)

    # Fall back to pydantic_ai format if OpenAI fields are None
    if prompt_tokens is None:
        prompt_tokens = getattr(usage, "input_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "output_tokens", None)

    # Convert None to 0
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0

    # Calculate total
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    state.token_usage[node_name] = {
        "prompt_tokens": current["prompt_tokens"] + prompt_tokens,
        "completion_tokens": current["completion_tokens"] + completion_tokens,
        "total_tokens": current["total_tokens"] + total_tokens,
    }
