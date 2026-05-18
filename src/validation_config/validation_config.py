"""
Configuration for Summary Validation & Refinement System

Controls model selection, thresholds, and iteration limits for the
claim extraction, validation, and refinement pipeline.
"""

from typing import Literal

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# ClaimExtractor - extracts discrete claims from summaries
# Simpler categorization task - cheaper model sufficient
CLAIM_EXTRACTOR_MODEL = "gpt-4.1-mini"
CLAIM_EXTRACTOR_TEMPERATURE = 0.0  # Deterministic for consistency

# ClaimValidator - validates claims against source data
# Needs BEST reasoning for nuanced tiered validation of interpretations vs facts
CLAIM_VALIDATOR_MODEL = "gpt-4.1-mini"  # Strong reasoning model for scientific validation
CLAIM_VALIDATOR_TEMPERATURE = 1

# SummaryRefiner - rewrites summaries with corrections
# Just fixing specific errors - doesn't need full power
SUMMARY_REFINER_MODEL = "gpt-4.1-mini"
SUMMARY_REFINER_TEMPERATURE = 0.3  # Some flexibility for natural rewriting

# ClaimClarifier - handles partially_accurate claims
# Decides SOFTEN/CORRECT/REJECT before refinement
CLAIM_CLARIFIER_MODEL = "gpt-4.1-mini"
CLAIM_CLARIFIER_TEMPERATURE = 1.0

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

# Minimum accuracy percentage to consider a summary acceptable
# Below this threshold, refinement is triggered
# Lowered from 95% to allow interpretive claims to pass validation
ACCURACY_THRESHOLD = 100.0  # Require 100% accuracy on verifiable claims

# Confidence threshold for claim validation
# Claims with validation confidence below this are flagged for review
CONFIDENCE_THRESHOLD = 0.7

# ============================================================================
# ITERATION LIMITS
# ============================================================================

# Maximum number of rewrite attempts after initial validation
# Each attempt includes: rewrite → re-validate → check accuracy
# With MAX_REWRITE_ATTEMPTS=3, you get 4 total passes:
#   Pass 1 = initial validation
#   Pass 2 = first rewrite + re-validate
#   Pass 3 = second rewrite + re-validate
#   Pass 4 = third rewrite + re-validate
MAX_REWRITE_ATTEMPTS = 3

# Legacy constant - kept for backward compatibility
# Total passes = 1 + MAX_REWRITE_ATTEMPTS
MAX_REFINEMENT_PASSES = 1 + MAX_REWRITE_ATTEMPTS

# ============================================================================
# SENTENCE/CLAIM STRUCTURE
# ============================================================================

# Expected sentence counts per section (min, max)
# Claims are extracted 1:1 with sentences for precise validation
EXPECTED_SENTENCES = {
    "functional_summary": (2, 3),
    "expression_interpretation": (2, 3),
    "interaction_interpretation": (2, 3),
    # experimental_relevance removed — the summariser never generates this section
    # because the experiment context is not in the summariser payload.
    # Re-add here (and to SECTIONS_TO_VALIDATE) when generation is implemented.
    "gene_summary": (6, 8),
    "literature_finding": (1, 2),  # Key findings are short (1-2 sentences)
}

# Minimum words for a valid sentence (filters fragments)
MIN_SENTENCE_WORDS = 4

# ============================================================================
# SECTION CONFIGURATION
# ============================================================================

# Which summary sections to validate
# Set to None to validate all sections.
# NOTE: experimental_relevance is intentionally absent — the summariser does not
# receive the experiment context needed to generate it, so it always comes back empty.
# It will always show "no claims extracted" until generation is added.
SECTIONS_TO_VALIDATE = [
    "functional_summary",
    "expression_interpretation",
    "interaction_interpretation",
    # gene_summary is intentionally absent here — it does not exist at
    # ValidateInterpretations time (generated later by GenerateGeneSummaries).
    # ValidateGeneSummaries validates it directly using sections_to_validate=['gene_summary'].
]

# Claim types and their expected data sources in state
CLAIM_TYPE_SOURCES = {
    "database_fact": ["all_gene_data", "gene_profiles"],
    "expression_pattern": ["all_gene_data.expression", "gene_profiles"],
    "go_annotation": ["all_gene_data.go_terms"],
    "protein_interaction": ["all_gene_data.interactions", "network_overlap_analysis"],
    "literature_finding": ["gene_top_papers", "literature_findings_summary"],
    "cross_gene_relationship": ["go_comparison_analysis", "network_overlap_analysis", "synthesis"],
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Directory for validation logs (used when EXPORT_VALIDATION_LOGS is True)
VALIDATION_LOG_DIR = "results/validation_logs"

# Whether to export validation logs to JSON files.
# The primary source of truth is state.validation_results (state-native).
# Set to True for archival/debugging purposes; False for state-only mode.
EXPORT_VALIDATION_LOGS = False

# Whether to save detailed claim-level logs (within exported JSON files)
SAVE_DETAILED_LOGS = True

# Whether to include raw state data in logs (can be large)
INCLUDE_STATE_IN_LOGS = False

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable/disable validation system
VALIDATION_ENABLED = True
VALIDATE_GENE_SUMMARY = True
VALIDATE_LITERATURE_FINDINGS = True

# Run validation in parallel for multiple genes (if True)
PARALLEL_VALIDATION = True

# Skip validation if pipeline is in "factual" mode (no AI summaries)
SKIP_IN_FACTUAL_MODE = True

# Selective refinement: only re-validate refined claims, preserve accurate ones
# If False, falls back to regenerating and re-validating entire summary each iteration
ENABLE_SELECTIVE_REFINEMENT = True

# Clarify step is mandatory for nuanced validation
ENABLE_CLARIFY_STEP = True

# Flag low confidence results when accuracy cannot meet threshold after retries
LOW_CONFIDENCE_THRESHOLD = 85.0
LOW_CONFIDENCE_FLAG_FIELD = "low_confidence"

# Rewrite softened claims via refiner for precise language
REWRITE_SOFTEN_CLAIMS = True

# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

# Maximum concurrent API calls for claim validation
# Reduce to 5 if you hit OpenAI rate limits
MAX_CONCURRENT_VALIDATIONS = 10

# Maximum concurrent API calls for claim refinement
MAX_CONCURRENT_REFINEMENTS = 5

# Maximum concurrent genes to rank papers for simultaneously
MAX_CONCURRENT_GENE_RANKINGS = 10

# ============================================================================
# LITE MODE (Pass 1 only — skip rewrite and Pass 2 re-validation)
# ============================================================================

# Literature key findings: short 1-2 sentence claims where full remediation is overkill
LITE_MODE_LITERATURE_FINDINGS = False

# Gene summaries: longer multi-claim summaries that benefit from full validation
LITE_MODE_GENE_SUMMARIES = False

# ============================================================================
# ERROR HANDLING
# ============================================================================

# Number of retries for API calls
API_RETRY_COUNT = 3

# Delay between retries (seconds)
API_RETRY_DELAY = 1.0

# Continue with original summary if validation fails completely
FALLBACK_ON_ERROR = True
