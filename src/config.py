"""
Configuration settings for the REVEAL pipeline

This centralizes all configuration to avoid hardcoding values throughout the codebase.
"""

from pathlib import Path

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Active model — controlled at runtime by the UI model selector
_ACTIVE_MODEL: str = "gpt-4.1-mini"


def get_active_model() -> str:
    """Return the currently selected pipeline model."""
    return _ACTIVE_MODEL


def set_active_model(model: str) -> None:
    """Set the active model for all pipeline calls."""
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = model


# OpenAI Models (kept for backward-compat; pipeline reads get_active_model() at call time)
GENE_EXTRACTION_MODEL = "gpt-4.1-mini"  # Fast and accurate for extraction
INTERPRETATION_MODEL = "gpt-4.1-mini"   # High quality for interpretation
STATEMENT_EXTRACTION_MODEL = "gpt-4.1-mini"  # Precise for statement extraction
VALIDATION_MODEL = "gpt-4.1-mini"            # Thorough for validation

# Model Settings
EXTRACTION_TEMPERATURE = 0.0  # Deterministic for gene extraction
INTERPRETATION_TEMPERATURE = 0.3  # Slightly creative for interpretation
VALIDATION_TEMPERATURE = 0.0  # Strict for validation

EXTRACTION_MAX_TOKENS = 1000
INTERPRETATION_MAX_TOKENS = 8000
STATEMENT_EXTRACTION_MAX_TOKENS = 10000
VALIDATION_MAX_TOKENS = 2000

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Default database path for standalone use.
# Created by: python scripts/setup_database.py
DEFAULT_DB_PATH = str(Path(__file__).parent.parent / "src" / "database" / "gene_database.sqlite")

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

# Retry settings
MAX_INTERPRETATION_RETRIES = 2
MAX_EXTRACTION_RETRIES = 2
MAX_PIPELINE_ITERATIONS = 10  # Prevent infinite loops

# Iteration strategy (for phase 2.5)
MIN_ITERATIONS = 3  # Try at least this many times to find best result
MAX_ITERATIONS = 3  # Don't go beyond this

# Validation thresholds
HALLUCINATION_THRESHOLD = 0.0  # Retry on ANY hallucination
ACCURACY_IMPROVEMENT_THRESHOLD = 0.0  # Always retry if accuracy < 100%

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Default output directory
DEFAULT_OUTPUT_DIR = "results/pipeline"

# Agent logs directory
AGENT_LOGS_DIR = "pipeline_logs"

# ============================================================================
# DATA RETRIEVAL CONFIGURATION
# ============================================================================

# PubMed retrieval mode
# - "best_match": PubMed relevance sorting, fetch top 50 (original default)
# - "date_pool": Date sorting (newest first), fetch top 200 for larger BM25 pool
# - "cascading": Cascading specificity search - starts specific, broadens until ~200 papers
RETRIEVAL_MODE = "cascading"  # <-- OPTIONS: "best_match", "date_pool", "cascading"

# Cascading mode settings
CASCADING_TARGET_POOL = 200  # Stop accumulating when we reach this many papers

# Number of top GO terms to select per category
MAX_GO_TERMS_BIOLOGICAL_PROCESS = 10
MAX_GO_TERMS_MOLECULAR_FUNCTION = 7
MAX_GO_TERMS_CELLULAR_COMPONENT = 5

# Number of expression tissues to include
MAX_EXPRESSION_TISSUES = 8

# Number of protein interactions to include
MAX_PROTEIN_INTERACTIONS = 8
TOP_INTERACTIONS_PER_GENE = 5

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

# Expression matching tolerance (TPM values within this range are considered matches)
EXPRESSION_TPM_TOLERANCE = 0.5

# Protein interaction score matching tolerance
INTERACTION_SCORE_TOLERANCE = 1

# Tissue name matching threshold (percentage of words that must match)
TISSUE_MATCH_THRESHOLD = 0.6

# GO term matching - minimum words in common for function matching
MIN_GO_TERM_WORDS_MATCH = 2

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log levels
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_db_path(custom_path: str = None) -> str:
    """Get database path, using custom path if provided"""
    return custom_path if custom_path else DEFAULT_DB_PATH


def get_output_dir(custom_dir: str = None) -> str:
    """Get output directory, using custom dir if provided"""
    return custom_dir if custom_dir else DEFAULT_OUTPUT_DIR


def get_agent_logs_dir(custom_dir: str = None) -> str:
    """Get agent logs directory, using custom dir if provided"""
    if custom_dir:
        return custom_dir

    # Create relative to output dir
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    return str(output_dir / "agent_logs")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []

    # Check retry limits
    if MAX_INTERPRETATION_RETRIES < 0:
        errors.append("MAX_INTERPRETATION_RETRIES must be >= 0")

    if MAX_EXTRACTION_RETRIES < 0:
        errors.append("MAX_EXTRACTION_RETRIES must be >= 0")

    if MAX_PIPELINE_ITERATIONS < 1:
        errors.append("MAX_PIPELINE_ITERATIONS must be >= 1")

    # Check thresholds
    if not (0 <= HALLUCINATION_THRESHOLD <= 1):
        errors.append("HALLUCINATION_THRESHOLD must be between 0 and 1")

    if EXPRESSION_TPM_TOLERANCE < 0:
        errors.append("EXPRESSION_TPM_TOLERANCE must be >= 0")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))


# Validate on import
validate_config()
