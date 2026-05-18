"""
ClaimExtractor Agent

Extracts discrete, testable factual claims from AI-generated summaries.
Uses 1:1 sentence-to-claim mapping for precise validation and refinement.
"""

import re
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dataclasses import dataclass
import json

from src.validation_config.validation_config import (
    CLAIM_EXTRACTOR_TEMPERATURE,
    EXPECTED_SENTENCES,
    MIN_SENTENCE_WORDS,
    MAX_CONCURRENT_VALIDATIONS,
)
from src.config import get_active_model
from src.graph.state import _accumulate_tokens


class ExtractedClaim(BaseModel):
    """A single factual claim extracted from a summary."""

    claim_id: str = Field(
        description="Unique identifier for this claim"
    )
    claim_text: str = Field(
        description="The exact factual claim as a single, testable statement"
    )
    section: str = Field(
        description="Which summary section this claim came from"
    )
    claim_type: Literal[
        "database_fact",
        "expression_pattern",
        "go_annotation",
        "protein_interaction",
        "literature_finding",
        "cross_gene_relationship",
        "quantitative_statement",
        "biological_interpretation",
        "reasonable_inference",
        "other"
    ] = Field(
        description="Category of the claim based on what data source should verify it"
    )
    original_sentence: Optional[str] = Field(
        default=None,
        description="The original sentence from which this claim was extracted"
    )


class ClaimExtractionResult(BaseModel):
    """Result of claim extraction from a summary section."""

    section: str = Field(description="The section that was analyzed")
    claims: List[ExtractedClaim] = Field(
        default_factory=list,
        description="List of extracted claims"
    )
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Notes about the extraction process"
    )


@dataclass
class ClaimExtractorDeps:
    """Dependencies for claim extraction."""
    gene_symbol: str
    section_type: str


# Lazy initialization
_claim_extractor = None

# Semaphore for rate limiting parallel extractions
_extraction_semaphore: asyncio.Semaphore = None


def _get_extraction_semaphore() -> asyncio.Semaphore:
    """
    Get or create the extraction semaphore (lazy initialization).

    Recreates the semaphore if the current event loop differs from the one
    it was created in, preventing 'bound to a different event loop' errors.
    """
    global _extraction_semaphore

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running - create semaphore anyway
        current_loop = None

    # Check if semaphore needs to be created or recreated
    if _extraction_semaphore is None:
        _extraction_semaphore = asyncio.Semaphore(MAX_CONCURRENT_VALIDATIONS)
    elif current_loop is not None:
        # Check if semaphore is bound to a different loop
        try:
            # Try to get the semaphore's loop
            semaphore_loop = getattr(_extraction_semaphore, '_loop', None)
            if semaphore_loop is not None and semaphore_loop is not current_loop:
                # Recreate semaphore for current loop
                _extraction_semaphore = asyncio.Semaphore(MAX_CONCURRENT_VALIDATIONS)
        except Exception:
            # If we can't check, recreate to be safe
            _extraction_semaphore = asyncio.Semaphore(MAX_CONCURRENT_VALIDATIONS)

    return _extraction_semaphore


_claim_extractor_model: str = ""


def get_claim_extractor():
    """Get or create the claim extractor agent, rebuilding if the active model changed."""
    global _claim_extractor, _claim_extractor_model
    current_model = get_active_model()
    if _claim_extractor is None or _claim_extractor_model != current_model:
        _claim_extractor_model = current_model
        openai_model = OpenAIChatModel(current_model)
        _claim_extractor = Agent(
            openai_model,
            deps_type=ClaimExtractorDeps,
            model_settings={
                'temperature': CLAIM_EXTRACTOR_TEMPERATURE,
                'max_completion_tokens': 2000,
            },
            system_prompt="""You are an expert at extracting and categorizing claims from scientific text.

=== KEY PRINCIPLE: ONE SENTENCE = ONE CLAIM ===

Extract exactly ONE claim from each sentence. The claim should capture the core statement.

=== CLAIM CATEGORIES (CRITICAL - CHOOSE CAREFULLY) ===

**FACTUAL CATEGORIES** (require direct data support):
- database_fact: Basic gene info (name, chromosome, gene type) - ONLY explicit database fields
- expression_pattern: Tissue expression with SPECIFIC TPM values mentioned
- go_annotation: Explicit GO term names mentioned verbatim
- protein_interaction: Explicit partner names and interaction counts
- quantitative_statement: Specific numbers, percentages, fold-changes, counts

**INTERPRETIVE CATEGORIES** (scientific reasoning, inference):
- biological_interpretation: Mechanistic reasoning, pathway inferences, functional implications
  * Look for: "suggests", "indicates", "consistent with", "may play a role", "enabling", "facilitating"
  * Example: "This positioning indicates involvement in the lactose digestion pathway"
- reasonable_inference: Logical conclusions drawn from combining multiple data points
  * Look for: "therefore", "thus", "underscores", "reflects", "aligning with"
  * Example: "The higher expression reflects greater contractile demands"
- literature_finding: Research themes, study conclusions, paper findings
- cross_gene_relationship: Relationships or comparisons between genes

**OTHER**:
- other: Claims that don't fit other categories

=== CRITICAL CATEGORIZATION RULES ===

1. If a sentence contains hedging language ("may", "suggests", "could", "indicates", "consistent with"),
   it is almost always "biological_interpretation" or "reasonable_inference", NOT a factual category.

2. If a sentence explains WHY something happens or WHAT IT MEANS biologically, use "biological_interpretation".

3. If a sentence states a specific number (TPM, count, percentage), use the appropriate factual category.

4. When in doubt between factual and interpretive, choose INTERPRETIVE. This protects scientific reasoning.

=== OUTPUT FORMAT ===

{
  "section": "section_name",
  "claims": [
    {
      "claim_id": "provided_id",
      "claim_text": "The claim from this sentence",
      "section": "section_name",
      "claim_type": "category",
      "original_sentence": "The original sentence"
    }
  ],
  "extraction_notes": "Notes if any"
}

Return ONLY valid JSON.
"""
        )
    return _claim_extractor


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, filtering out fragments.

    Args:
        text: The summary text to split

    Returns:
        List of valid sentences (filtered by MIN_SENTENCE_WORDS)
    """
    # Split on sentence-ending punctuation
    # Handle common abbreviations and decimal numbers
    text = text.replace('et al.', 'et al@')
    text = text.replace('e.g.', 'e@g@')
    text = text.replace('i.e.', 'i@e@')

    # Split on . ! ? but keep the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Restore abbreviations and clean up
    cleaned = []
    for s in sentences:
        s = s.replace('et al@', 'et al.')
        s = s.replace('e@g@', 'e.g.')
        s = s.replace('i@e@', 'i.e.')
        s = s.strip()

        # Filter fragments
        word_count = len(s.split())
        if word_count >= MIN_SENTENCE_WORDS:
            cleaned.append(s)

    return cleaned


async def extract_claims(
    summary_text: str,
    section_type: str,
    gene_symbol: str,
    *,
    state=None,
    node_name: str = "ClaimExtractor"
) -> ClaimExtractionResult:
    """
    Extract factual claims from an AI-generated summary section.

    Uses 1:1 sentence-to-claim mapping: each sentence becomes one claim.

    Args:
        summary_text: The AI-generated summary text to analyze
        section_type: Type of section (e.g., "functional_summary", "expression_interpretation")
        gene_symbol: The gene being described

    Returns:
        ClaimExtractionResult with list of extracted claims (one per sentence)
    """
    if not summary_text or not summary_text.strip():
        return ClaimExtractionResult(
            section=section_type,
            claims=[],
            extraction_notes="Empty summary text provided"
        )

    stripped_text = summary_text.strip()
    placeholder_prefixes = (
        "[LOW_EVIDENCE]",
        "[LOW_EVIDENCE",
        "[NO_DATA]",
        "[NO_DATA",
    )

    # Handle LOW_EVIDENCE / NO_DATA markers - single existence check, no full extraction needed
    if any(stripped_text.upper().startswith(prefix) for prefix in placeholder_prefixes):
        closing_idx = stripped_text.find(']')
        clean_text = stripped_text[closing_idx + 1:].strip() if closing_idx != -1 else stripped_text
        clean_text = clean_text or stripped_text

        # Return a single existence claim that's pre-validated as accurate
        # (the marker itself confirms the data absence was verified)
        existence_claim = ExtractedClaim(
            claim_id=f"{gene_symbol}_{section_type[:4]}_existence",
            claim_text=clean_text,
            section=section_type,
            claim_type="database_fact",
            original_sentence=clean_text
        )

        return ClaimExtractionResult(
            section=section_type,
            claims=[existence_claim],
            extraction_notes="Single existence check - LOW_EVIDENCE marker detected"
        )

    # Split summary into sentences
    sentences = split_into_sentences(summary_text)

    if not sentences:
        fallback_text = stripped_text
        if not fallback_text:
            return ClaimExtractionResult(
                section=section_type,
                claims=[],
                extraction_notes="No valid sentences found in summary"
            )

        fallback_claim = ExtractedClaim(
            claim_id=f"{gene_symbol}_{section_type[:4]}_fallback",
            claim_text=fallback_text,
            section=section_type,
            claim_type="other",
            original_sentence=fallback_text
        )

        return ClaimExtractionResult(
            section=section_type,
            claims=[fallback_claim],
            extraction_notes="Fallback claim generated from short section"
        )

    # Generate claim IDs for each sentence
    sentence_list = "\n".join([
        f"{i+1}. [ID: {gene_symbol}_{section_type[:4]}_{i+1:03d}] {s}"
        for i, s in enumerate(sentences)
    ])

    deps = ClaimExtractorDeps(
        gene_symbol=gene_symbol,
        section_type=section_type
    )

    # Detect multi-gene input via comma presence
    is_multi_gene = "," in gene_symbol

    if is_multi_gene:
        intro_line = f"Extract ONE factual claim from EACH sentence below. These claims describe relationships across the following genes: {gene_symbol}."
    else:
        intro_line = f"Extract ONE factual claim from EACH sentence below for gene {gene_symbol}."

    prompt = f"""{intro_line}

=== SECTION: {section_type} ===

=== SENTENCES (extract one claim per sentence) ===

{sentence_list}

=== INSTRUCTIONS ===

For each numbered sentence:
1. Extract the core factual claim
2. Use the provided claim ID
3. Categorize by data source needed to verify it

Return {len(sentences)} claims total (one per sentence).
Return ONLY valid JSON.
"""

    extractor = get_claim_extractor()
    result = await extractor.run(prompt, deps=deps)

    # Track token usage
    usage = getattr(getattr(result, "response", None), "usage", None)
    _accumulate_tokens(state, node_name, usage)

    try:
        # Extract output from agent result
        if hasattr(result, 'output'):
            response_text = result.output
        elif hasattr(result, 'data'):
            response_text = result.data
        else:
            response_text = str(result)

        # Clean up markdown code blocks
        response_text = str(response_text).strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Parse JSON
        data = json.loads(response_text)

        # Ensure all claims have proper IDs and original sentences
        claims = []
        for i, claim_data in enumerate(data.get('claims', [])):
            if not claim_data.get('claim_id'):
                claim_data['claim_id'] = f"{gene_symbol}_{section_type[:4]}_{i+1:03d}"
            if not claim_data.get('section'):
                claim_data['section'] = section_type
            # Map back to original sentence if not provided
            if not claim_data.get('original_sentence') and i < len(sentences):
                claim_data['original_sentence'] = sentences[i]
            claims.append(ExtractedClaim(**claim_data))

        return ClaimExtractionResult(
            section=section_type,
            claims=claims,
            extraction_notes=f"Extracted {len(claims)} claims from {len(sentences)} sentences"
        )

    except json.JSONDecodeError as e:
        return ClaimExtractionResult(
            section=section_type,
            claims=[],
            extraction_notes=f"JSON parsing failed: {str(e)}"
        )
    except Exception as e:
        return ClaimExtractionResult(
            section=section_type,
            claims=[],
            extraction_notes=f"Extraction failed: {str(e)}"
        )


async def _extract_claims_with_semaphore(
    summary_text: str,
    section_type: str,
    gene_symbol: str,
) -> ClaimExtractionResult:
    """Wrap extract_claims with semaphore for rate limiting."""
    semaphore = _get_extraction_semaphore()
    async with semaphore:
        return await extract_claims(
            summary_text=summary_text,
            section_type=section_type,
            gene_symbol=gene_symbol
        )


async def extract_all_claims(
    summaries: dict,
    gene_symbol: str,
    sections_to_validate: List[str] = None
) -> List[ClaimExtractionResult]:
    """
    Extract claims from multiple summary sections in parallel.

    Args:
        summaries: Dictionary mapping section names to summary texts
        gene_symbol: The gene being analyzed
        sections_to_validate: List of section names to process (None = all)

    Returns:
        List of ClaimExtractionResult, one per section
    """
    # Build list of sections to process
    sections_to_process = []
    for section_name, summary_text in summaries.items():
        # Skip if not in validation list
        if sections_to_validate and section_name not in sections_to_validate:
            continue

        # Skip empty summaries
        if not summary_text or not str(summary_text).strip():
            continue

        sections_to_process.append((section_name, str(summary_text)))

    if not sections_to_process:
        return []

    # Create parallel extraction tasks
    extraction_tasks = [
        _extract_claims_with_semaphore(
            summary_text=summary_text,
            section_type=section_name,
            gene_symbol=gene_symbol
        )
        for section_name, summary_text in sections_to_process
    ]

    # Run all extractions in parallel
    results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

    # Filter out exceptions (return empty result for failed extractions)
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            section_name = sections_to_process[i][0]
            valid_results.append(ClaimExtractionResult(
                section=section_name,
                claims=[],
                extraction_notes=f"Extraction failed: {str(result)}"
            ))
        else:
            valid_results.append(result)

    return valid_results
