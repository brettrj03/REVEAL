"""
Agent-based Gene Extraction with Experimental Context Parsing

Uses LLM to intelligently extract gene names and experimental metadata
from natural language queries.
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic import BaseModel, Field
from typing import List, Optional
from dataclasses import dataclass
import json
import re

from src.graph.state import _accumulate_tokens


class GeneExtractionResult(BaseModel):
    """Structured output from gene extraction agent with categorized identifiers"""

    # Categorized identifier fields (new)
    gene_symbols: List[str] = Field(
        default_factory=list,
        description="Standard gene symbols (e.g., ['BRCA1', 'TP53', 'EGFR'])"
    )

    gene_ensembl_ids: List[str] = Field(
        default_factory=list,
        description="Ensembl gene IDs (e.g., ['ENSG00000012048', 'ENSG00000141510'])"
    )

    gene_full_names: List[str] = Field(
        default_factory=list,
        description="Full descriptive gene names (e.g., ['tumor protein p53', 'clusterin'])"
    )

    gene_aliases: List[str] = Field(
        default_factory=list,
        description="Known or suspected gene aliases (e.g., ['p53', 'BRC1'])"
    )

    # Combined list for backward compatibility
    genes: List[str] = Field(
        default_factory=list,
        description="Combined list of all gene identifiers (for backward compatibility)"
    )

    organism: Optional[str] = Field(
        default="human",
        description="Organism studied (e.g., 'human', 'mouse', 'rat', 'zebrafish')"
    )

    cell_type: Optional[str] = Field(
        default=None,
        description="Cell type or tissue studied (e.g., 'HeLa cells', 'T cells', 'liver', 'hippocampus')"
    )

    treatment: Optional[str] = Field(
        default=None,
        description="Treatment or condition applied (e.g., 'RSPO4', 'hypoxia', 'drug X')"
    )

    timepoint: Optional[str] = Field(
        default=None,
        description="Timepoint of measurement (e.g., '24h', '48 hours', 'day 7')"
    )

    comparison: Optional[str] = Field(
        default=None,
        description="What is being compared (e.g., 'treated vs control', 'tumour vs normal', 'before vs after')"
    )

    hypothesis: Optional[str] = Field(
        default=None,
        description="The biological hypothesis or question being investigated"
    )

    confidence: float = Field(
        default=1.0,
        description="Confidence in extraction (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    notes: Optional[str] = Field(
        default=None,
        description="Any notes about ambiguities or extraction issues"
    )


@dataclass
class ExtractorDeps:
    """Dependencies for the extraction agent"""
    query: str


from src.config import EXTRACTION_TEMPERATURE, EXTRACTION_MAX_TOKENS, get_active_model

# Lazy initialization to avoid requiring API key at import time
_gene_extractor = None
_gene_extractor_model: str = ""

def get_gene_extractor():
    """Get or create the gene extractor agent, rebuilding if the active model changed."""
    global _gene_extractor, _gene_extractor_model
    current_model = get_active_model()
    if _gene_extractor is None or _gene_extractor_model != current_model:
        _gene_extractor_model = current_model
        openai_model = OpenAIChatModel(current_model)
        _gene_extractor = Agent(
            openai_model,
            deps_type=ExtractorDeps,
            model_settings={
                'temperature': EXTRACTION_TEMPERATURE,
                'max_tokens': EXTRACTION_MAX_TOKENS,
            },
            system_prompt="""
You are an expert bioinformatics assistant specializing in parsing scientific queries about gene expression experiments.

Your task is to extract:
1. **Gene names** - All gene symbols mentioned
2. **Experimental context** - Organism, cell type, treatment, timepoint, comparison, hypothesis

=== GENE EXTRACTION RULES ===

**What to extract as genes:**
- Standard gene symbols (e.g., TP53, BRCA1, EGFR, LEF1, FOXQ1)
- Mouse genes (e.g., Tp53, Brca1, Slc12a2)
- Gene families with numbers (e.g., CD4, CD8A, HLA-A, IL2)
- Hyphenated genes (e.g., NF-KB, IL-6)

**What NOT to extract as genes:**
- "DEG" or "DEGs" (differentially expressed genes - this is NOT a gene name)
- "gene" or "genes" (generic term)
- Analysis terms: "RNA-seq", "sequencing", "analysis", "expression"
- Descriptors: "upregulated", "downregulated", "increased", "decreased"
- Tissue/cell types used descriptively: "kidney", "liver", "cortex" (unless clearly a gene)
- Statistical terms: "fold change", "p-value", "significant"
- Protein isoforms or variants (see below)

=== IDENTIFIER TYPE CATEGORIZATION ===

Categorize extracted identifiers into the appropriate field:

**gene_ensembl_ids** - Ensembl gene IDs:
- Pattern: ENSG followed by 11 digits (e.g., ENSG00000012048, ENSG00000141510)
- Always extract when this exact pattern appears
- Example: "ENSG00000139618" → gene_ensembl_ids: ["ENSG00000139618"]

**gene_symbols** - Standard gene symbols:
- Standard uppercase symbols: TP53, BRCA1, EGFR, MYC
- Mouse genes: Tp53, Brca1, Slc12a2
- Hyphenated: HLA-A, IL-6, NF-KB
- Numbers allowed: CD4, CD8A, IL2
- Example: "BRCA1 and TP53" → gene_symbols: ["BRCA1", "TP53"]

**gene_full_names** - Descriptive gene names:
- Multi-word descriptive phrases that refer to genes
- Examples: "tumor protein p53", "clusterin", "breast cancer type 1"
- Usually contain multiple words describing the gene's function
- Example: "the gene for clusterin" → gene_full_names: ["clusterin"]
- Example: "tumor suppressor protein p53" → gene_full_names: ["tumor protein p53"]

**gene_aliases** - Alternative/legacy names:
- Short informal names: "p53" (for TP53), "BRC1" (for BRCA1)
- Legacy symbols, historical names
- Names that look like symbols but might be aliases
- Example: "p53 expression" → gene_aliases: ["p53"]

**genes** - Combined list (for backward compatibility):
- Populate with ALL identifiers from all categories combined
- Include: gene_symbols + gene_ensembl_ids + gene_full_names + gene_aliases
- Remove duplicates while preserving order

**Protein Isoforms and Variants (CRITICAL - DO NOT extract as separate genes):**
- **APOE isoforms**: E2, E3, E4, ε2, ε3, ε4, epsilon2, epsilon3, epsilon4
  → These are protein variants of APOE, NOT separate genes
  → If detected, extract only "APOE" and note the isoforms in notes field
  → Example: "APOE E2, E3, E4" → Extract ["APOE"], note "Query mentions APOE isoforms"

- **TP53 variants**: p53, P53, p-53
  → These refer to TP53 protein, NOT separate genes
  → Extract as "TP53"

- **Single letter + number combinations**: E1, E2, E3, E4, A1, B2, C3, etc.
  → These are often enzyme subunits, isoforms, or very ambiguous
  → Do NOT extract unless clearly used as standalone gene symbols with strong context
  → Example: "E2 expression" alone → Too ambiguous, do NOT extract
  → Example: "IL2 expression in T cells" → Clear gene context, DO extract

**Isoform Detection and Handling:**
If the query discusses protein isoforms:
1. Extract ONLY the parent gene name (e.g., "APOE" not "E2, E3, E4")
2. Add detailed note explaining the isoform context
3. Include isoform information in comparison or hypothesis fields if relevant
4. Set confidence to 1.0 if gene identification is clear

Example isoform phrases to watch for:
- "isoforms E2, E3, E4"
- "alleles ε2, ε3, ε4"
- "variants epsilon2, epsilon3, epsilon4"
- "polymorphism at codons 112 and 158"
- "three major isoforms"

**Ambiguous cases:**
- **Gene vs. tissue/cell type** - Use context:
  - "Expression of CD4 in T cells" → CD4 is a gene ✓
  - "Isolated from kidney cortex" → cortex is NOT a gene ✗

- **Protein isoforms vs. genes**:
  - "APOE E2, E3, E4 polymorphism" → Extract only ["APOE"], note isoforms ✓
  - "The E4 allele increases risk" → Extract ["APOE"] if apolipoprotein mentioned, else note ambiguity
  - "E2 expression in liver" → If no other context, mark confidence <0.7, note ambiguity

- **Very short symbols (1-2 characters + number)**:
  - Check if it's part of a gene family clearly mentioned
  - If standalone without clear gene context → likely NOT a gene
  - Examples:
    - "E2" alone → Too ambiguous ✗
    - "IL2 in immune response" → Clear gene context ✓
    - "CD8A and CD4 in T cells" → Clear gene context ✓

**Capitalization guidance:**
- Human genes: Usually all caps (TP53, BRCA1) or mixed (HoxA1)
- Mouse genes: Usually start with capital then lowercase (Tp53, Brca1)
- If unsure, preserve the exact capitalization from the query

=== CRITICAL: COMMA-SEPARATED GENE LISTS ===

**If the user query contains a comma-separated list of gene symbols, you MUST extract EVERY SINGLE item in that list without exception:**
- Do NOT skip, filter, or omit any genes from an explicit list
- Do NOT filter based on whether you recognize the gene name
- Unknown or unfamiliar gene symbols MUST still be extracted
- lncRNAs, pseudogenes, and non-protein-coding genes MUST be extracted just like protein-coding genes
- After extracting, COUNT the genes in your output and VERIFY it matches the number of genes in the input list

**Examples of comma-separated lists:**
- "What is the role of MED12, F8A2, F8A3, PCDHA6, RGPD1, ADPRHL1, PEG3, MIMT1, ZIM2, PCDHGA3, and EOMES in..."
  → MUST extract ALL 11 genes, even if some are unfamiliar (MIMT1 is an lncRNA, PCDHGA3 is protein-coding)

- "Expression of GENE1, GENE2, and GENE3"
  → MUST extract all 3 genes

- "Analyzing TP53, BRCA1, MYC, UNKNOWNGENE"
  → MUST extract all 4 genes, including UNKNOWNGENE even if you don't recognize it

**Gene type inclusion:**
- Protein-coding genes: Extract ✓
- lncRNAs: Extract ✓
- Pseudogenes: Extract ✓
- miRNAs: Extract ✓
- snoRNAs: Extract ✓
- Any gene with a valid symbol: Extract ✓

=== EXAMPLES ===

[... keep existing examples 1-4 ...]

**Example 5 (ISOFORM HANDLING - CRITICAL):**
Query: "The apolipoprotein E gene encodes a multifunctional glycoprotein with three major isoforms (E2, E3, E4) arising from single nucleotide polymorphisms at codons 112 and 158. The E4 allele confers substantially elevated risk for late-onset Alzheimer's disease."

Response:
{
  "genes": ["APOE"],
  "organism": "human",
  "cell_type": null,
  "treatment": null,
  "timepoint": null,
  "comparison": "E2 vs E3 vs E4 isoforms",
  "hypothesis": "APOE E4 isoform increases Alzheimer's disease risk compared to E2 and E3 isoforms",
  "confidence": 1.0,
  "notes": "Query describes APOE protein isoforms (E2, E3, E4). These are variants of the APOE protein encoded by a single gene (APOE), not separate genes. Isoforms differ by amino acid substitutions at positions 112 and 158 due to SNPs. E2, E3, E4 were NOT extracted as separate genes."
}

**Example 6 (AMBIGUOUS SHORT CODES):**
Query: "E2 and E3 enzyme activities were measured in hepatocytes"

Response:
{
  "genes": [],
  "organism": "human",
  "cell_type": "hepatocytes",
  "treatment": null,
  "timepoint": null,
  "comparison": "E2 vs E3 enzyme activity",
  "hypothesis": "Comparing E2 and E3 enzyme activities",
  "confidence": 0.6,
  "notes": "E2 and E3 are ambiguous. Could be: (1) APOE isoforms, (2) enzyme subunit designations, (3) estrogen metabolites, or (4) other proteins. Without additional context mentioning apolipoprotein or specific enzyme names, cannot confidently identify gene symbols. No genes extracted."
}

=== CRITICAL REMINDERS ===

1. **NEVER extract "DEG" or "DEGs" as a gene name**
2. **NEVER extract protein isoforms (E2, E3, E4, ε2, ε3, ε4) as separate genes**
3. **ALWAYS check if a capitalized word is a gene or just a descriptive term**
4. **For isoforms: Extract parent gene only, explain in notes**
5. **Preserve exact capitalization from the query for gene names**
6. **Extract ALL genes mentioned, not just the most prominent ones**
7. **COMMA-SEPARATED LISTS: Extract EVERY gene from explicit lists - do not skip unfamiliar or non-coding genes**
8. **COUNT YOUR GENES: After extraction, verify gene count matches the input list count**
9. **INCLUDE ALL GENE TYPES: lncRNAs, pseudogenes, miRNAs must be extracted just like protein-coding genes**
10. **Be conservative with confidence scores - if unsure, score lower**
11. **Use notes field to explain any ambiguities, especially for isoforms**
12. **ALWAYS respond with valid JSON only - no other text**
13. **Categorize identifiers correctly into gene_symbols, gene_ensembl_ids, gene_full_names, gene_aliases**
14. **Always populate the 'genes' field with all identifiers combined**

=== OUTPUT FORMAT ===

Return a JSON object with these fields:
{
  "gene_symbols": ["BRCA1", "TP53"],      // Standard gene symbols
  "gene_ensembl_ids": ["ENSG00000139618"], // Ensembl IDs found
  "gene_full_names": ["clusterin"],        // Descriptive names
  "gene_aliases": ["p53"],                 // Alternative names
  "genes": ["BRCA1", "TP53", "ENSG00000139618", "clusterin", "p53"],  // All combined
  "organism": "human",
  "cell_type": null,
  "treatment": null,
  "timepoint": null,
  "comparison": null,
  "hypothesis": null,
  "confidence": 1.0,
  "notes": null
}

Now extract genes and experimental context from the provided query.
"""
            )
    return _gene_extractor


def _detect_explicit_gene_list(query: str) -> Optional[List[str]]:
    """
    Detect if query contains an explicit comma-separated list of all-caps gene symbols.

    This is a direct regex-based parser for "factual mode" queries where the user
    provides an explicit, unambiguous list of genes.

    Returns:
        List of gene symbols if an explicit list is detected, None otherwise.

    Examples that should be detected:
        - "MED12, F8A2, F8A3, PCDHA6, RGPD1, ADPRHL1, PEG3, MIMT1, ZIM2, PCDHGA3, and EOMES"
        - "TP53, BRCA1, MYC"
        - "genes: GENE1, GENE2, GENE3"
    """
    # Pattern for comma-separated all-caps gene-like tokens
    # Looks for at least 2 gene symbols separated by commas (with optional "and")
    # Gene symbols must be 2-15 chars, start with letter, can contain letters/numbers/hyphens
    explicit_list_patterns = [
        # Pattern 1: "GENE1, GENE2, GENE3" or "GENE1, GENE2 and GENE3"
        r'(?<![A-Z0-9-])([A-Z][A-Z0-9-]{1,14})(?:\s*,\s*([A-Z][A-Z0-9-]{1,14}))+(?:\s*,?\s*(?:and|&)\s+([A-Z][A-Z0-9-]{1,14}))?',

        # Pattern 2: "genes: X, Y, Z" or "DEGs: A, B, C"
        r'(?:genes?|DEGs?|symbols?)[:\s]+([A-Z][A-Z0-9-]{1,14}(?:\s*,\s*[A-Z][A-Z0-9-]{1,14})+(?:\s*,?\s*(?:and|&)\s+[A-Z][A-Z0-9-]{1,14})?)',
    ]

    for pattern in explicit_list_patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            # Extract all gene-like tokens from the matched region
            matched_text = match.group(0)
            # Find all all-caps gene-like tokens in the match (including numbers)
            gene_tokens = re.findall(r'(?<![A-Z0-9-])([A-Z][A-Z0-9-]{1,14})(?![A-Z0-9-])', matched_text)

            # Filter out common non-gene words
            filtered_genes = []
            for token in gene_tokens:
                # Skip stopwords
                if token in _GENE_STOPWORDS:
                    continue
                # Skip common analysis terms
                if token in {'AND', 'OR', 'THE', 'ARE', 'WITH', 'FROM', 'TO', 'IN', 'OF', 'FOR'}:
                    continue
                # Skip technical terms
                if token in {'RNA', 'SEQ', 'DNA', 'PCR', 'DEGS', 'DEG', 'GENES', 'GENE', 'SYMBOLS', 'SYMBOL'}:
                    continue
                filtered_genes.append(token)

            # If we found at least 2 genes, consider this an explicit list
            if len(filtered_genes) >= 2:
                print(f"\n  ✓ Detected explicit gene list with {len(filtered_genes)} genes via direct regex parsing")
                print(f"    Bypassing LLM extraction for: {', '.join(filtered_genes[:10])}")
                if len(filtered_genes) > 10:
                    print(f"    ... and {len(filtered_genes) - 10} more")
                return filtered_genes

    return None


def _validate_extraction_completeness(query: str, extracted_genes: List[str]) -> None:
    """
    Post-extraction validation to warn about potentially missing genes.

    Checks if gene-like tokens in the query (all-caps 2-10 character strings)
    appear in the extracted gene list. Logs warnings for any that are missing.

    Args:
        query: The original user query
        extracted_genes: List of genes extracted by the LLM
    """
    # Extract potential gene-like tokens from query (all-caps strings 2-10 chars)
    # Updated pattern to explicitly include numbers (e.g., MIMT1, CD8A, HLA-A)
    gene_like_pattern = r'(?<![A-Z0-9-])([A-Z][A-Z0-9-]{1,14})(?![A-Z0-9-])'
    potential_genes = re.findall(gene_like_pattern, query)

    # Normalize extracted genes for case-insensitive comparison
    extracted_upper = set(g.upper() for g in extracted_genes)

    # Common non-gene words to skip
    skip_words = _GENE_STOPWORDS | {
        'RNA', 'SEQ', 'DNA', 'PCR', 'RT', 'QPCR', 'WB', 'IHC', 'FC', 'FDR', 'LOG',
        'VS', 'VS.', 'FIG', 'TABLE', 'CTRL', 'WT', 'KO', 'OE', 'KD', 'CRISPR'
    }

    # Check for missing gene-like tokens
    missing_tokens = []
    for token in potential_genes:
        if token in skip_words:
            continue
        if token not in extracted_upper:
            missing_tokens.append(token)

    # Log warnings for missing tokens
    if missing_tokens:
        print(f"\n⚠️  POST-EXTRACTION VALIDATION WARNING:")
        print(f"    The following gene-like tokens from the query were NOT extracted:")
        for token in missing_tokens:
            print(f"      - {token}")
        print(f"    Query: {query[:100]}...")
        print(f"    Extracted: {', '.join(extracted_genes)}")
        print(f"    If these are genes, they may have been incorrectly filtered by the LLM.\n")


async def extract_genes_with_context(
    query: str,
    *,
    state=None,
    node_name: str = "ExtractGenesStateful"
) -> GeneExtractionResult:
    """
    Extract genes and experimental context from a natural language query

    Uses BOTH direct regex parsing AND LLM extraction, then merges results.
    This ensures maximum gene coverage - if one method misses a gene, the other catches it.

    Args:
        query: User's natural language query about genes
        state: Optional state object for token tracking
        node_name: Name of the calling node for token tracking

    Returns:
        GeneExtractionResult with genes and experimental metadata
    """

    # ================================================================
    # STEP 1: Direct regex parsing for explicit gene lists
    # ================================================================
    # Detect comma-separated lists but DO NOT return early - continue to LLM
    regex_genes = _detect_explicit_gene_list(query)

    deps = ExtractorDeps(query=query)

    prompt = f"""Extract all gene identifiers and experimental context from this query:

"{query}"

CRITICAL REQUIREMENTS:
- If the query contains a COMMA-SEPARATED LIST of genes, extract EVERY SINGLE gene in that list without exception
- Do NOT skip, filter, or omit genes based on familiarity or recognition
- Extract lncRNAs, pseudogenes, and non-coding genes just like protein-coding genes
- After extraction, COUNT your genes and verify the count matches any explicit list in the query
- Extract ALL gene identifiers mentioned (don't miss any!)

Categorization:
- gene_symbols: Standard symbols like BRCA1, TP53, EGFR, MIMT1, PCDHGA3
- gene_ensembl_ids: IDs starting with ENSG followed by numbers
- gene_full_names: Descriptive phrases like "tumor protein p53", "clusterin"
- gene_aliases: Alternative names like "p53" (for TP53)
- Populate 'genes' with ALL identifiers combined (for backward compatibility)

Remember:
- "DEG" or "DEGs" is NOT a gene name
- Include organism, cell type, treatment, and other context
- Set confidence based on clarity of the query
- Use notes to explain any uncertainties

Respond with ONLY a JSON object, no other text.
"""

    try:
        gene_extractor = get_gene_extractor()
        result = await gene_extractor.run(prompt, deps=deps)

        # Track token usage
        usage = getattr(getattr(result, "response", None), "usage", None)
        _accumulate_tokens(state, node_name, usage)
    except ModelHTTPError as e:
        reason = _describe_model_http_error(e)
        print(f"    Warning: {reason} -- using heuristic fallback")
        return _fallback_gene_extraction(query, reason, regex_genes)
    except Exception as e:
        if _should_use_fallback(e):
            reason = f"LLM unavailable ({str(e)})"
            print(f"    Warning: {reason} -- using heuristic fallback")
            return _fallback_gene_extraction(query, reason, regex_genes)
        raise

    # Parse the JSON response
    try:
        # Extract the output from AgentRunResult
        if hasattr(result, 'output'):
            response_text = result.output
        elif hasattr(result, 'data'):
            response_text = result.data
        else:
            response_text = str(result)

        # Clean up the response if it has markdown code blocks
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

        # Ensure categorized fields exist with defaults
        data.setdefault('gene_symbols', [])
        data.setdefault('gene_ensembl_ids', [])
        data.setdefault('gene_full_names', [])
        data.setdefault('gene_aliases', [])

        # Build combined genes list if not provided or incomplete
        # Combine all categorized identifiers, deduplicating while preserving order
        all_identifiers = (
            data.get('gene_symbols', []) +
            data.get('gene_ensembl_ids', []) +
            data.get('gene_full_names', []) +
            data.get('gene_aliases', [])
        )

        # If 'genes' field exists but categorized fields are empty, use genes as gene_symbols (backward compat)
        if data.get('genes') and not any([
            data.get('gene_symbols'),
            data.get('gene_ensembl_ids'),
            data.get('gene_full_names'),
            data.get('gene_aliases')
        ]):
            data['gene_symbols'] = data['genes']
            all_identifiers = data['genes']

        # Deduplicate while preserving order
        seen = set()
        combined_genes = []
        llm_genes = set()  # Track which genes came from LLM

        for identifier in all_identifiers:
            if identifier and identifier not in seen:
                combined_genes.append(identifier)
                seen.add(identifier)
                llm_genes.add(identifier.upper())

        # ================================================================
        # MERGE: Combine regex-detected genes with LLM-extracted genes
        # ================================================================
        # Run BOTH methods and merge - ensures no gene is dropped
        regex_only = []
        llm_only = list(llm_genes)
        both_methods = []

        if regex_genes:
            print(f"\n  Merging regex and LLM extraction results...")
            regex_upper = set(g.upper() for g in regex_genes)

            # Find genes detected by regex but missed by LLM
            for gene in regex_genes:
                gene_upper = gene.upper()
                if gene_upper not in seen:
                    # Regex found it, LLM missed it - add to results
                    combined_genes.append(gene)
                    seen.add(gene_upper)
                    regex_only.append(gene)

                    # Also add to gene_symbols for categorization
                    if gene not in data['gene_symbols']:
                        data['gene_symbols'].append(gene)
                elif gene_upper in llm_genes:
                    # Both methods found it
                    both_methods.append(gene)
                    llm_only.remove(gene_upper) if gene_upper in llm_only else None

            # Log provenance
            print(f"  Gene sources:")
            if both_methods:
                print(f"    Both regex + LLM ({len(both_methods)}): {', '.join(sorted(both_methods))}")
            if llm_only:
                llm_only_formatted = [g for g in combined_genes if g.upper() in llm_only]
                print(f"    LLM only ({len(llm_only_formatted)}): {', '.join(sorted(llm_only_formatted))}")
            if regex_only:
                print(f"    Regex only ({len(regex_only)}): {', '.join(sorted(regex_only))}")
                print(f"    ⚠️  WARNING: {len(regex_only)} gene(s) missed by LLM but caught by regex!")

            # Update notes to reflect merged extraction
            merge_note = f"Merged regex ({len(regex_genes)} genes) + LLM ({len(llm_genes)} genes) = {len(combined_genes)} total genes. "
            if regex_only:
                merge_note += f"Regex caught {len(regex_only)} gene(s) that LLM missed: {', '.join(regex_only)}. "
            if data.get('notes'):
                data['notes'] = merge_note + data['notes']
            else:
                data['notes'] = merge_note.strip()

        # Update combined genes field
        data['genes'] = combined_genes if combined_genes else data.get('genes', [])

        # POST-EXTRACTION VALIDATION: Check for missing gene-like tokens
        _validate_extraction_completeness(query, data['genes'])

        # Create and return the result object
        return GeneExtractionResult(**data)

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response was: {response_text}")
        return _fallback_gene_extraction(query, f"JSON parsing failed ({str(e)})", regex_genes)
    except Exception as e:
        print(f"Unexpected error in extraction: {e}")
        import traceback
        traceback.print_exc()
        return _fallback_gene_extraction(query, f"Extractor error ({str(e)})", regex_genes)


_GENE_STOPWORDS = {
    'A', 'AN', 'ABOUT', 'ABSTRACT', 'AD', 'ADULT', 'ADULTS', 'AL', 'ALL', 'ALSO', 'ANALYSIS', 'AND',
    'ARE', 'ARISING', 'ASSOCIATED', 'ASSOCIATION', 'AT', 'CELL', 'CELLS', 'CHILDREN', 'CLINICAL',
    'COMPARISON', 'CONGENITAL', 'CONTEXT', 'DATA', 'DEG', 'DEGS', 'DELAY', 'DEVELOPMENT',
    'DEVELOPMENTAL', 'DISEASE', 'DISEASES', 'DISORDER', 'DISORDERS', 'DNA', 'DOWN', 'DYSFUNCTION',
    'EFFECT', 'EFFECTS', 'ET', 'ETAL', 'EVIDENCE', 'EXPRESSION', 'FINDINGS', 'FOR', 'FUNCTION',
    'FUNCTIONAL', 'GENE', 'GENES', 'HEALTH', 'HUMAN', 'HUMANS', 'IMPACT', 'IMPACTS', 'IN', 'INTO',
    'INVOLVED', 'INVOLVING', 'IS', 'ITS', 'LINK', 'LINKED', 'LOSS', 'MECHANISM', 'MECHANISMS', 'MICE',
    'MODEL', 'MODELS', 'MOUSE', 'MUTATION', 'MUTATIONS', 'NEURON', 'NEURONS', 'NEURODEVELOPMENTAL',
    'NOVEL', 'OF', 'ON', 'OR', 'PATHWAY', 'PATHWAYS', 'PATIENT', 'PATIENTS', 'PHENOTYPE',
    'PHENOTYPES', 'PROGRESSION', 'PROTEIN', 'PROTEINS', 'QUESTION', 'RELATED', 'RELATIONSHIP',
    'RELEVANCE', 'RESPONSE', 'RESULT', 'RESULTS', 'ROLE', 'ROLES', 'SIGNALING', 'STUDIES', 'STUDY',
    'SYNDROME', 'SYNDROMES', 'TARGET', 'TARGETS', 'THE', 'THERAPY', 'THERAPIES', 'TO', 'TREATED',
    'TREATMENT', 'UNDERSTAND', 'UNDERSTANDING', 'UPREGULATED', 'VERSUS', 'VS', 'WHAT', 'WHICH',
    'WITH', 'WITHOUT'
}

_ORGANISM_KEYWORDS = [
    ('zebrafish', 'zebrafish'),
    ('drosophila', 'fruit fly'),
    ('fruit fly', 'fruit fly'),
    ('mouse', 'mouse'),
    ('murine', 'mouse'),
    ('rat', 'rat'),
    ('macaque', 'macaque'),
    ('yeast', 'yeast'),
    ('arabidopsis', 'arabidopsis'),
    ('plant', 'plant'),
    ('bovine', 'cow'),
    ('canine', 'dog'),
]

_CELL_TYPE_KEYWORDS = [
    ('cortical neuron', 'cortical neurons'),
    ('neuronal', 'neurons'),
    ('neuron', 'neurons'),
    ('astrocyte', 'astrocytes'),
    ('oligodendrocyte', 'oligodendrocytes'),
    ('microglia', 'microglia'),
    ('hepatocyte', 'hepatocytes'),
    ('cardiomyocyte', 'cardiomyocytes'),
    ('fibroblast', 'fibroblasts'),
    ('stem cell', 'stem cells'),
    ('ipsc', 'iPSCs'),
    ('t cell', 'T cells'),
    ('b cell', 'B cells'),
    ('macrophage', 'macrophages'),
    ('dendritic cell', 'dendritic cells'),
]


def _fallback_gene_extraction(query: str, reason: str, regex_genes: Optional[List[str]] = None) -> GeneExtractionResult:
    """
    Heuristic gene extraction used when the LLM call fails.

    Merges regex-detected genes (if available) with heuristic extraction.
    """

    heuristic_genes = _extract_potential_gene_symbols(query)

    # Merge regex genes with heuristic genes if available
    if regex_genes:
        # Use regex genes as primary source, add heuristic genes as backup
        seen = set(g.upper() for g in regex_genes)
        gene_symbols = list(regex_genes)

        for gene in heuristic_genes:
            if gene.upper() not in seen:
                gene_symbols.append(gene)
                seen.add(gene.upper())

        confidence = 0.85  # Higher confidence when we have regex genes
        fallback_note = f"LLM extraction failed ({reason}). Used regex extraction ({len(regex_genes)} genes) + heuristic fallback ({len(heuristic_genes)} genes) = {len(gene_symbols)} total genes."
    else:
        gene_symbols = heuristic_genes
        confidence = 0.35 if gene_symbols else 0.15
        fallback_note = f"Heuristic extraction used because {reason}."

    organism = _infer_organism(query)
    cell_type = _infer_cell_type(query)

    return GeneExtractionResult(
        gene_symbols=gene_symbols,
        gene_ensembl_ids=[],
        gene_full_names=[],
        gene_aliases=[],
        genes=list(gene_symbols),
        organism=organism,
        cell_type=cell_type,
        confidence=confidence,
        notes=fallback_note
    )


def _extract_potential_gene_symbols(query: str) -> List[str]:
    """Lightweight parser for comma-separated or standalone gene-like tokens."""

    if not query:
        return []

    # PRE-PASS: Extract hyphenated gene candidates first (e.g., NETO1-DT, NKX2-3, IL-6)
    # This bypasses stopword filtering for hyphenated tokens
    hyphenated = re.findall(r'[A-Z][A-Z0-9]+-[A-Z0-9]+', query)
    seen = set()
    genes: List[str] = []

    # Add hyphenated genes directly (bypassing stopword filtering)
    for gene in hyphenated:
        if gene not in seen and len(gene) >= 4 and len(gene) <= 15:
            genes.append(gene)
            seen.add(gene)

    candidates = re.findall(r'[A-Za-z0-9-]+', query)

    for token in candidates:
        cleaned = token.strip('-')
        if not cleaned:
            continue
        normalized = cleaned.upper()
        if normalized in seen:
            continue
        if normalized in _GENE_STOPWORDS:
            continue
        if normalized.endswith('S') and normalized[:-1] in _GENE_STOPWORDS:
            continue
        if len(normalized) < 2 or len(normalized) > 12:
            continue
        if normalized.isdigit():
            continue
        if normalized.startswith('HTTP'):
            continue
        if not re.search(r'[A-Z]', normalized):
            continue
        if not re.search(r'[A-Z]{2}', normalized) and not re.search(r'\d', normalized):
            continue
        if normalized in {'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9'}:
            continue

        genes.append(normalized)
        seen.add(normalized)

    return genes


def _infer_organism(query: str) -> str:
    """Guess organism from keywords, defaulting to human."""

    if not query:
        return 'human'

    lower_query = query.lower()
    for keyword, organism in _ORGANISM_KEYWORDS:
        if keyword in lower_query:
            return organism
    return 'human'


def _infer_cell_type(query: str) -> Optional[str]:
    """Basic cell type detection from common keywords."""

    if not query:
        return None

    lower_query = query.lower()
    for keyword, cell_type in _CELL_TYPE_KEYWORDS:
        if keyword in lower_query:
            return cell_type
    return None


def _describe_model_http_error(error: ModelHTTPError) -> str:
    status = getattr(error, 'status_code', 'unknown')
    body = getattr(error, 'body', None)
    body_str = ''
    if isinstance(body, dict):
        msg = body.get('message') or body.get('error') or ''
        code = body.get('code') or ''
        body_parts = [part for part in [msg, code] if part]
        if body_parts:
            body_str = f" ({'; '.join(body_parts)})"
    elif body:
        body_str = f" ({body})"
    return f"OpenAI request failed with status {status}{body_str}".strip()


def _should_use_fallback(error: Exception) -> bool:
    text = str(error).lower()
    keywords = ['api key', 'quota', 'rate limit', 'insufficient', 'unauthorized', '429', '503', 'timeout']
    return any(keyword in text for keyword in keywords)


# ============================================================================
# TESTING
# ============================================================================

async def test_gene_extraction():
    """Test the agent-based extraction"""

    test_cases = [
        "Researchers conducted an RNA-seq experiment on human nail matrix keratinocytes treated with RSPO4 to investigate Wnt/β-catenin signaling dynamics. The analysis revealed increased expression of TWIST1, CRABP1, FOXQ1, and LEF1",

        "RNA sequencing of kidney cortex revealed 4 DEG which were Tspan6, Slc12a2, Prss23 and Miox",

        "What do TP53, BRCA1, and MYC do in cancer?",

        "Analysis of T cell exhaustion markers CD4, CD8A, FOXP3 after 48h PD-1 blockade",

        "Expression of HLA-A, HLA-B, and HLA-C in dendritic cells",

        "The top 5 DEGs were MYC, JUN, FOS, EGR1, and STAT3 in response to serum stimulation",

        "Study of Foxp3 and Cd8a expression in regulatory T cells from spleen tissue",

        # MIMT1 test case - gene with number suffix that should be extracted
        "What is the role of MED12, F8A2, F8A3, PCDHA6, RGPD1, ADPRHL1, PEG3, MIMT1, ZIM2, PCDHGA3, and EOMES in development?",
    ]

    print("\n" + "="*70)
    print("AGENT-BASED GENE EXTRACTION TESTING")
    print("="*70)

    for i, query in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{len(test_cases)}")
        print(f"{'─'*70}")
        print(f"Query: {query}\n")

        try:
            result = await extract_genes_with_context(query)

            print(f"✓ Genes extracted: {', '.join(result.genes)}")
            print(f"  Organism: {result.organism}")
            if result.cell_type:
                print(f"  Cell type: {result.cell_type}")
            if result.treatment:
                print(f"  Treatment: {result.treatment}")
            if result.timepoint:
                print(f"  Timepoint: {result.timepoint}")
            if result.comparison:
                print(f"  Comparison: {result.comparison}")
            if result.hypothesis:
                print(f"  Hypothesis: {result.hypothesis}")
            print(f"  Confidence: {result.confidence:.2f}")
            if result.notes:
                print(f"  Notes: {result.notes}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_gene_extraction())
