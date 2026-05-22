"""
LLM-based biomedical context extraction for literature queries.

Uses an LLM to extract diseases, processes, cell types, and other biomedical
terms from user queries WITHOUT hardcoding. This approach generalizes to any
biomedical domain.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Dict, List, Any
from openai import AsyncOpenAI
import os
from src.config import get_active_model
from src.graph.state import _accumulate_tokens

logger = logging.getLogger(__name__)


class LLMContextExtractor:
    """
    Extract biomedical context using LLM.

    This replaces hardcoded disease lists with flexible LLM-based extraction
    that works for ANY disease, process, or condition.
    """

    def __init__(self, openai_api_key: str | None = None):
        """
        Initialize the extractor.

        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.client = AsyncOpenAI(api_key=api_key)

    async def extract_context(
        self,
        user_query: str,
        experiment_context: Any = None,
        *,
        state=None,
        node_name: str = "BuildLiteratureQueryPlan"
    ) -> Dict[str, List[str]]:
        """
        Extract biomedical context terms from query using LLM.

        Args:
            user_query: User's research question
            experiment_context: Optional experiment context with cell_type, tissue, etc.
            state: Optional state object for token tracking
            node_name: Name of the calling node for token tracking

        Returns:
            Dictionary with keys:
                - diseases: List of disease terms and synonyms
                - processes: List of biological processes
                - cell_types: List of cell types or tissues
                - modifiers: List of important modifiers
                - context_genes: List of gene symbols mentioned as context
        """

        print("\n  Extracting biomedical context with LLM...")

        try:
            # Build prompt
            prompt = self._build_extraction_prompt(user_query, experiment_context)

            # Call LLM
            response = await self.client.chat.completions.create(
                model=get_active_model(),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a biomedical term extraction assistant. "
                                 "Extract comprehensive biomedical terms for PubMed literature search. "
                                 "Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            # Track token usage
            usage = getattr(response, "usage", None)
            _accumulate_tokens(state, node_name, usage)

            # Parse response
            content = response.choices[0].message.content
            context_terms = json.loads(content)

            # Merge with experiment context if available
            if experiment_context:
                if hasattr(experiment_context, 'cell_type') and experiment_context.cell_type:
                    cell_type = str(experiment_context.cell_type).strip()
                    if cell_type and cell_type not in context_terms.get('cell_types', []):
                        context_terms.setdefault('cell_types', []).append(cell_type)

                if hasattr(experiment_context, 'tissue') and experiment_context.tissue:
                    tissue = str(experiment_context.tissue).strip()
                    if tissue and tissue not in context_terms.get('cell_types', []):
                        context_terms.setdefault('cell_types', []).append(tissue)

                if hasattr(experiment_context, 'disease') and experiment_context.disease:
                    disease = str(experiment_context.disease).strip()
                    if disease and disease not in context_terms.get('diseases', []):
                        context_terms.setdefault('diseases', []).append(disease)

            # Log results
            self._log_extraction(context_terms)

            return context_terms

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            print(f"  Warning: LLM extraction failed: {e}")
            print(f"  Falling back to simple keyword extraction")

            # Fallback to simple extraction
            return self._fallback_extraction(user_query, experiment_context)

    def _build_extraction_prompt(
        self,
        user_query: str,
        experiment_context: Any = None
    ) -> str:
        """Build the LLM prompt for context extraction."""

        context_info = ""
        if experiment_context:
            parts = []
            if hasattr(experiment_context, 'cell_type') and experiment_context.cell_type:
                parts.append(f"Cell type: {experiment_context.cell_type}")
            if hasattr(experiment_context, 'tissue') and experiment_context.tissue:
                parts.append(f"Tissue: {experiment_context.tissue}")
            if hasattr(experiment_context, 'species') and experiment_context.species:
                parts.append(f"Species: {experiment_context.species}")
            if hasattr(experiment_context, 'disease') and experiment_context.disease:
                parts.append(f"Disease: {experiment_context.disease}")

            if parts:
                context_info = f"\n\nADDITIONAL CONTEXT:\n{chr(10).join(parts)}"

        prompt = f"""You are a biomedical PubMed query builder. Extract search terms from this research question.

RESEARCH QUESTION:
"{user_query}"{context_info}

══════════════════════════════════════════════════════════════════════
EXTRACTION RULES (READ CAREFULLY)
══════════════════════════════════════════════════════════════════════

1. DISEASES — Named medical conditions, syndromes, disorders
   ✓ Extract: "congenital heart disease", "breast cancer", "tetralogy of Fallot", "Kabuki syndrome"
   ✓ Named syndromes: "Down syndrome", "Rett syndrome", "Angelman syndrome"
   ✓ Expand abbreviations: CHD → "congenital heart disease", "congenital heart defect"
   ✓ Add synonyms: "breast cancer" → also "breast carcinoma"
   ✗ Do NOT include generic words like "disease" or "syndrome" alone
   ✗ Phenotypes (observable symptoms) go in phenotypes, NOT here

2. PHENOTYPES — Observable symptoms, presentations, functional impairments
   ✓ Neurodevelopmental: "neurodevelopmental delay", "intellectual disability", "developmental delay", "cognitive impairment", "learning disability"
   ✓ Behavioural: "autism spectrum disorder", "ASD", "ADHD", "behavioural abnormalities"
   ✓ Neurological: "epilepsy", "seizures", "hypotonia", "ataxia", "spasticity"
   ✓ Dysmorphic: "facial dysmorphism", "microcephaly", "macrocephaly"
   ✓ Functional: "hearing loss", "vision impairment", "feeding difficulties"
   ✓ Add synonyms: "neurodevelopmental delay" → also "developmental delay", "intellectual disability"
   ✗ NOT named syndromes (those are diseases)
   ✗ NOT tissues or anatomical locations
   ✗ NOT generic terms like "disorder", "delay" alone

3. TISSUES — Anatomical organs/tissues (MUST parse from disease/phenotype context!)
   ✓ "congenital heart disease" → tissues: ["heart", "cardiac"]
   ✓ "breast cancer" → tissues: ["breast", "mammary"]
   ✓ "cortical neurons" → tissues: ["brain", "cerebral", "cortical"]
   ✓ "neurodevelopmental delay" → tissues: ["brain", "cerebral", "neural"]
   ✓ "hepatocellular carcinoma" → tissues: ["liver", "hepatic"]
   ✗ NOT cell types (cardiomyocytes, hepatocytes, neurons)
   ✗ NOT populations (paediatric, women)

4. POPULATIONS — Patient demographics (age, sex, life stage)
   ✓ Age: "pediatric", "paediatric", "child", "children", "infant", "neonatal", "newborn", "adult", "elderly", "geriatric"
   ✓ Sex: "women", "men", "female", "male"
   ✓ Other: "pregnant", "maternal", "postmenopausal"
   ✗ These are NOT cell types — never put them in cell_types!
   ✗ These are NOT tissues — never put them in tissues!

5. CELL TYPES — Specific cell types used in experiments
   ✓ Examples: "iPSC-derived cortical neurons", "cortical neurons", "cardiomyocytes", "hepatocytes", "iPSCs", "stem cells", "fibroblasts", "macrophages", "T cells"
   ✗ NOT tissues (heart, liver, brain)
   ✗ NOT populations (paediatric, child, women, infant)

6. PROCESSES — Biological/molecular processes
   ✓ Examples: "transcriptional regulation", "apoptosis", "differentiation", "proliferation", "migration"

7. MODIFIERS — Disease/phenotype qualifiers
   ✓ Examples: "congenital", "acute", "chronic", "metastatic", "invasive", "de novo", "inherited"

8. CONTEXT GENES — Gene symbols mentioned (not the main gene being studied)
   ✓ All-caps symbols 2-10 characters: MYC, TP53, BRCA1

══════════════════════════════════════════════════════════════════════
NOISE TERMS — NEVER EXTRACT THESE
══════════════════════════════════════════════════════════════════════
Do NOT include: "genetic", "genetics", "genomic", "precision", "precision medicine",
"molecular", "variant", "variants", "mutation", "mutations", "novel", "new",
"analysis", "approach", "study", "research", "investigate", "interpret", "role",
"function", "mechanism", "patient", "case", "functionally validated"

══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════════════════════
Return ONLY valid JSON. If a category has no terms, return empty list [].

{{
  "diseases": [],
  "phenotypes": [],
  "tissues": [],
  "populations": [],
  "cell_types": [],
  "processes": [],
  "modifiers": [],
  "context_genes": []
}}

══════════════════════════════════════════════════════════════════════
EXAMPLES
══════════════════════════════════════════════════════════════════════

EXAMPLE 1:
Query: "In pediatric females with neurodevelopmental delay, a de novo MED12 variant was functionally validated in iPSC-derived cortical neurons, demonstrating altered transcriptional regulation in developing brain tissue."
Response:
{{
  "diseases": [],
  "phenotypes": ["neurodevelopmental delay", "intellectual disability", "developmental delay"],
  "tissues": ["brain", "cerebral", "cortical"],
  "populations": ["pediatric", "paediatric", "female", "children"],
  "cell_types": ["iPSC-derived cortical neurons", "cortical neurons"],
  "processes": ["transcriptional regulation"],
  "modifiers": ["de novo"],
  "context_genes": []
}}
Note: No named disease/syndrome, so diseases is empty.
Note: "neurodevelopmental delay" is a phenotype, NOT a disease.
Note: "brain" and "cortical" extracted as tissues from "cortical neurons" and "brain tissue".
Note: "pediatric" and "female" are populations.
Note: "iPSC-derived cortical neurons" is a cell type.

EXAMPLE 2:
Query: "A precision medicine approach to interpret a GATA4 genetic variant in a paediatric patient with congenital heart disease"
Response:
{{
  "diseases": ["congenital heart disease", "congenital heart defect", "CHD"],
  "phenotypes": [],
  "tissues": ["heart", "cardiac"],
  "populations": ["paediatric", "pediatric", "child", "children", "infant"],
  "cell_types": [],
  "processes": [],
  "modifiers": ["congenital"],
  "context_genes": []
}}
Note: "congenital heart disease" is a named disease, NOT a phenotype.
Note: "precision medicine", "genetic variant", "interpret", "patient" are noise — not extracted.
Note: "heart" extracted from "congenital heart disease".

EXAMPLE 3:
Query: "MED12 mutations cause X-linked intellectual disability with epilepsy and hypotonia"
Response:
{{
  "diseases": ["X-linked intellectual disability"],
  "phenotypes": ["intellectual disability", "epilepsy", "seizures", "hypotonia", "developmental delay"],
  "tissues": ["brain", "cerebral", "neural"],
  "populations": [],
  "cell_types": [],
  "processes": [],
  "modifiers": ["X-linked"],
  "context_genes": []
}}
Note: "X-linked intellectual disability" is a named disease/syndrome.
Note: "intellectual disability", "epilepsy", "hypotonia" are also phenotypes (observable symptoms).
Note: "brain" inferred from neurological phenotypes.

EXAMPLE 4:
Query: "BRCA1 expression in breast cancer stem cells from women"
Response:
{{
  "diseases": ["breast cancer", "breast carcinoma"],
  "phenotypes": [],
  "tissues": ["breast", "mammary"],
  "populations": ["women", "female"],
  "cell_types": ["stem cells", "cancer stem cells"],
  "processes": ["expression"],
  "modifiers": [],
  "context_genes": []
}}
Note: "women" is population, NOT cell type.
Note: "stem cells" is cell type.
Note: "breast" extracted from "breast cancer".

══════════════════════════════════════════════════════════════════════

Now extract terms from the research question above. Return ONLY the JSON."""

        return prompt

    def _fallback_extraction(
        self,
        user_query: str,
        experiment_context: Any = None
    ) -> Dict[str, List[str]]:
        """
        Fallback extraction using simple keyword matching.

        Used when LLM is unavailable or fails.
        """
        query_lower = user_query.lower()
        query_original = user_query  # Keep original for gene symbol extraction

        context = {
            'diseases': [],
            'phenotypes': [],
            'tissues': [],
            'populations': [],
            'processes': [],
            'cell_types': [],
            'modifiers': [],
            'context_genes': []
        }

        # Phenotype terms to extract (observable symptoms/presentations)
        phenotype_keywords = [
            'neurodevelopmental delay', 'developmental delay', 'intellectual disability',
            'cognitive impairment', 'learning disability', 'mental retardation',
            'autism spectrum disorder', 'autism', 'asd', 'adhd',
            'epilepsy', 'seizures', 'seizure', 'hypotonia', 'hypertonia',
            'ataxia', 'spasticity', 'dystonia',
            'microcephaly', 'macrocephaly', 'facial dysmorphism',
            'hearing loss', 'deafness', 'vision impairment', 'blindness',
            'feeding difficulties', 'failure to thrive',
            'cardiac abnormalities', 'heart defects',
        ]

        # Extract phenotypes
        for phenotype in phenotype_keywords:
            if phenotype in query_lower:
                if phenotype not in context['phenotypes']:
                    context['phenotypes'].append(phenotype)
                # Also add brain/neural tissues for neuro phenotypes
                neuro_phenotypes = ['neurodevelopmental', 'intellectual', 'cognitive',
                                    'epilepsy', 'seizure', 'ataxia', 'spasticity']
                if any(np in phenotype for np in neuro_phenotypes):
                    for tissue in ['brain', 'cerebral', 'neural']:
                        if tissue not in context['tissues']:
                            context['tissues'].append(tissue)

        # Tissue extraction map: disease prefix -> tissue terms
        tissue_map = {
            'heart': ['heart', 'cardiac'],
            'cardiac': ['heart', 'cardiac'],
            'breast': ['breast', 'mammary'],
            'mammary': ['breast', 'mammary'],
            'lung': ['lung', 'pulmonary'],
            'pulmonary': ['lung', 'pulmonary'],
            'liver': ['liver', 'hepatic'],
            'hepatic': ['liver', 'hepatic'],
            'hepato': ['liver', 'hepatic'],
            'brain': ['brain', 'cerebral'],
            'cerebral': ['brain', 'cerebral'],
            'colon': ['colon', 'colorectal'],
            'colorectal': ['colon', 'rectum', 'colorectal'],
            'rectal': ['rectum', 'colorectal'],
            'kidney': ['kidney', 'renal'],
            'renal': ['kidney', 'renal'],
            'pancreatic': ['pancreas', 'pancreatic'],
            'pancreas': ['pancreas', 'pancreatic'],
            'ovarian': ['ovary', 'ovarian'],
            'ovary': ['ovary', 'ovarian'],
            'prostate': ['prostate'],
            'gastric': ['stomach', 'gastric'],
            'stomach': ['stomach', 'gastric'],
            'skin': ['skin', 'cutaneous'],
            'thyroid': ['thyroid'],
            'bladder': ['bladder'],
        }

        # Population terms to extract
        population_keywords = [
            'paediatric', 'pediatric', 'child', 'children', 'infant', 'infants',
            'neonatal', 'neonate', 'newborn', 'adult', 'adults',
            'elderly', 'geriatric', 'aged',
            'women', 'woman', 'men', 'man', 'male', 'female',
            'pregnant', 'pregnancy', 'postmenopausal',
        ]

        # Extract diseases and tissues from disease patterns
        disease_patterns = [
            (r'(\w+(?:\s+\w+)?)\s+cancer', 'cancer'),
            (r'(\w+(?:\s+\w+)?)\s+carcinoma', 'carcinoma'),
            (r'(\w+(?:\s+\w+)?)\s+disease', 'disease'),
            (r'(\w+(?:\s+\w+)?)\s+tumor', 'tumor'),
            (r'(\w+(?:\s+\w+)?)\s+neoplasm', 'neoplasm'),
            (r'(\w+(?:\s+\w+)?)\s+syndrome', 'syndrome'),
        ]

        for pattern, suffix in disease_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                disease_term = f"{match.strip()} {suffix}"
                if disease_term not in context['diseases']:
                    context['diseases'].append(disease_term)

                # Extract tissue from disease prefix
                for word in match.strip().split():
                    word_clean = word.strip()
                    if word_clean in tissue_map:
                        for tissue in tissue_map[word_clean]:
                            if tissue not in context['tissues']:
                                context['tissues'].append(tissue)

        # Extract population terms
        for pop_term in population_keywords:
            if pop_term in query_lower:
                if pop_term not in context['populations']:
                    context['populations'].append(pop_term)

        # Extract common biological processes
        process_keywords = [
            'apoptosis', 'cell death', 'proliferation', 'migration', 'invasion',
            'metastasis', 'differentiation', 'development', 'signaling',
            'regulation', 'expression', 'activation', 'inhibition'
        ]
        for keyword in process_keywords:
            if keyword in query_lower:
                context['processes'].append(keyword)

        # Extract modifiers
        modifiers = ['invasive', 'metastatic', 'acute', 'chronic', 'aggressive', 'advanced', 'congenital']
        for modifier in modifiers:
            if modifier in query_lower:
                context['modifiers'].append(modifier)

        # Extract context genes (uppercase words 2-10 chars that look like gene symbols)
        gene_pattern = r'\b([A-Z][A-Z0-9]{1,9})\b'
        potential_genes = re.findall(gene_pattern, query_original)
        non_genes = {'THE', 'AND', 'FOR', 'WITH', 'NOT', 'ARE', 'WAS', 'HAS', 'HAD',
                     'THIS', 'THAT', 'FROM', 'INTO', 'ROLE', 'HOW', 'WHY', 'WHAT',
                     'MHC', 'DNA', 'RNA', 'ATP', 'ADP', 'GTP', 'UDP', 'GENE', 'GENES'}
        for gene in potential_genes:
            if gene not in non_genes and gene not in context['context_genes']:
                context['context_genes'].append(gene)

        # Add from experiment context
        if experiment_context:
            if hasattr(experiment_context, 'cell_type') and experiment_context.cell_type:
                cell_type = str(experiment_context.cell_type).strip()
                if cell_type:
                    context['cell_types'].append(cell_type)

            if hasattr(experiment_context, 'tissue') and experiment_context.tissue:
                tissue = str(experiment_context.tissue).strip()
                if tissue and tissue not in context['tissues']:
                    context['tissues'].append(tissue)

        return context

    def _log_extraction(self, context_terms: Dict[str, List[str]]) -> None:
        """Log extracted terms for user visibility."""

        diseases = context_terms.get('diseases', [])
        phenotypes = context_terms.get('phenotypes', [])
        tissues = context_terms.get('tissues', [])
        populations = context_terms.get('populations', [])
        processes = context_terms.get('processes', [])
        cell_types = context_terms.get('cell_types', [])
        modifiers = context_terms.get('modifiers', [])
        context_genes = context_terms.get('context_genes', [])

        print(f"    Diseases: {diseases if diseases else '(none)'}")
        print(f"    Phenotypes: {phenotypes if phenotypes else '(none)'}")
        print(f"    Tissues: {tissues if tissues else '(none)'}")
        print(f"    Populations: {populations if populations else '(none)'}")
        print(f"    Processes: {processes if processes else '(none)'}")
        print(f"    Cell types: {cell_types if cell_types else '(none)'}")
        print(f"    Modifiers: {modifiers if modifiers else '(none)'}")
        print(f"    Context genes: {context_genes if context_genes else '(none)'}")


def build_pubmed_query_from_context(
    gene_symbol: str,
    context_terms: Dict[str, List[str]],
    require_all_terms: bool = False
) -> str:
    """
    Build PubMed query from extracted context terms.

    Args:
        gene_symbol: Gene symbol (e.g., "BRCA1")
        context_terms: Context extracted by LLMContextExtractor
        require_all_terms: If True, requires ALL term categories (strict).
                          If False, uses OR logic (more inclusive).

    Returns:
        PubMed query string
    """

    query_parts = [f'{gene_symbol}[Title/Abstract]']

    # Add cell types
    cell_types = context_terms.get('cell_types', [])
    if cell_types:
        cell_query = " OR ".join([f'"{ct}"[Title/Abstract]' for ct in cell_types])
        query_parts.append(f'({cell_query})')

    # Add diseases (with synonyms)
    diseases = context_terms.get('diseases', [])
    if diseases:
        disease_query = " OR ".join([f'"{d}"[Title/Abstract]' for d in diseases])
        query_parts.append(f'({disease_query})')

    # Add biological processes
    processes = context_terms.get('processes', [])
    if processes:
        process_query = " OR ".join([f'"{p}"[Title/Abstract]' for p in processes])
        query_parts.append(f'({process_query})')

    # Build final query
    if require_all_terms:
        # Strict: gene AND cell_type AND disease AND process
        return " AND ".join(query_parts)
    else:
        # Moderate: gene AND (cell_type OR disease OR process)
        if len(query_parts) > 1:
            context_clause = " OR ".join(query_parts[1:])
            return f"{query_parts[0]} AND ({context_clause})"
        else:
            return query_parts[0]


def build_disease_filter_from_context(context_terms: Dict[str, List[str]]) -> str:
    """
    Build disease filter clause for PubMed queries.

    Args:
        context_terms: Context extracted by LLMContextExtractor

    Returns:
        Disease filter string for PubMed (e.g., "breast cancer"[tiab] OR "breast carcinoma"[tiab])
    """

    diseases = context_terms.get('diseases', [])
    if not diseases:
        return ""

    # Build OR clause with all disease terms
    disease_parts = []
    for disease in diseases:
        if " " in disease:
            disease_parts.append(f'"{disease}"[Title/Abstract]')
        else:
            disease_parts.append(f'{disease}[Title/Abstract]')

    return " OR ".join(disease_parts)
