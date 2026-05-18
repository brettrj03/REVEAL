"""
Data models for the gene annotation system
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ExperimentContext(BaseModel):
    """Context information about the experiment"""
    organism: str = "human"
    cell_type: Optional[str] = None
    treatment: Optional[str] = None
    timepoint: Optional[str] = None
    comparison: str  # Required - describes the experimental comparison
    hypothesis: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "organism": "human",
                "cell_type": "HeLa cells",
                "treatment": "Drug X",
                "timepoint": "24 hours",
                "comparison": "Treated vs Control",
                "hypothesis": "Drug X increases TP53 activity"
            }
        }

# Add this new model to your models.py file, after the existing models

class GeneNameMapping(BaseModel):
    """Tracks gene name normalization, duplicates, and canonical resolution details"""

    user_genes: List[str] = Field(
        default_factory=list,
        description="Raw gene names extracted from the user query (before cleanup)"
    )

    normalized_genes: List[str] = Field(
        default_factory=list,
        description="Normalized gene names aligned to our matching rules (includes duplicates)"
    )

    unique_genes: List[str] = Field(
        default_factory=list,
        description="Order-preserving list of unique normalized genes used for retrieval"
    )

    normalization_records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-gene breakdown of cleanup steps, transformations, and suggestions"
    )

    duplicates: Dict[str, int] = Field(
        default_factory=dict,
        description="Map of genes that appeared multiple times: {'MYL2': 2}"
    )

    transformations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of applied corrections (e.g., prefix stripping, alias resolution)"
    )

    suggestions: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Potential fixes for unresolved genes (e.g., {'BNP': {'suggested': 'NPPB'}})"
    )

    user_to_canonical: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps user-provided names to canonical database names"
    )

    canonical_to_user: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Reverse mapping to display which user inputs yielded a canonical gene"
    )

    resolution_status: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-normalized-gene resolution details (found, match type, official symbol)"
    )

    notes: List[str] = Field(
        default_factory=list,
        description="Miscellaneous notes about normalization/resolution"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_genes": ["ANP", "NPPA", "MYL2", "MYL2"],
                "normalized_genes": ["ANP", "NPPA", "MYL2", "MYL2"],
                "unique_genes": ["NPPA", "MYL2"],
                "normalization_records": [
                    {"original": "ANP", "normalized": "NPPA", "reasons": ["alias"]}
                ],
                "duplicates": {"MYL2": 2},
                "transformations": [
                    {"original": "ANP", "normalized": "NPPA", "reasons": ["alias"]}
                ],
                "suggestions": {"BNP": {"suggested": "NPPB", "reason": "alias"}},
                "user_to_canonical": {"ANP": "NPPA", "NPPA": "NPPA"},
                "canonical_to_user": {"NPPA": ["ANP", "NPPA"]},
                "resolution_status": {
                    "NPPA": {"found": True, "official_symbol": "NPPA", "direct_input": True}
                },
                "notes": ["MYL2 appeared twice"]
            }
        }

class GeneInfo(BaseModel):
    """Basic gene information"""
    gene_id: str
    symbol: str
    gene_type: str
    chrom: str
    start: int
    end: int
    strand: str
    source: Optional[str] = None


class GeneFunction(BaseModel):
    """Gene function description"""
    description: Optional[str] = None
    evidence_tier: Optional[str] = None
    source: Optional[str] = None
    updated_at: Optional[datetime] = None


class GOTerm(BaseModel):
    """Gene Ontology term"""
    go_id: str
    name: str
    namespace: str
    aspect: str
    evidence: Optional[str] = None
    ref: Optional[str] = None


class Expression(BaseModel):
    """Expression data"""
    tissue: str
    level: float
    units: Optional[str] = None
    dataset: Optional[str] = None


class ProteinInteraction(BaseModel):
    """Protein-protein interaction"""
    partner_symbol: str
    interaction_type: Optional[str] = None
    combined_score: Optional[int] = None
    confidence: Optional[str] = None
    pubmed_id: Optional[str] = None
    source: Optional[str] = None
    evidence_code: Optional[str] = None


class GeneProfile(BaseModel):
    """Complete gene profile"""
    gene_id: str
    core: Dict[str, Any] = Field(default_factory=dict)
    function: Dict[str, Any] = Field(default_factory=dict)
    aliases: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    go_terms: List[Dict[str, Any]] = Field(default_factory=list)
    expression: List[Dict[str, Any]] = Field(default_factory=list)
    interactions: List[Dict[str, Any]] = Field(default_factory=list)


class ProteinNetworkSummary(BaseModel):
    """Summary of protein interaction network analysis"""
    total_genes: int
    protein_coding_with_data: int
    direct_interactions_found: int
    genes_analyzed: int


class ProteinNetwork(BaseModel):
    """Protein interaction network data"""
    protein_status: Dict[str, Any]
    proteins_with_data: List[str]
    direct_interactions: List[Dict[str, Any]]
    top_interactions: Dict[str, List[Dict[str, Any]]]
    summary: ProteinNetworkSummary


class GeneDataCollection(BaseModel):
    """Complete collection of gene data"""
    experiment_context: ExperimentContext
    gene_list: List[str]
    mapping: Dict[str, Dict[str, Any]]
    protein_network: Optional[Dict[str, Any]] = None
    profiles: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True


class DataProvenance(BaseModel):
    """Tracks the source and origin of each piece of data"""
    field_name: str = Field(description="Name of the data field (e.g., 'chromosome', 'description')")
    value: Any = Field(description="The actual value")
    database_path: str = Field(description="Path to the database file")
    table_name: str = Field(description="Table where data was retrieved")
    column_name: Optional[str] = Field(default=None, description="Specific column in table")
    source_release: Optional[str] = Field(default=None, description="e.g., 'GENCODE v49 (Ensembl 115)'")
    record_id: Optional[str] = Field(default=None, description="Primary key of the record")
    retrieved_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "field_name": "chromosome",
                "value": "chr5",
                "database_path": "src/database/gene_database.sqlite",
                "table_name": "genes",
                "column_name": "chrom",
                "source_release": "GENCODE v49 (Ensembl 115)",
                "record_id": "ENSG00000141510",
                "retrieved_at": "2025-01-20T10:30:00"
            }
        }


class GeneInterpretation(BaseModel):
    """Results from gene data interpretation"""
    biological_processes: List[str]
    molecular_functions: List[str]
    cellular_components: List[str]
    protein_interaction_summary: str
    context_specific_insights: str
    key_genes: List[Dict[str, str]]
    suggested_followup: List[str]

class InterpretationRequest(BaseModel):
    """Request for gene data interpretation"""
    data_file: str
    focus: Optional[str] = None
    output_file: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "data_file": "results/fetcher/gene_data.json",
                "focus": "protein interactions",
                "output_file": "results/interpretations/analysis.txt"
            }
        }
