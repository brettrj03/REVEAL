"""
Data Provenance Tracking - Logs the exact source of all information

This module provides utilities to track where each piece of data comes from,
including database, table, column, and source release information.
"""

import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime


TABLE_SOURCE_MAP = {
    'genes': 'GENCODE_GTF',
    'gene_function': 'BioMart_Ensembl',
    'gene_alias': 'BioMart_Ensembl',
    'gene_go': 'GO_GAF',
    'go_terms': 'GO_OBO',
    'expression': 'GTEx',
    'string_interactions': 'STRING',
    'string_proteins': 'STRING',
    'string_aliases': 'STRING'
}


class ProvenanceTracker:
    """Tracks data provenance for gene information"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._metadata_cache = None

    def get_database_metadata(self) -> Dict[str, str]:
        """Get database source metadata (GENCODE version, etc.)"""
        if self._metadata_cache:
            return self._metadata_cache

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        metadata = {}
        try:
            # Get metadata from metadata table
            cursor.execute("SELECT source, release FROM metadata")
            for row in cursor.fetchall():
                metadata[row[0]] = row[1]
        except sqlite3.OperationalError:
            # Table might not exist
            pass

        conn.close()
        self._metadata_cache = metadata
        return metadata

    def track_gene_field(self, gene_id: str, field_name: str, value: Any,
                         table_name: str, column_name: str) -> Dict[str, Any]:
        """
        Create a provenance record for a gene field

        Args:
            gene_id: Ensembl gene ID
            field_name: Descriptive name (e.g., 'chromosome', 'gene_type')
            value: The actual value
            table_name: Database table name
            column_name: Column in the table

        Returns:
            Provenance record dictionary
        """
        metadata = self.get_database_metadata()

        source_release = self._resolve_source_release(metadata, table_name)

        return {
            'field_name': field_name,
            'value': value,
            'database_path': self.db_path,
            'table_name': table_name,
            'column_name': column_name,
            'source_release': source_release,
            'record_id': gene_id,
            'retrieved_at': datetime.now().isoformat()
        }

    def _resolve_source_release(self, metadata: Dict[str, str], table_name: str) -> str:
        """Resolve metadata release string for a given table."""
        source_key = TABLE_SOURCE_MAP.get(table_name, table_name)
        fallback = metadata.get('GENCODE_GTF') or metadata.get('GENCODE') or 'Unknown'
        return metadata.get(source_key, fallback)

    def track_go_term(self, gene_id: str, go_term: Dict[str, Any]) -> Dict[str, Any]:
        """Track provenance for a GO term annotation"""
        metadata = self.get_database_metadata()

        return {
            'field_name': f"GO term: {go_term.get('name', 'Unknown')}",
            'value': go_term.get('go_id'),  # Store just the GO ID, not the entire object
            'database_path': self.db_path,
            'table_name': 'gene_go',
            'column_name': 'go_id',
            'source_release': self._resolve_source_release(metadata, 'gene_go'),
            'record_id': gene_id,
            'go_id': go_term.get('go_id'),
            'evidence': go_term.get('evidence'),
            'reference': go_term.get('ref'),
            'retrieved_at': datetime.now().isoformat()
        }

    def track_interaction(self, gene_id: str, interaction: Dict[str, Any],
                         source: str = 'STRING') -> Dict[str, Any]:
        """Track provenance for a protein interaction"""
        metadata = self.get_database_metadata()

        table_name = 'string_interactions' if source == 'STRING' else 'gene_interactions'
        score_value = interaction.get('score', interaction.get('combined_score'))

        return {
            'field_name': f"Protein interaction: {interaction.get('partner_symbol', 'Unknown')}",
            'value': interaction.get('partner_symbol'),  # Store just the partner symbol, not the entire object
            'database_path': self.db_path,
            'table_name': table_name,
            'column_name': 'partner_symbol',
            'source_release': self._resolve_source_release(metadata, table_name if source == 'STRING' else source),
            'record_id': gene_id,
            'interaction_type': interaction.get('interaction_type'),
            'evidence': interaction.get('evidence_code'),
            'combined_score': score_value,
            'retrieved_at': datetime.now().isoformat()
        }

    def generate_provenance_report(self, gene_profiles: List[Dict[str, Any]],
                                   output_file: Optional[str] = None) -> str:
        """
        Generate a human-readable provenance report

        Args:
            gene_profiles: List of gene profile dictionaries with provenance data
            output_file: Optional path to save report

        Returns:
            Report text
        """
        metadata = self.get_database_metadata()

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA PROVENANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Database: {self.db_path}")
        report_lines.append("")

        # Database sources
        report_lines.append("DATABASE SOURCES:")
        report_lines.append("-" * 80)
        for source, release in metadata.items():
            report_lines.append(f"  • {source}: {release}")
        report_lines.append("")

        # Per-gene provenance
        for profile in gene_profiles:
            gene_symbol = profile.get('symbol', 'Unknown')
            gene_id = profile.get('gene_id', 'Unknown')

            report_lines.append(f"GENE: {gene_symbol} ({gene_id})")
            report_lines.append("-" * 80)

            # Core gene information
            if 'provenance' in profile:
                for prov in profile['provenance']:
                    field = prov['field_name']
                    value = prov['value']
                    table = prov['table_name']
                    column = prov['column_name']
                    source = prov.get('source_release', 'Unknown')

                    report_lines.append(f"  {field}: {value}")
                    report_lines.append(f"    └─ Source: {table}.{column} ({source})")

            # GO Terms
            if profile.get('go_terms'):
                report_lines.append(f"\n  GO TERMS ({len(profile['go_terms'])} total):")
                for i, go_term in enumerate(profile['go_terms'][:3], 1):  # Show first 3
                    report_lines.append(f"    {i}. {go_term.get('name', 'Unknown')} ({go_term.get('go_id')})")
                    report_lines.append(f"       └─ Source: gene_go table | Evidence: {go_term.get('evidence', 'N/A')}")
                if len(profile['go_terms']) > 3:
                    report_lines.append(f"    ... and {len(profile['go_terms']) - 3} more")

            # Protein Interactions
            if profile.get('interactions'):
                report_lines.append(f"\n  PROTEIN INTERACTIONS ({len(profile['interactions'])} total):")
                for i, interaction in enumerate(profile['interactions'][:3], 1):
                    partner = interaction.get('partner_symbol', 'Unknown')
                    score = interaction.get('score', interaction.get('combined_score', 'N/A'))
                    report_lines.append(f"    {i}. {partner} (score: {score})")
                    report_lines.append(f"       └─ Source: string_interactions table")
                if len(profile['interactions']) > 3:
                    report_lines.append(f"    ... and {len(profile['interactions']) - 3} more")

            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF PROVENANCE REPORT")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # Save if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"📄 Provenance report saved: {output_file}")

        return report_text


def create_field_provenance(gene_id: str, field_name: str, value: Any,
                            table: str, column: str, source_release: str = None,
                            db_path: str = "src/database/gene_database.sqlite") -> Dict[str, Any]:
    """
    Quick helper to create a single provenance record

    Example:
        prov = create_field_provenance(
            gene_id="ENSG00000141510",
            field_name="chromosome",
            value="chr5",
            table="genes",
            column="chromosome",
            source_release="GENCODE v49 (Ensembl 115)"
        )
    """
    return {
        'field_name': field_name,
        'value': value,
        'database_path': db_path,
        'table_name': table,
        'column_name': column,
        'source_release': source_release,
        'record_id': gene_id,
        'retrieved_at': datetime.now().isoformat()
    }
