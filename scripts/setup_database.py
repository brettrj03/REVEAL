#!/usr/bin/env python3
"""
Database Setup Script for REVEAL

Downloads data from public sources and builds the gene annotation database from scratch.

Usage:
    python scripts/setup_database.py                    # Full setup
    python scripts/setup_database.py --skip-download    # Use existing data files
    python scripts/setup_database.py --data-dir /path   # Custom data directory
    python scripts/setup_database.py --output /path.db  # Custom output path

Data Sources:
    - GENCODE: Gene annotations (v49)
    - GO: Gene Ontology terms and associations
    - GTEx: Tissue expression data (v8)
    - STRING: Protein-protein interactions (v12.0)
    - NCBI: Gene descriptions and aliases

Author: REVEAL Team
Date: 2025-01
"""

import argparse
import gzip
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "src" / "database" / "gene_database.sqlite"
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "src" / "database" / "data"

# Data source URLs and expected file info
DATA_SOURCES = {
    "gencode": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz",
        "filename": "gencode.v49.annotation.gtf.gz",
        "description": "GENCODE v49 gene annotations",
        "release": "v49",
        "label": "GENCODE",
    },
    "go_obo": {
        "url": "http://current.geneontology.org/ontology/go-basic.obo",
        "fallback_urls": [
            "http://purl.obolibrary.org/obo/go/go-basic.obo",
            "https://release.geneontology.org/2024-01-17/ontology/go-basic.obo",
        ],
        "filename": "go-basic.obo",
        "description": "Gene Ontology terms",
        "release": "2025-01",
        "manual_url": "https://geneontology.org/docs/download-ontology/",
        "label": "GO",
    },
    "go_gaf": {
        "url": "http://current.geneontology.org/annotations/goa_human.gaf.gz",
        "fallback_urls": [
            "http://geneontology.org/gene-associations/goa_human.gaf.gz",
            "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz",
        ],
        "filename": "goa_human.gaf.gz",
        "description": "Human GO annotations",
        "release": "2025-01",
        "manual_url": "https://geneontology.org/docs/download-go-annotations/",
        "label": "GO",
    },
    "gtex": {
        "url": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz",
        "filename": "GTEx_gene_median_tpm.gct.gz",
        "description": "GTEx v8 tissue expression",
        "release": "v8",
        "label": "GTEx",
    },
    "string_proteins": {
        "url": "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
        "filename": "9606.protein.info.v12.0.txt.gz",
        "description": "STRING protein info",
        "release": "v12.0",
        "label": "STRING",
    },
    "string_interactions": {
        "url": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        "filename": "9606.protein.links.v12.0.txt.gz",
        "description": "STRING protein interactions",
        "release": "v12.0",
        "label": "STRING",
    },
    "string_aliases": {
        "url": "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
        "filename": "9606.protein.aliases.v12.0.txt.gz",
        "description": "STRING protein aliases",
        "release": "v12.0",
        "label": "STRING",
    },
    "ncbi_gene_info": {
        "url": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
        "filename": "Homo_sapiens.gene_info.gz",
        "description": "NCBI gene info (descriptions, aliases)",
        "release": "2025-01",
        "label": "NCBI",
    },
}


# ============================================================================
# Utility Functions
# ============================================================================

def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(text: str, status: str = "info") -> None:
    """Print a step with status indicator."""
    symbols = {"info": "→", "ok": "✓", "warn": "⚠", "error": "✗", "skip": "○"}
    print(f"  {symbols.get(status, '→')} {text}")


def file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file(url: str, dest: Path, description: str, fallback_urls: List[str] = None, manual_url: str = None) -> bool:
    """Download a file with progress reporting, retry logic, and fallback URLs.

    Args:
        url: Primary download URL
        dest: Destination file path
        description: Human-readable description for logging
        fallback_urls: List of alternative URLs to try if primary fails
        manual_url: URL to show user for manual download instructions

    Returns:
        True if download succeeded, False otherwise
    """
    import time

    # HTTP headers to avoid 403 errors (mimic a browser)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    # Build list of URLs to try (primary + fallbacks)
    urls_to_try = [url]
    if fallback_urls:
        urls_to_try.extend(fallback_urls)

    max_retries = 3
    retry_delay = 2  # seconds

    print_step(f"Downloading {description}...")

    for url_idx, current_url in enumerate(urls_to_try):
        if url_idx > 0:
            print_step(f"Trying fallback URL ({url_idx}/{len(urls_to_try)-1})...", "warn")

        print(f"      URL: {current_url}")
        print(f"      Destination: {dest}")

        for attempt in range(1, max_retries + 1):
            try:
                # Create request with headers
                request = urllib.request.Request(current_url, headers=headers)

                # Open URL and download with progress
                with urllib.request.urlopen(request, timeout=60) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                    block_size = 8192
                    downloaded = 0

                    with open(dest, 'wb') as f:
                        while True:
                            chunk = response.read(block_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total_size > 0:
                                pct = min(100, downloaded * 100 // total_size)
                                mb_done = downloaded / 1e6
                                mb_total = total_size / 1e6
                                print(f"\r      Progress: {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)
                            else:
                                mb_done = downloaded / 1e6
                                print(f"\r      Downloaded: {mb_done:.1f} MB", end="", flush=True)

                print()  # newline after progress
                print_step(f"Downloaded: {dest.stat().st_size / 1e6:.1f} MB", "ok")
                return True

            except urllib.error.HTTPError as e:
                print()
                print_step(f"HTTP Error {e.code}: {e.reason} (attempt {attempt}/{max_retries})", "error")
                if e.code == 403:
                    print_step("Server rejected request (403 Forbidden)", "error")
                if attempt < max_retries:
                    print_step(f"Retrying in {retry_delay}s...", "info")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff

            except urllib.error.URLError as e:
                print()
                print_step(f"URL Error: {e.reason} (attempt {attempt}/{max_retries})", "error")
                if attempt < max_retries:
                    print_step(f"Retrying in {retry_delay}s...", "info")
                    time.sleep(retry_delay)
                    retry_delay *= 2

            except Exception as e:
                print()
                print_step(f"Download failed: {e} (attempt {attempt}/{max_retries})", "error")
                if attempt < max_retries:
                    print_step(f"Retrying in {retry_delay}s...", "info")
                    time.sleep(retry_delay)
                    retry_delay *= 2

        # Reset retry delay for next URL
        retry_delay = 2

    # All URLs failed - show manual download instructions
    print()
    print_step("All download attempts failed!", "error")
    print()
    print("      ┌─────────────────────────────────────────────────────────────┐")
    print("      │  MANUAL DOWNLOAD INSTRUCTIONS                               │")
    print("      ├─────────────────────────────────────────────────────────────┤")
    print(f"      │  File needed: {description}")
    print(f"      │  Save as: {dest}")
    if manual_url:
        print(f"      │  Download page: {manual_url}")
    else:
        print(f"      │  Try URL: {url}")
    print("      │                                                             │")
    print("      │  After manual download, re-run with --skip-download         │")
    print("      └─────────────────────────────────────────────────────────────┘")
    print()

    return False


# ============================================================================
# Schema Creation
# ============================================================================

SCHEMA_SQL = """
-- Core gene table
CREATE TABLE IF NOT EXISTS genes (
    gene_id TEXT PRIMARY KEY,
    gene_symbol TEXT NOT NULL,
    gene_type TEXT,
    chromosome TEXT,
    start INTEGER,
    end INTEGER,
    strand TEXT,
    data_source TEXT
);

-- Gene descriptions
CREATE TABLE IF NOT EXISTS gene_function (
    gene_id TEXT PRIMARY KEY,
    description TEXT,
    data_source TEXT,
    FOREIGN KEY (gene_id) REFERENCES genes(gene_id)
);

-- Gene aliases
CREATE TABLE IF NOT EXISTS gene_alias (
    symbol_alias TEXT NOT NULL,
    gene_id TEXT NOT NULL,
    data_source TEXT,
    PRIMARY KEY (symbol_alias, gene_id),
    FOREIGN KEY (gene_id) REFERENCES genes(gene_id)
);

-- GO terms
CREATE TABLE IF NOT EXISTS go_terms (
    go_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    namespace TEXT,
    definition TEXT,
    is_obsolete INTEGER DEFAULT 0,
    data_source TEXT
);

-- GO hierarchy edges
CREATE TABLE IF NOT EXISTS go_edges (
    child_go_id TEXT NOT NULL,
    parent_go_id TEXT NOT NULL,
    relation_type TEXT DEFAULT 'is_a',
    PRIMARY KEY (child_go_id, parent_go_id, relation_type),
    FOREIGN KEY (child_go_id) REFERENCES go_terms(go_id),
    FOREIGN KEY (parent_go_id) REFERENCES go_terms(go_id)
);

-- Gene-GO associations
CREATE TABLE IF NOT EXISTS gene_go (
    gene_id TEXT NOT NULL,
    go_id TEXT NOT NULL,
    evidence TEXT,
    go_namespace TEXT,
    qualifier TEXT,
    data_source TEXT,
    PRIMARY KEY (gene_id, go_id, evidence),
    FOREIGN KEY (gene_id) REFERENCES genes(gene_id),
    FOREIGN KEY (go_id) REFERENCES go_terms(go_id)
);

-- Expression data
CREATE TABLE IF NOT EXISTS expression (
    gene_id TEXT NOT NULL,
    tissue TEXT NOT NULL,
    tpm_value REAL NOT NULL,
    unit TEXT DEFAULT 'TPM',
    data_source TEXT,
    PRIMARY KEY (gene_id, tissue),
    FOREIGN KEY (gene_id) REFERENCES genes(gene_id)
);

-- STRING proteins
CREATE TABLE IF NOT EXISTS string_proteins (
    protein_id TEXT PRIMARY KEY,
    symbol TEXT,
    gene_id TEXT,
    annotation TEXT,
    data_source TEXT,
    FOREIGN KEY (gene_id) REFERENCES genes(gene_id)
);

-- STRING interactions
CREATE TABLE IF NOT EXISTS string_interactions (
    protein_id_1 TEXT NOT NULL,
    protein_id_2 TEXT NOT NULL,
    combined_score INTEGER NOT NULL,
    data_source TEXT,
    PRIMARY KEY (protein_id_1, protein_id_2),
    FOREIGN KEY (protein_id_1) REFERENCES string_proteins(protein_id),
    FOREIGN KEY (protein_id_2) REFERENCES string_proteins(protein_id)
);

-- STRING aliases
CREATE TABLE IF NOT EXISTS string_aliases (
    symbol_alias TEXT NOT NULL,
    protein_id TEXT NOT NULL,
    alias_source TEXT,
    data_source TEXT,
    PRIMARY KEY (symbol_alias, protein_id, alias_source),
    FOREIGN KEY (protein_id) REFERENCES string_proteins(protein_id)
);

-- Metadata
CREATE TABLE IF NOT EXISTS metadata (
    source TEXT PRIMARY KEY,
    release TEXT,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    download_url TEXT,
    file_hash TEXT,
    record_count INTEGER,
    notes TEXT
);
"""

INDEX_SQL = """
-- Core gene indices
CREATE INDEX IF NOT EXISTS idx_genes_symbol ON genes(gene_symbol);
CREATE INDEX IF NOT EXISTS idx_genes_chrom ON genes(chromosome, start);
CREATE INDEX IF NOT EXISTS idx_genes_type ON genes(gene_type);

-- Gene alias indices
CREATE INDEX IF NOT EXISTS idx_gene_alias_symbol ON gene_alias(symbol_alias);
CREATE INDEX IF NOT EXISTS idx_gene_alias_gene_id ON gene_alias(gene_id);

-- GO indices
CREATE INDEX IF NOT EXISTS idx_go_terms_namespace ON go_terms(namespace);
CREATE INDEX IF NOT EXISTS idx_go_edges_parent ON go_edges(parent_go_id);
CREATE INDEX IF NOT EXISTS idx_go_edges_child ON go_edges(child_go_id);

-- Gene-GO indices
CREATE INDEX IF NOT EXISTS idx_gene_go_gene_id ON gene_go(gene_id);
CREATE INDEX IF NOT EXISTS idx_gene_go_go_id ON gene_go(go_id);
CREATE INDEX IF NOT EXISTS idx_gene_go_namespace ON gene_go(go_namespace);

-- Expression indices
CREATE INDEX IF NOT EXISTS idx_expression_gene_id ON expression(gene_id);
CREATE INDEX IF NOT EXISTS idx_expression_tissue ON expression(tissue);
CREATE INDEX IF NOT EXISTS idx_expression_composite ON expression(gene_id, tissue);

-- STRING indices
CREATE INDEX IF NOT EXISTS idx_string_proteins_symbol ON string_proteins(symbol);
CREATE INDEX IF NOT EXISTS idx_string_proteins_gene_id ON string_proteins(gene_id);
CREATE INDEX IF NOT EXISTS idx_string_interactions_p1 ON string_interactions(protein_id_1);
CREATE INDEX IF NOT EXISTS idx_string_interactions_p2 ON string_interactions(protein_id_2);
CREATE INDEX IF NOT EXISTS idx_string_interactions_score ON string_interactions(combined_score);
CREATE INDEX IF NOT EXISTS idx_string_aliases_symbol ON string_aliases(symbol_alias);
CREATE INDEX IF NOT EXISTS idx_string_aliases_protein ON string_aliases(protein_id);
"""

VIEW_SQL = """
-- Comprehensive gene summary view
CREATE VIEW IF NOT EXISTS v_gene_summary AS
SELECT
    g.gene_id,
    g.gene_symbol,
    g.gene_type,
    g.chromosome,
    g.start,
    g.end,
    gf.description,
    (SELECT COUNT(*) FROM gene_go gg WHERE gg.gene_id = g.gene_id) as go_count,
    (SELECT COUNT(DISTINCT tissue) FROM expression e WHERE e.gene_id = g.gene_id) as tissue_count,
    (SELECT AVG(tpm_value) FROM expression e WHERE e.gene_id = g.gene_id) as avg_tpm
FROM genes g
LEFT JOIN gene_function gf ON g.gene_id = gf.gene_id;

-- Protein interactions with gene symbols
CREATE VIEW IF NOT EXISTS v_protein_interactions_with_symbols AS
SELECT
    si.protein_id_1,
    sp1.symbol as symbol_1,
    sp1.gene_id as gene_id_1,
    si.protein_id_2,
    sp2.symbol as symbol_2,
    sp2.gene_id as gene_id_2,
    si.combined_score
FROM string_interactions si
JOIN string_proteins sp1 ON si.protein_id_1 = sp1.protein_id
JOIN string_proteins sp2 ON si.protein_id_2 = sp2.protein_id
WHERE si.combined_score >= 400;
"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create database schema."""
    print_step("Creating tables...")
    conn.executescript(SCHEMA_SQL)
    print_step("Tables created", "ok")

    print_step("Creating indices...")
    conn.executescript(INDEX_SQL)
    print_step("Indices created", "ok")

    print_step("Creating views...")
    conn.executescript(VIEW_SQL)
    print_step("Views created", "ok")

    conn.commit()


# ============================================================================
# Data Loaders
# ============================================================================

def load_gencode(conn: sqlite3.Connection, data_dir: Path) -> int:
    """Load GENCODE gene annotations."""
    config = DATA_SOURCES["gencode"]
    filepath = data_dir / config["filename"]
    data_source = config["label"]

    if not filepath.exists():
        print_step(f"GENCODE file not found: {filepath}", "error")
        return 0

    print_step("Loading GENCODE genes...")

    genes = []
    open_func = gzip.open if filepath.suffix == '.gz' else open

    with open_func(filepath, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue

            # Parse attributes
            attrs = {}
            for attr in fields[8].split(';'):
                attr = attr.strip()
                if ' ' in attr:
                    key, val = attr.split(' ', 1)
                    attrs[key] = val.strip('"')

            gene_id = attrs.get('gene_id', '').split('.')[0]  # Remove version
            if not gene_id:
                continue

            genes.append((
                gene_id,
                attrs.get('gene_name', ''),
                attrs.get('gene_type', ''),
                fields[0],  # chrom
                int(fields[3]),  # start
                int(fields[4]),  # end
                fields[6],  # strand
                data_source
            ))

    conn.executemany(
        "INSERT OR REPLACE INTO genes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        genes
    )
    conn.commit()

    # Update metadata
    conn.execute("""
        INSERT OR REPLACE INTO metadata (source, release, download_url, file_hash, record_count, loaded_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (data_source, config["release"], config["url"], file_hash(filepath), len(genes)))
    conn.commit()

    print_step(f"Loaded {len(genes):,} genes", "ok")
    return len(genes)


def load_ncbi_gene_info(conn: sqlite3.Connection, data_dir: Path) -> Tuple[int, int]:
    """Load NCBI gene info (descriptions and aliases)."""
    config = DATA_SOURCES["ncbi_gene_info"]
    filepath = data_dir / config["filename"]
    data_source = config["label"]

    if not filepath.exists():
        print_step(f"NCBI gene info not found: {filepath}", "error")
        return 0, 0

    print_step("Loading NCBI gene descriptions and aliases...")

    # Get existing gene symbols for mapping
    cursor = conn.execute("SELECT gene_symbol, gene_id FROM genes")
    symbol_to_geneid = {row[0]: row[1] for row in cursor.fetchall()}

    descriptions = []
    aliases = []

    open_func = gzip.open if filepath.suffix == '.gz' else open

    with open_func(filepath, 'rt') as f:
        header = f.readline()  # Skip header
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 15:
                continue

            symbol = fields[2]
            if symbol not in symbol_to_geneid:
                continue

            gene_id = symbol_to_geneid[symbol]
            description = fields[8] if fields[8] != '-' else None

            if description:
                descriptions.append((gene_id, description, data_source))

            # Parse aliases (field 4)
            if fields[4] != '-':
                for alias in fields[4].split('|'):
                    alias = alias.strip()
                    if alias and alias != symbol:
                        aliases.append((alias, gene_id, data_source))

    # Insert descriptions
    conn.executemany(
        "INSERT OR REPLACE INTO gene_function (gene_id, description, data_source) VALUES (?, ?, ?)",
        descriptions
    )

    # Insert aliases
    conn.executemany(
        "INSERT OR IGNORE INTO gene_alias (symbol_alias, gene_id, data_source) VALUES (?, ?, ?)",
        aliases
    )
    conn.commit()

    # Update metadata
    conn.execute("""
        INSERT OR REPLACE INTO metadata (source, release, download_url, file_hash, record_count, loaded_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (data_source, config["release"], config["url"], file_hash(filepath), len(descriptions)))
    conn.commit()

    print_step(f"Loaded {len(descriptions):,} descriptions, {len(aliases):,} aliases", "ok")
    return len(descriptions), len(aliases)


def load_go_terms(conn: sqlite3.Connection, data_dir: Path) -> int:
    """Load GO ontology terms from OBO file."""
    config = DATA_SOURCES["go_obo"]
    filepath = data_dir / config["filename"]
    data_source = config["label"]

    if not filepath.exists():
        print_step(f"GO OBO file not found: {filepath}", "error")
        return 0

    print_step("Loading GO terms...")

    terms = []
    edges = []
    current_term = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if line == '[Term]':
                if current_term.get('id'):
                    terms.append((
                        current_term.get('id'),
                        current_term.get('name', ''),
                        current_term.get('namespace', ''),
                        current_term.get('def', ''),
                        1 if current_term.get('is_obsolete') else 0,
                        data_source
                    ))
                    # Add edges
                    for parent in current_term.get('is_a', []):
                        parent_id = parent.split('!')[0].strip()
                        edges.append((current_term['id'], parent_id, 'is_a'))
                current_term = {}

            elif line.startswith('id: GO:'):
                current_term['id'] = line[4:]
            elif line.startswith('name: '):
                current_term['name'] = line[6:]
            elif line.startswith('namespace: '):
                current_term['namespace'] = line[11:]
            elif line.startswith('def: '):
                current_term['def'] = line[5:].split('"')[1] if '"' in line else ''
            elif line.startswith('is_obsolete: true'):
                current_term['is_obsolete'] = True
            elif line.startswith('is_a: '):
                current_term.setdefault('is_a', []).append(line[6:])

    # Don't forget last term
    if current_term.get('id'):
        terms.append((
            current_term.get('id'),
            current_term.get('name', ''),
            current_term.get('namespace', ''),
            current_term.get('def', ''),
            1 if current_term.get('is_obsolete') else 0,
            data_source
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO go_terms VALUES (?, ?, ?, ?, ?, ?)",
        terms
    )
    conn.executemany(
        "INSERT OR IGNORE INTO go_edges VALUES (?, ?, ?)",
        edges
    )
    conn.commit()

    print_step(f"Loaded {len(terms):,} GO terms, {len(edges):,} edges", "ok")
    return len(terms)


def load_go_annotations(conn: sqlite3.Connection, data_dir: Path) -> int:
    """Load GO gene annotations from GAF file."""
    config = DATA_SOURCES["go_gaf"]
    filepath = data_dir / config["filename"]
    data_source = config["label"]

    if not filepath.exists():
        print_step(f"GO GAF file not found: {filepath}", "error")
        return 0

    print_step("Loading GO annotations...")

    # Get existing gene symbols
    cursor = conn.execute("SELECT gene_symbol, gene_id FROM genes")
    symbol_to_geneid = {row[0]: row[1] for row in cursor.fetchall()}

    annotations = []
    open_func = gzip.open if filepath.suffix == '.gz' else open

    with open_func(filepath, 'rt') as f:
        for line in f:
            if line.startswith('!'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 15:
                continue

            symbol = fields[2]
            if symbol not in symbol_to_geneid:
                continue

            gene_id = symbol_to_geneid[symbol]
            go_id = fields[4]
            evidence = fields[6]
            aspect = fields[8]  # P=process, F=function, C=component
            qualifier = fields[3]

            annotations.append((gene_id, go_id, evidence, aspect, qualifier, data_source))

    conn.executemany(
        "INSERT OR IGNORE INTO gene_go VALUES (?, ?, ?, ?, ?, ?)",
        annotations
    )
    conn.commit()

    # Update metadata
    conn.execute("""
        INSERT OR REPLACE INTO metadata (source, release, download_url, file_hash, record_count, loaded_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (data_source, config["release"], config["url"], file_hash(filepath), len(annotations)))
    conn.commit()

    print_step(f"Loaded {len(annotations):,} GO annotations", "ok")
    return len(annotations)


def load_gtex(conn: sqlite3.Connection, data_dir: Path) -> int:
    """Load GTEx expression data."""
    config = DATA_SOURCES["gtex"]
    filepath = data_dir / config["filename"]
    data_source = config["label"]

    if not filepath.exists():
        print_step(f"GTEx file not found: {filepath}", "error")
        return 0

    print_step("Loading GTEx expression data...")

    # Get existing gene IDs
    cursor = conn.execute("SELECT gene_id FROM genes")
    valid_gene_ids = {row[0] for row in cursor.fetchall()}

    expressions = []
    tissues = []

    open_func = gzip.open if filepath.suffix == '.gz' else open

    with open_func(filepath, 'rt') as f:
        # Skip first 2 lines (GCT header)
        f.readline()
        f.readline()

        # Read header with tissue names
        header = f.readline().strip().split('\t')
        tissues = header[2:]  # First two columns are gene_id and Description

        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 3:
                continue

            gene_id = fields[0].split('.')[0]  # Remove version
            if gene_id not in valid_gene_ids:
                continue

            for i, tissue in enumerate(tissues):
                try:
                    level = float(fields[i + 2])
                    if level > 0:  # Only store non-zero expression
                        expressions.append((gene_id, tissue, level, 'TPM', data_source))
                except (ValueError, IndexError):
                    continue

    conn.executemany(
        "INSERT OR REPLACE INTO expression VALUES (?, ?, ?, ?, ?)",
        expressions
    )
    conn.commit()

    # Update metadata
    conn.execute("""
        INSERT OR REPLACE INTO metadata (source, release, download_url, file_hash, record_count, loaded_at, notes)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
    """, (data_source, config["release"], config["url"], file_hash(filepath), len(expressions),
          f"{len(tissues)} tissues"))
    conn.commit()

    print_step(f"Loaded {len(expressions):,} expression values ({len(tissues)} tissues)", "ok")
    return len(expressions)


def load_string_proteins(conn: sqlite3.Connection, data_dir: Path) -> int:
    """Load STRING protein info."""
    config = DATA_SOURCES["string_proteins"]
    filepath = data_dir / config["filename"]
    data_source = config["label"]

    if not filepath.exists():
        print_step(f"STRING proteins file not found: {filepath}", "error")
        return 0

    print_step("Loading STRING proteins...")

    # Get existing gene symbols for mapping
    cursor = conn.execute("SELECT gene_symbol, gene_id FROM genes")
    symbol_to_geneid = {row[0]: row[1] for row in cursor.fetchall()}

    proteins = []
    open_func = gzip.open if filepath.suffix == '.gz' else open

    with open_func(filepath, 'rt') as f:
        header = f.readline()  # Skip header
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 2:
                continue

            # Strip '9606.' prefix from protein_id
            protein_id = fields[0].removeprefix('9606.')
            symbol = fields[1] if len(fields) > 1 else ''
            annotation = fields[2] if len(fields) > 2 else ''

            # Try to map to gene_id
            gene_id = symbol_to_geneid.get(symbol)

            proteins.append((protein_id, symbol, gene_id, annotation, data_source))

    conn.executemany(
        "INSERT OR REPLACE INTO string_proteins VALUES (?, ?, ?, ?, ?)",
        proteins
    )
    conn.commit()

    print_step(f"Loaded {len(proteins):,} STRING proteins", "ok")
    return len(proteins)


def load_string_interactions(conn: sqlite3.Connection, data_dir: Path) -> int:
    """Load STRING protein interactions."""
    config = DATA_SOURCES["string_interactions"]
    filepath = data_dir / config["filename"]
    data_source = config["label"]

    if not filepath.exists():
        print_step(f"STRING interactions file not found: {filepath}", "error")
        return 0

    print_step("Loading STRING interactions (this may take a while)...")

    # Get valid protein IDs (already stored without '9606.' prefix)
    cursor = conn.execute("SELECT protein_id FROM string_proteins")
    valid_proteins = {row[0] for row in cursor.fetchall()}

    interactions = []
    open_func = gzip.open if filepath.suffix == '.gz' else open

    with open_func(filepath, 'rt') as f:
        header = f.readline()  # Skip header
        for line in f:
            fields = line.strip().split(' ')
            if len(fields) < 3:
                continue

            # Strip '9606.' prefix from protein_ids
            p1 = fields[0].removeprefix('9606.')
            p2 = fields[1].removeprefix('9606.')
            try:
                score = int(fields[2])
            except ValueError:
                continue

            # Only store if both proteins exist and score >= 400 (medium confidence)
            if p1 in valid_proteins and p2 in valid_proteins and score >= 400:
                # Store in consistent order
                if p1 < p2:
                    interactions.append((p1, p2, score, data_source))
                else:
                    interactions.append((p2, p1, score, data_source))

    conn.executemany(
        "INSERT OR IGNORE INTO string_interactions VALUES (?, ?, ?, ?)",
        interactions
    )
    conn.commit()

    # Update metadata
    conn.execute("""
        INSERT OR REPLACE INTO metadata (source, release, download_url, file_hash, record_count, loaded_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (data_source, config["release"], config["url"], file_hash(filepath), len(interactions)))
    conn.commit()

    print_step(f"Loaded {len(interactions):,} interactions (score >= 400)", "ok")
    return len(interactions)


def load_string_aliases(conn: sqlite3.Connection, data_dir: Path) -> int:
    """Load STRING protein aliases."""
    config = DATA_SOURCES["string_aliases"]
    filepath = data_dir / config["filename"]
    data_source = config["label"]

    if not filepath.exists():
        print_step(f"STRING aliases file not found: {filepath}", "error")
        return 0

    print_step("Loading STRING aliases...")

    # Get valid protein IDs (already stored without '9606.' prefix)
    cursor = conn.execute("SELECT protein_id FROM string_proteins")
    valid_proteins = {row[0] for row in cursor.fetchall()}

    aliases = []
    open_func = gzip.open if filepath.suffix == '.gz' else open

    with open_func(filepath, 'rt') as f:
        header = f.readline()  # Skip header
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 3:
                continue

            # Strip '9606.' prefix from protein_id
            protein_id = fields[0].removeprefix('9606.')
            alias = fields[1]
            source = fields[2]

            if protein_id in valid_proteins:
                aliases.append((alias, protein_id, source, data_source))

    conn.executemany(
        "INSERT OR IGNORE INTO string_aliases VALUES (?, ?, ?, ?)",
        aliases
    )
    conn.commit()

    print_step(f"Loaded {len(aliases):,} aliases", "ok")
    return len(aliases)


# ============================================================================
# Verification
# ============================================================================

def verify_database(conn: sqlite3.Connection) -> bool:
    """Run basic verification checks."""
    print_header("Verification")

    checks = [
        ("genes", 50000),
        ("gene_function", 50000),
        ("gene_go", 100000),
        ("go_terms", 40000),
        ("expression", 100000),
        ("string_proteins", 15000),
        ("string_interactions", 100000),
    ]

    all_ok = True
    for table, min_rows in checks:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count >= min_rows:
                print_step(f"{table}: {count:,} rows", "ok")
            else:
                print_step(f"{table}: {count:,} rows (expected >= {min_rows:,})", "warn")
                all_ok = False
        except sqlite3.OperationalError as e:
            print_step(f"{table}: ERROR - {e}", "error")
            all_ok = False

    # Check indices
    cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")
    idx_count = cursor.fetchone()[0]
    print_step(f"Indices: {idx_count}", "ok" if idx_count >= 15 else "warn")

    # Database size
    cursor = conn.execute("PRAGMA page_count")
    page_count = cursor.fetchone()[0]
    cursor = conn.execute("PRAGMA page_size")
    page_size = cursor.fetchone()[0]
    size_mb = (page_count * page_size) / 1e6
    print_step(f"Database size: {size_mb:.1f} MB")

    return all_ok


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build gene annotation database from public data sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/setup_database.py                       # Full setup
    python scripts/setup_database.py --skip-download       # Use existing files
    python scripts/setup_database.py --data-dir ./my_data  # Custom data location
    python scripts/setup_database.py --output ./my.db      # Custom output path
        """
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output database path (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory for downloaded data files (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step, use existing files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing database"
    )

    args = parser.parse_args()

    print_header("Gene Annotation Database Setup")
    print(f"  Output: {args.output}")
    print(f"  Data directory: {args.data_dir}")

    # Check if output exists
    if args.output.exists() and not args.force:
        print(f"\nError: Database already exists at {args.output}")
        print("Use --force to overwrite")
        sys.exit(1)

    # Create directories
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Download data
    if not args.skip_download:
        print_header("Downloading Data")
        for source, config in DATA_SOURCES.items():
            dest = args.data_dir / config["filename"]
            if dest.exists():
                print_step(f"{config['description']}: already exists", "skip")
            else:
                success = download_file(
                    config["url"],
                    dest,
                    config["description"],
                    fallback_urls=config.get("fallback_urls"),
                    manual_url=config.get("manual_url"),
                )
                if not success:
                    print(f"\nError downloading {source}. Use --skip-download with existing files.")
                    sys.exit(1)
    else:
        print_step("Skipping downloads (using existing files)")

    # Remove existing database if force
    if args.output.exists():
        args.output.unlink()

    # Create database
    print_header("Creating Database")
    conn = sqlite3.connect(args.output)

    try:
        # Create schema
        create_schema(conn)

        # Load data
        print_header("Loading Data")
        load_gencode(conn, args.data_dir)
        load_ncbi_gene_info(conn, args.data_dir)  # SHORT descriptions
        load_go_terms(conn, args.data_dir)
        load_go_annotations(conn, args.data_dir)
        load_gtex(conn, args.data_dir)
        load_string_proteins(conn, args.data_dir)
        load_string_interactions(conn, args.data_dir)
        load_string_aliases(conn, args.data_dir)

        # Optimise
        print_header("Optimising")
        print_step("Running ANALYZE...")
        conn.execute("ANALYZE")
        print_step("Running VACUUM...")
        conn.execute("VACUUM")
        print_step("Optimisation complete", "ok")

        # Verify
        success = verify_database(conn)

        print_header("Summary")
        if success:
            print_step("Database setup completed successfully!", "ok")
            print(f"\n  Database: {args.output}")
            print(f"  Size: {args.output.stat().st_size / 1e9:.2f} GB")
        else:
            print_step("Database setup completed with warnings", "warn")
            sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
