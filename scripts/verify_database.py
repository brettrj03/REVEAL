#!/usr/bin/env python3
"""
Database Verification Script for REVEAL

Verifies database integrity, schema, and data quality.

Usage:
    python scripts/verify_database.py                    # Verify default database
    python scripts/verify_database.py --db /path/to.db   # Verify specific database
    python scripts/verify_database.py --detailed         # Show detailed output
    python scripts/verify_database.py --quick            # Quick check only

Author: REVEAL Team
Date: 2025-01
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple


DEFAULT_DB_PATH = Path(__file__).parent.parent / "src" / "database" / "gene_database.sqlite"


# ============================================================================
# Color Codes for Terminal Output
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class CheckResult(NamedTuple):
    """Result of a verification check."""
    name: str
    status: str  # 'pass', 'warn', 'fail'
    actual: str
    expected: str
    category: str


# ============================================================================
# Expected Schema
# ============================================================================

EXPECTED_TABLES = {
    "genes": {
        "min_rows": 70000,
        "max_rows": 90000,
        "required_columns": ["gene_id", "gene_symbol", "gene_type", "chromosome", "start", "end", "strand", "data_source"],
    },
    "gene_function": {
        "min_rows": 35000,
        "max_rows": 50000,
        "required_columns": ["gene_id", "description"],
    },
    "gene_go": {
        "min_rows": 350000,
        "max_rows": 500000,
        "required_columns": ["gene_id", "go_id", "evidence"],
    },
    "go_terms": {
        "min_rows": 40000,
        "max_rows": 60000,
        "required_columns": ["go_id", "name", "namespace"],
    },
    "go_edges": {
        "min_rows": 50000,
        "max_rows": 70000,
        "required_columns": ["child_go_id", "parent_go_id", "relation_type"],
    },
    "expression": {
        "min_rows": 1400000,
        "max_rows": 1700000,
        "required_columns": ["gene_id", "tissue", "tpm_value"],
    },
    "string_proteins": {
        "min_rows": 15000,
        "max_rows": 25000,
        "required_columns": ["protein_id", "symbol"],
    },
    "string_interactions": {
        "min_rows": 800000,
        "max_rows": 1200000,
        "required_columns": ["protein_id_1", "protein_id_2", "combined_score"],
    },
    "string_aliases": {
        "min_rows": 2500000,
        "max_rows": 4000000,
        "required_columns": ["symbol_alias", "protein_id"],
    },
    "gene_alias": {
        "min_rows": 60000,
        "max_rows": 90000,
        "required_columns": ["symbol_alias", "gene_id"],
    },
    "metadata": {
        "min_rows": 4,
        "max_rows": 20,
        "required_columns": ["source", "release", "loaded_at"],
    },
}

EXPECTED_INDICES = [
    "idx_genes_symbol",
    "idx_gene_go_gene_id",
    "idx_expression_gene_id",
    "idx_string_proteins_symbol",
]

CRITICAL_INDICES = [
    "idx_genes_symbol",
    "idx_gene_go_gene_id",
]


# ============================================================================
# Verification Functions
# ============================================================================

def check_tables(conn: sqlite3.Connection, detailed: bool = False) -> Tuple[int, int, List[str], List[CheckResult]]:
    """Check all expected tables exist and have correct row counts."""
    passed = 0
    failed = 0
    issues = []
    results = []

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}

    total_tables = len(EXPECTED_TABLES)
    for idx, (table, config) in enumerate(EXPECTED_TABLES.items(), 1):
        print(f"  [{idx}/{total_tables}] Checking {table}...", end=" ")

        if table not in existing_tables:
            print(f"{Colors.RED}✗ MISSING{Colors.RESET}")
            failed += 1
            issues.append(f"Table {table} is missing")
            results.append(CheckResult(
                name=table,
                status='fail',
                actual='MISSING',
                expected='Required',
                category='Tables'
            ))
            continue

        # Check row count
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        min_rows = config["min_rows"]
        max_rows = config["max_rows"]

        if min_rows <= count <= max_rows:
            print(f"{Colors.GREEN}✓ {count:,} rows{Colors.RESET}")
            passed += 1
            results.append(CheckResult(
                name=table,
                status='pass',
                actual=f"{count:,}",
                expected=f"{min_rows:,}-{max_rows:,}",
                category='Tables'
            ))
        else:
            print(f"{Colors.YELLOW}⚠ {count:,} rows (expected {min_rows:,}-{max_rows:,}){Colors.RESET}")
            failed += 1
            issues.append(f"Table {table} has {count:,} rows (expected {min_rows:,}-{max_rows:,})")
            results.append(CheckResult(
                name=table,
                status='warn',
                actual=f"{count:,}",
                expected=f"{min_rows:,}-{max_rows:,}",
                category='Tables'
            ))

        # Check required columns
        if detailed:
            cursor = conn.execute(f"PRAGMA table_info({table})")
            columns = {row[1] for row in cursor.fetchall()}
            missing = set(config["required_columns"]) - columns
            if missing:
                print(f"    {Colors.YELLOW}⚠ Missing columns: {missing}{Colors.RESET}")
                issues.append(f"Table {table} missing columns: {missing}")

    return passed, failed, issues, results


def check_indices(conn: sqlite3.Connection, detailed: bool = False) -> Tuple[int, int, List[str], List[CheckResult]]:
    """Check important indices exist."""
    passed = 0
    failed = 0
    issues = []
    results = []

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")
    existing_indices = {row[0] for row in cursor.fetchall()}

    print(f"\n  {Colors.BLUE}Found {len(existing_indices)} indices{Colors.RESET}")

    # Check critical indices
    total_critical = len(CRITICAL_INDICES)
    for idx, index in enumerate(CRITICAL_INDICES, 1):
        print(f"  [{idx}/{total_critical}] Checking {index}...", end=" ")
        if index in existing_indices:
            print(f"{Colors.GREEN}✓ Present{Colors.RESET}")
            passed += 1
            results.append(CheckResult(
                name=index,
                status='pass',
                actual='Present',
                expected='Required',
                category='Indices'
            ))
        else:
            print(f"{Colors.RED}✗ MISSING (critical){Colors.RESET}")
            failed += 1
            issues.append(f"Critical index {index} is missing")
            results.append(CheckResult(
                name=index,
                status='fail',
                actual='Missing',
                expected='Required',
                category='Indices'
            ))

    if detailed:
        print(f"\n  All indices:")
        for idx in sorted(existing_indices):
            print(f"    - {idx}")

    return passed, failed, issues, results


def check_data_quality(conn: sqlite3.Connection) -> Tuple[int, int, List[str], List[CheckResult]]:
    """Run data quality checks."""
    passed = 0
    failed = 0
    issues = []
    results = []

    checks = [
        # Genes with symbols
        ("Genes have symbols",
         "SELECT COUNT(*) FROM genes WHERE gene_symbol IS NULL OR gene_symbol = ''",
         0, "=="),

        # GO terms have names
        ("GO terms have names",
         "SELECT COUNT(*) FROM go_terms WHERE name IS NULL OR name = ''",
         100, "<="),  # Some obsolete terms may lack names

        # Expression values are positive
        ("Expression values positive",
         "SELECT COUNT(*) FROM expression WHERE tpm_value < 0",
         0, "=="),

        # STRING scores in valid range
        ("STRING scores valid",
         "SELECT COUNT(*) FROM string_interactions WHERE combined_score < 0 OR combined_score > 1000",
         0, "=="),

        # Expression has multiple tissues
        ("Expression tissue count",
         "SELECT COUNT(DISTINCT tissue) FROM expression",
         54, ">="),

        # Genes linked to expression
        ("Genes with expression",
         "SELECT COUNT(DISTINCT gene_id) FROM expression",
         40000, ">="),

        # GO terms linked to genes
        ("GO terms with genes",
         "SELECT COUNT(DISTINCT go_id) FROM gene_go",
         15000, ">="),
    ]

    total_checks = len(checks)
    for idx, (name, query, expected, op) in enumerate(checks, 1):
        print(f"  [{idx}/{total_checks}] Checking {name}...", end=" ")
        actual = conn.execute(query).fetchone()[0]

        expected_str = f"{op} {expected:,}"

        if op == "==" and actual == expected:
            print(f"{Colors.GREEN}✓ {actual:,}{Colors.RESET}")
            passed += 1
            results.append(CheckResult(
                name=name,
                status='pass',
                actual=f"{actual:,}",
                expected=expected_str,
                category='Data Quality'
            ))
        elif op == "<=" and actual <= expected:
            print(f"{Colors.GREEN}✓ {actual:,}{Colors.RESET}")
            passed += 1
            results.append(CheckResult(
                name=name,
                status='pass',
                actual=f"{actual:,}",
                expected=expected_str,
                category='Data Quality'
            ))
        elif op == ">=" and actual >= expected:
            print(f"{Colors.GREEN}✓ {actual:,}{Colors.RESET}")
            passed += 1
            results.append(CheckResult(
                name=name,
                status='pass',
                actual=f"{actual:,}",
                expected=expected_str,
                category='Data Quality'
            ))
        else:
            print(f"{Colors.YELLOW}⚠ {actual:,} (expected {op} {expected:,}){Colors.RESET}")
            failed += 1
            issues.append(f"{name}: got {actual:,}, expected {op} {expected:,}")
            results.append(CheckResult(
                name=name,
                status='warn',
                actual=f"{actual:,}",
                expected=expected_str,
                category='Data Quality'
            ))

    return passed, failed, issues, results


def check_foreign_keys(conn: sqlite3.Connection) -> Tuple[int, int, List[str], List[CheckResult]]:
    """Check referential integrity."""
    passed = 0
    failed = 0
    issues = []
    results = []

    # Enable foreign key checking
    conn.execute("PRAGMA foreign_keys = ON")

    checks = [
        # gene_go references valid genes
        ("gene_go → genes",
         """SELECT COUNT(*) FROM gene_go gg
            LEFT JOIN genes g ON gg.gene_id = g.gene_id
            WHERE g.gene_id IS NULL""",
         1000),  # Allow some orphans from removed genes

        # gene_go references valid go_terms
        ("gene_go → go_terms",
         """SELECT COUNT(*) FROM gene_go gg
            LEFT JOIN go_terms gt ON gg.go_id = gt.go_id
            WHERE gt.go_id IS NULL""",
         100),  # Allow some orphans from obsolete terms

        # expression references valid genes
        ("expression → genes",
         """SELECT COUNT(*) FROM expression e
            LEFT JOIN genes g ON e.gene_id = g.gene_id
            WHERE g.gene_id IS NULL""",
         1000),

        # string_proteins have gene mappings
        ("string_proteins → genes",
         """SELECT COUNT(*) FROM string_proteins sp
            WHERE sp.gene_id IS NOT NULL
            AND sp.gene_id NOT IN (SELECT gene_id FROM genes)""",
         500),
    ]

    total_checks = len(checks)
    for idx, (name, query, max_orphans) in enumerate(checks, 1):
        print(f"  [{idx}/{total_checks}] Checking {name}...", end=" ")
        orphans = conn.execute(query).fetchone()[0]

        if orphans <= max_orphans:
            print(f"{Colors.GREEN}✓ {orphans:,} orphans (max {max_orphans:,}){Colors.RESET}")
            passed += 1
            results.append(CheckResult(
                name=name,
                status='pass',
                actual=f"{orphans:,} orphans",
                expected=f"≤ {max_orphans:,}",
                category='Referential Integrity'
            ))
        else:
            print(f"{Colors.YELLOW}⚠ {orphans:,} orphans (max {max_orphans:,}){Colors.RESET}")
            failed += 1
            issues.append(f"{name}: {orphans:,} orphan records")
            results.append(CheckResult(
                name=name,
                status='warn',
                actual=f"{orphans:,} orphans",
                expected=f"≤ {max_orphans:,}",
                category='Referential Integrity'
            ))

    return passed, failed, issues, results


def print_summary_table(all_results: List[CheckResult], total_passed: int, total_failed: int):
    """Print a comprehensive summary table of all checks."""
    print(f"\n{'='*80}")
    print(f"{Colors.BOLD}  VERIFICATION SUMMARY{Colors.RESET}")
    print(f"{'='*80}\n")

    # Group results by category
    categories = {}
    for result in all_results:
        if result.category not in categories:
            categories[result.category] = []
        categories[result.category].append(result)

    # Print table header
    print(f"  {Colors.BOLD}{'Category':<25} {'Check':<30} {'Status':<10} {'Actual':<15} {'Expected':<15}{Colors.RESET}")
    print(f"  {'-'*25} {'-'*30} {'-'*10} {'-'*15} {'-'*15}")

    # Print results by category
    for category in ['Tables', 'Indices', 'Data Quality', 'Referential Integrity']:
        if category not in categories:
            continue

        category_results = categories[category]
        for idx, result in enumerate(category_results):
            category_label = category if idx == 0 else ""

            # Status symbol with color
            if result.status == 'pass':
                status_str = f"{Colors.GREEN}✓ PASS{Colors.RESET}"
            elif result.status == 'warn':
                status_str = f"{Colors.YELLOW}⚠ WARN{Colors.RESET}"
            else:
                status_str = f"{Colors.RED}✗ FAIL{Colors.RESET}"

            # Truncate long names
            check_name = result.name[:28] + "..." if len(result.name) > 30 else result.name
            actual = result.actual[:13] + ".." if len(result.actual) > 15 else result.actual
            expected = result.expected[:13] + ".." if len(result.expected) > 15 else result.expected

            print(f"  {category_label:<25} {check_name:<30} {status_str:<20} {actual:<15} {expected:<15}")

        print()  # Blank line between categories

    # Overall summary with color
    print(f"{'='*80}")
    pass_rate = (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0

    print(f"\n  {Colors.BOLD}Total Checks:{Colors.RESET} {total_passed + total_failed}")
    print(f"  {Colors.GREEN}{Colors.BOLD}Passed:{Colors.RESET}       {total_passed} ({pass_rate:.1f}%)")

    if total_failed > 0:
        print(f"  {Colors.YELLOW}{Colors.BOLD}Warnings:{Colors.RESET}     {total_failed}")

    # Final verdict
    print(f"\n{'='*80}")
    if total_failed == 0:
        print(f"  {Colors.GREEN}{Colors.BOLD}✓ DATABASE VERIFICATION PASSED{Colors.RESET}")
        print(f"  All checks completed successfully!")
    else:
        print(f"  {Colors.YELLOW}{Colors.BOLD}⚠ DATABASE VERIFICATION COMPLETED WITH WARNINGS{Colors.RESET}")
        print(f"  Review warnings above. Database is functional but some values are outside expected ranges.")

    print(f"{'='*80}\n")


def quick_check(conn: sqlite3.Connection) -> bool:
    """Quick sanity check."""
    print("\n=== Quick Check ===\n")

    # Check core tables exist and have data
    core = [
        ("genes", 50000),
        ("gene_go", 100000),
        ("expression", 100000),
        ("string_interactions", 100000),
    ]

    all_ok = True
    for table, min_rows in core:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count >= min_rows:
                print(f"  ✓ {table}: {count:,} rows")
            else:
                print(f"  ✗ {table}: {count:,} rows (need {min_rows:,}+)")
                all_ok = False
        except sqlite3.OperationalError:
            print(f"  ✗ {table}: missing!")
            all_ok = False

    return all_ok


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify gene annotation database integrity"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to database file"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check only"
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database not found at {args.db}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Database Verification")
    print(f"{'='*60}")
    print(f"\n  Path: {args.db}")
    print(f"  Size: {args.db.stat().st_size / 1e9:.2f} GB")

    conn = sqlite3.connect(args.db)

    try:
        if args.quick:
            success = quick_check(conn)
            sys.exit(0 if success else 1)

        total_passed = 0
        total_failed = 0
        all_issues = []
        all_results = []

        # Table checks
        print(f"\n{'='*60}")
        print(f"  {Colors.CYAN}{Colors.BOLD}[1/4] Checking Tables{Colors.RESET}")
        print(f"{'='*60}\n")
        passed, failed, issues, results = check_tables(conn, args.detailed)
        total_passed += passed
        total_failed += failed
        all_issues.extend(issues)
        all_results.extend(results)

        # Index checks
        print(f"\n{'='*60}")
        print(f"  {Colors.CYAN}{Colors.BOLD}[2/4] Checking Indices{Colors.RESET}")
        print(f"{'='*60}")
        passed, failed, issues, results = check_indices(conn, args.detailed)
        total_passed += passed
        total_failed += failed
        all_issues.extend(issues)
        all_results.extend(results)

        # Data quality checks
        print(f"\n{'='*60}")
        print(f"  {Colors.CYAN}{Colors.BOLD}[3/4] Checking Data Quality{Colors.RESET}")
        print(f"{'='*60}\n")
        passed, failed, issues, results = check_data_quality(conn)
        total_passed += passed
        total_failed += failed
        all_issues.extend(issues)
        all_results.extend(results)

        # Foreign key checks
        print(f"\n{'='*60}")
        print(f"  {Colors.CYAN}{Colors.BOLD}[4/4] Checking Referential Integrity{Colors.RESET}")
        print(f"{'='*60}\n")
        passed, failed, issues, results = check_foreign_keys(conn)
        total_passed += passed
        total_failed += failed
        all_issues.extend(issues)
        all_results.extend(results)

        # Print comprehensive summary table
        print_summary_table(all_results, total_passed, total_failed)

        sys.exit(0 if total_failed == 0 else 1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
