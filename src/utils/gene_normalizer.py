"""Rule-based gene name normalization utilities.

The normalization process is intentionally database-independent so that we can
clean noisy extraction results before attempting any lookups.
"""

from collections import Counter
from typing import Dict, List, Any, Optional
import re

from src.models.models import GeneNameMapping

_WHITESPACE_PATTERN = re.compile(r'\s+')


def _basic_clean(name: str) -> str:
    if name is None:
        return ''
    cleaned = name.strip()
    cleaned = re.sub(_WHITESPACE_PATTERN, ' ', cleaned)
    return cleaned


def normalize_gene_list(genes: List[str]) -> GeneNameMapping:
    """Normalize a list of genes using deterministic heuristics."""
    normalization_records: List[Dict[str, Any]] = []
    normalized_values: List[str] = []
    base_values: List[str] = []
    transformations: List[Dict[str, Any]] = []
    suggestions: Dict[str, Dict[str, str]] = {}

    for raw in genes:
        original = str(raw)
        cleaned = _basic_clean(original)
        if not cleaned:
            continue

        normalized = cleaned
        base_normalized = cleaned.upper()

        record = {
            'original': original,
            'cleaned': cleaned,
            'base_normalized': base_normalized,
            'normalized': normalized,
            'reasons': [],
            'notes': [],
            'changed': False,
            'suggestion': None
        }
        normalization_records.append(record)
        normalized_values.append(normalized)
        base_values.append(base_normalized)

    duplicates: Dict[str, int] = {}
    first_seen: Dict[str, str] = {}
    counts = Counter(base_values)
    for record in normalization_records:
        key = record['base_normalized']
        first_seen.setdefault(key, record['original'])
    for key, count in counts.items():
        if count > 1:
            display = first_seen.get(key, key)
            duplicates[display] = count

    unique_genes: List[str] = []
    seen_keys = set()
    for value in normalized_values:
        if value not in seen_keys:
            unique_genes.append(value)
            seen_keys.add(value)

    return GeneNameMapping(
        user_genes=[record['original'] for record in normalization_records],
        normalized_genes=normalized_values,
        unique_genes=unique_genes,
        normalization_records=normalization_records,
        duplicates=duplicates,
        transformations=transformations,
        suggestions=suggestions,
    )

def _wrap_gene_list(items: List[str], indent: str = '', width: int = 68) -> str:
    if not items:
        return ''

    lines: List[str] = []
    current = ''
    for item in items:
        piece = item if not current else f", {item}"
        if not current:
            candidate = item
        else:
            candidate = current + f", {item}"

        if len(candidate) > width and current:
            lines.append(indent + current)
            current = item
        else:
            current = candidate

    if current:
        lines.append(indent + current)

    return '\n'.join(lines)


def format_gene_resolution_summary(
    mapping_data: Optional[Any],
    retrieval_stats: Optional[Dict[str, Any]] = None,
    gene_mapping: Optional[Dict[str, Any]] = None
) -> str:
    if not mapping_data:
        return ''

    if isinstance(mapping_data, GeneNameMapping):
        data = mapping_data.model_dump()
    else:
        data = mapping_data

    user_genes = data.get('user_genes', [])
    duplicates = data.get('duplicates', {}) or {}
    normalized = data.get('unique_genes') or data.get('normalized_genes', [])
    transformations = data.get('transformations', []) or []
    suggestions = data.get('suggestions', {}) or {}
    resolution_status = data.get('resolution_status', {}) or {}

    if not resolution_status and gene_mapping:
        resolution_status = {
            gene: {
                'found': info.get('found', False),
                'official_symbol': info.get('official_symbol'),
                'direct_input': True
            }
            for gene, info in gene_mapping.items()
        }

    input_total = len(user_genes)
    duplicate_instances = sum(max(count - 1, 0) for count in duplicates.values())
    unique_total = len(normalized)

    if retrieval_stats:
        found_total = retrieval_stats.get('genes_found') or 0
    else:
        found_total = sum(1 for status in resolution_status.values() if status.get('found'))

    not_found_total = max(unique_total - found_total, 0)
    success_pct = (found_total / unique_total * 100) if unique_total else 0
    missing_pct = (not_found_total / unique_total * 100) if unique_total else 0

    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("GENES ANALYZED")
    lines.append("=" * 70)
    lines.append("")

    if duplicate_instances:
        plural = 's' if duplicate_instances > 1 else ''
        lines.append(f"Input genes: {input_total} (including {duplicate_instances} duplicate{plural})")
    else:
        lines.append(f"Input genes: {input_total}")
    lines.append(f"Unique genes: {unique_total}")
    lines.append(f"Successfully mapped: {found_total}/{unique_total} ({success_pct:.1f}%)")
    lines.append(f"Not found: {not_found_total}/{unique_total} ({missing_pct:.1f}%)")
    lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("GENE NAME RESOLUTION SUMMARY")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")

    direct_matches: List[str] = []
    included_genes: List[str] = []
    for gene in normalized:
        status = resolution_status.get(gene, {})
        official = status.get('official_symbol') or gene
        if status.get('found') and official not in included_genes:
            included_genes.append(official)
        if status.get('found') and status.get('direct_input'):
            if official not in direct_matches:
                direct_matches.append(official)

    lines.append(f"✓ DIRECT MATCHES ({len(direct_matches)} genes):")
    if direct_matches:
        lines.append(_wrap_gene_list(direct_matches))
    else:
        lines.append("(none)")
    lines.append("")

    tier_counts = Counter(
        status.get('tier')
        for status in resolution_status.values()
        if status.get('found') and status.get('tier')
    )
    lines.append("Lookup outcomes:")
    lines.append(f"  • Found in common_genes table: {tier_counts.get('common_genes', 0)}")
    lines.append(f"  • Found in gene_alias table: {tier_counts.get('gene_alias', 0)}")
    lines.append(f"  • Found in string_alias table: {tier_counts.get('string_alias', 0)}")
    lines.append(f"  • NOT FOUND: {not_found_total}")
    lines.append("")

    lines.append(f"🔄  MAPPED VIA CORRECTION ({len(transformations)}):")
    if transformations:
        reason_labels = {
            'common_genes': 'found in common_genes table',
            'gene_alias': 'found in gene_alias table',
            'string_alias': 'found in string_alias table'
        }
        for entry in transformations:
            reasons = entry.get('reasons', [])
            if reasons:
                tag = ', '.join(reason_labels.get(r, r) for r in reasons)
            else:
                tag = 'database lookup'
            lines.append(f"  • {entry['original']} → {entry['normalized']} ({tag})")
    else:
        lines.append("  • None")
    lines.append("")

    if duplicates:
        duplicate_genes = len(duplicates)
        lines.append(f"⚠️  DUPLICATES REMOVED ({duplicate_genes}):")
        for gene, count in duplicates.items():
            plural = 'times' if count > 1 else 'time'
            lines.append(f"  • {gene} (appeared {count} {plural} in input)")
    else:
        lines.append("⚠️  DUPLICATES REMOVED: None")
    lines.append("")

    missing_genes: List[str] = [
        gene for gene, status in resolution_status.items()
        if not status.get('found')
    ]
    lines.append(f"❌ NOT FOUND ({len(missing_genes)} genes):")
    if missing_genes:
        for gene in missing_genes:
            status = resolution_status.get(gene, {})
            note = status.get('notes') or ''
            suffix = f" ({note})" if note else ''
            lines.append(f"  • {gene}{suffix}")
    else:
        lines.append("  • None")
    lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"GENES INCLUDED IN ANALYSIS ({len(included_genes)} total):")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if included_genes:
        lines.append(_wrap_gene_list(included_genes))
    else:
        lines.append("No genes retrieved from the database.")

    return '\n'.join(lines) + '\n'


__all__ = ['normalize_gene_list', 'format_gene_resolution_summary']
