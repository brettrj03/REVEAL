"""Phase 3 interpretation of GO overlap patterns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import sqlite3
import time

from contextlib import closing

from pydantic_graph.nodes import BaseNode, GraphRunContext

from src.agents.go_interpretation_agent import (
    GoInterpretationContext,
    generate_go_interpretation,
)
from src.graph.state import GeneState
from src.utils.tracing import trace_event


@dataclass
class InterpretGoPatterns(BaseNode[GeneState]):
    """Generate GO overlap interpretation when in interpreted mode."""

    async def run(self, ctx: GraphRunContext[GeneState]) -> "ValidateGoInterpretation":
        _t0 = time.perf_counter()
        try:
            from src.nodes.validate_go_interpretation import ValidateGoInterpretation

            print(f"\n{'='*70}")
            print("NODE: Interpret GO Patterns")
            print(f"{'='*70}")

            analysis = ctx.state.go_comparison_analysis or {}
            has_terms = bool(analysis.get('shared_terms'))

            if ctx.state.output_mode == "factual" or not has_terms:
                print("Skipping GO interpretation (factual mode or no shared terms)")
                return ValidateGoInterpretation()

            shared_terms_raw = (analysis.get('shared_terms') or [])[:100]
            enriched_terms = _enrich_shared_terms(shared_terms_raw, ctx.state.db_path)

            try:
                context = GoInterpretationContext(
                    genes=sorted(ctx.state.gene_profiles.keys()),
                    shared_terms=enriched_terms,
                    overlap_stats=analysis.get('overlap_stats', {}),
                    experimental_context=
                        ctx.state.experiment_context.model_dump()
                        if ctx.state.experiment_context
                        else None,
                )
                description = await generate_go_interpretation(
                    context,
                    state=ctx.state,
                    node_name="InterpretGoPatterns",
                )
                ctx.state.go_comparison_analysis['interpretation'] = description
                print("✓ GO interpretation generated")
                trace_event(
                    "interpretation.go_overlap",
                    genes=context.genes,
                    terms=len(context.shared_terms),
                    state_inputs=['go_comparison_analysis']
                )
            except Exception as exc:  # pragma: no cover - best effort
                print(f"⚠️  Failed to interpret GO overlap: {exc}")

            return ValidateGoInterpretation()
        finally:
            ctx.state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )


def _enrich_shared_terms(terms: List[Dict[str, Any]], db_path: str | None) -> List[Dict[str, Any]]:
    """Attach GO definitions and depth information for each shared term."""

    if not terms:
        return []

    resolved_db = _resolve_db_path(db_path)
    connection = None
    has_depth_column = False

    if resolved_db:
        try:
            connection = sqlite3.connect(f"file:{resolved_db}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            has_depth_column = _table_has_column(connection, 'go_terms', 'depth')
        except sqlite3.Error as exc:
            print(f"⚠️  Unable to open GO database at {resolved_db}: {exc}")
            connection = None

    depth_cache: Dict[str, int] = {}
    enriched: List[Dict[str, Any]] = []

    try:
        for term in terms:
            go_id = term.get('go_id') or term.get('id') or term.get('term_id') or ''
            genes = term.get('genes') or term.get('shared_by') or []
            if isinstance(genes, set):
                genes = sorted(genes)
            elif not isinstance(genes, list):
                genes = list(genes) if genes else []

            enriched_term = {
                'term_name': term.get('name') or term.get('term') or '',
                'go_id': go_id,
                'namespace': term.get('namespace') or term.get('aspect') or term.get('category') or '',
                'definition': '',
                'depth': 0,
                'genes': genes,
            }

            if connection and go_id:
                try:
                    if has_depth_column:
                        row = connection.execute(
                            "SELECT go_id, name, namespace, definition, depth FROM go_terms WHERE go_id = ?",
                            (go_id,)
                        ).fetchone()
                    else:
                        row = connection.execute(
                            "SELECT go_id, name, namespace, definition FROM go_terms WHERE go_id = ?",
                            (go_id,)
                        ).fetchone()
                except sqlite3.Error as exc:
                    print(f"⚠️  GO term lookup failed for {go_id}: {exc}")
                    row = None

                if row:
                    enriched_term['definition'] = row['definition'] or ''
                    if not enriched_term['namespace']:
                        enriched_term['namespace'] = row['namespace'] or ''
                    if not enriched_term['term_name']:
                        enriched_term['term_name'] = row['name'] or ''
                    if has_depth_column and 'depth' in row.keys():
                        depth_val = row['depth']
                        enriched_term['depth'] = int(depth_val) if depth_val is not None else 0
                    else:
                        enriched_term['depth'] = _compute_term_depth(connection, go_id, depth_cache)
                else:
                    enriched_term['definition'] = enriched_term['definition'] or ''
                    enriched_term['depth'] = _compute_term_depth(connection, go_id, depth_cache)
            else:
                enriched_term['definition'] = enriched_term['definition'] or ''

            enriched.append(enriched_term)
    finally:
        if connection:
            connection.close()

    enriched.sort(key=lambda t: t.get('depth', 0) or 0, reverse=True)
    return enriched


def _table_has_column(connection: sqlite3.Connection, table: str, column: str) -> bool:
    with closing(connection.cursor()) as cursor:
        cursor.execute(f"PRAGMA table_info({table})")
        return any(row[1] == column for row in cursor.fetchall())


def _compute_term_depth(connection: sqlite3.Connection, go_id: str, cache: Dict[str, int]) -> int:
    if not go_id:
        return 0
    if go_id in cache:
        return cache[go_id]

    query = """
        WITH RECURSIVE ancestors(go_id, depth, path) AS (
            SELECT ?, 0, '|' || ? || '|'
            UNION ALL
            SELECT ge.parent_go_id, ancestors.depth + 1, ancestors.path || ge.parent_go_id || '|'
            FROM go_edges ge
            JOIN ancestors ON ge.child_go_id = ancestors.go_id
            WHERE ancestors.depth < 40
              AND INSTR(ancestors.path, '|' || ge.parent_go_id || '|') = 0
        )
        SELECT MAX(depth) FROM ancestors;
    """

    depth_value = 0
    try:
        row = connection.execute(query, (go_id, go_id)).fetchone()
        if row and row[0] is not None:
            depth_value = int(row[0])
    except sqlite3.Error as exc:
        print(f"⚠️  Unable to compute GO term depth for {go_id}: {exc}")

    cache[go_id] = depth_value
    return depth_value


def _resolve_db_path(custom_path: str | None) -> Path | None:
    candidates: List[Path] = []
    if custom_path:
        candidates.append(Path(custom_path).expanduser())
    candidates.append(Path('src/database/gene_database.sqlite'))

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None
