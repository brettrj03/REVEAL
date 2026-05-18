"""
Build Literature Query Plan with Tiered Fallback Strategy
Uses LLM-based context extraction and builds adaptive query tiers.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
import time
from pydantic_graph.nodes import BaseNode, GraphRunContext
from src.graph.state import GeneState
from src.utils.llm_context_extractor import LLMContextExtractor
from src.utils.adaptive_literature_queries import AdaptiveLiteratureQueryBuilder, get_gene_info_from_db
import os

if TYPE_CHECKING:
    from src.nodes.fetch_adaptive_literature import FetchAdaptiveLiterature


@dataclass
class BuildLiteratureQueryPlan(BaseNode[GeneState]):
    """Build tiered query plans for adaptive literature search"""

    async def run(self, ctx: GraphRunContext[GeneState]) -> "FetchAdaptiveLiterature":
        from src.nodes.fetch_adaptive_literature import FetchAdaptiveLiterature

        print(f"\n{'='*70}")
        print("NODE: Build Adaptive Literature Query Plan")
        print("Strategy: Tiered queries from specific to broad")
        print(f"{'='*70}")

        state = ctx.state
        _t0 = time.perf_counter()

        try:
            # Extract context using LLM
            if os.getenv('OPENAI_API_KEY'):
                try:
                    extractor = LLMContextExtractor()
                    context_terms = await extractor.extract_context(
                        user_query=state.user_query,
                        experiment_context=state.experiment_context,
                        state=state,
                        node_name=self.__class__.__name__
                    )
                    print("Context extracted with LLM")
                except Exception as e:
                    print(f"LLM extraction failed, using fallback: {e}")
                    extractor = LLMContextExtractor(openai_api_key="dummy")
                    context_terms = extractor._fallback_extraction(
                        state.user_query,
                        state.experiment_context
                    )
            else:
                print("No OpenAI key, using fallback extraction")
                extractor = LLMContextExtractor(openai_api_key="dummy")
                context_terms = extractor._fallback_extraction(
                    state.user_query,
                    state.experiment_context
                )

            # Store extracted context
            state.literature_context_terms = context_terms
            state.literature_disease_terms_detected = context_terms.get('diseases', [])

            # Build tiered queries for each gene
            builder = AdaptiveLiteratureQueryBuilder()
            genes = state.get_genes_found()

            print(f"\nBuilding tiered queries for {len(genes)} genes...")
            print(f"Context extracted:")
            print(f"  Diseases: {context_terms.get('diseases', [])[:3]}")
            print(f"  Processes: {context_terms.get('processes', [])[:3]}")
            print(f"  Cell types: {context_terms.get('cell_types', [])[:3]}")
            print()

            for gene in genes:
                tiers = builder.build_all_tiers(
                    gene=gene,
                    context_terms=context_terms,
                    user_query=state.user_query
                )

                state.literature_query_tiers[gene] = tiers
                state.literature_query_plan[gene] = {
                    tier.name: tier.query
                    for tier in tiers
                }

                print(f"  {gene}: Built {len(tiers)} query tiers")
                for tier in tiers:
                    print(f"    Tier {tier.tier_number}: {tier.description}")

            print(f"\n{'='*70}")
            print("QUERY PLAN SUMMARY")
            print(f"{'='*70}")
            print(f"Genes: {len(genes)}")
            print(f"Diseases: {len(context_terms.get('diseases', []))}")
            print(f"Processes: {len(context_terms.get('processes', []))}")
            print(f"Cell types: {len(context_terms.get('cell_types', []))}")
            context_genes = context_terms.get('context_genes', [])
            if context_genes:
                print(f"Context genes (from query): {context_genes}")
            print(f"{'='*70}\n")

            return FetchAdaptiveLiterature()
        finally:
            state.log_node_execution(
                self.__class__.__name__,
                round(time.perf_counter() - _t0, 3)
            )
