"""
Analyze Literature Findings Node

Analyses collected literature to identify research trends, themes,
and what's already known about each gene in the research context.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, List
from dataclasses import dataclass
from collections import Counter
from datetime import datetime
import os
import json
import logging
import time

from pydantic_graph.nodes import BaseNode, GraphRunContext
from src.graph.state import GeneState, _accumulate_tokens

if TYPE_CHECKING:
    from src.nodes.interpret_all_genes import InterpretAllGenes

logger = logging.getLogger(__name__)


@dataclass
class AnalyzeLiteratureFindings(BaseNode[GeneState]):
    """
    Analyse collected literature to extract research trends and themes.

    Reads from:
        - state.gene_top_papers: Ranked papers per gene
        - state.gene_literature_candidates: All candidate papers
        - state.pubmed_counts: Paper counts per search tier

    Writes to:
        - state.literature_findings_summary: Summary dict per gene
    """

    use_llm_themes: bool = True  # Use LLM for theme extraction

    async def run(self, ctx: GraphRunContext[GeneState]) -> "InterpretAllGenes":
        _t0 = time.perf_counter()
        state = ctx.state
        try:
            from src.nodes.interpret_all_genes import InterpretAllGenes
    
            print(f"\n{'='*70}")
            print("NODE: Analyze Literature Findings")
            print("Extracting research trends and themes from collected papers")
            print(f"{'='*70}")
    
            genes = state.get_genes_found()
    
            current_year = datetime.now().year
            recent_cutoff = current_year - 2  # Papers from last 2 years
    
            print(f"\nAnalysing literature for {len(genes)} genes...")
            print(f"Recent papers cutoff: {recent_cutoff}-{current_year}")
            print()
    
            for gene in genes:
                print(f"  {gene}:")
    
                # Get papers from various sources
                top_papers_data = state.gene_top_papers.get(gene, {})
                top_papers = top_papers_data.get('top_papers', [])
                candidates_data = state.gene_literature_candidates.get(gene, {})
                candidate_papers = candidates_data.get('candidate_papers', [])
    
                # Use candidates if available, otherwise top papers
                all_papers = candidate_papers if candidate_papers else top_papers
    
                if not all_papers:
                    print(f"    No papers found, skipping analysis")
                    state.literature_findings_summary[gene] = {
                        'gene': gene,
                        'total_papers': 0,
                        'recent_papers_count': 0,
                        'publication_types': {},
                        'research_themes': [],
                        'top_papers': [],
                        'publication_timeline': {},
                        'analysis_status': 'no_papers'
                    }
                    continue
    
                # Analyse papers
                findings = await self._analyze_gene_literature(
                    gene=gene,
                    papers=all_papers,
                    top_papers=top_papers[:5],  # Top 5 for display
                    recent_cutoff=recent_cutoff,
                    user_query=state.user_query,
                    context=state.experiment_context,
                    state=state
                )
    
                state.literature_findings_summary[gene] = findings
    
                # Print summary
                print(f"    Total papers analysed: {findings['total_papers']}")
                print(f"    Recent papers (last 2 years): {findings['recent_papers_count']}")
                print(f"    Research themes identified: {len(findings['research_themes'])}")
                if findings['research_themes']:
                    print(f"      Top theme: {findings['research_themes'][0]}")
    
            # Summary
            print()
            print(f"{'='*70}")
            print("LITERATURE ANALYSIS COMPLETE")
            print(f"{'='*70}")
            print(f"Genes analysed: {len(genes)}")
    
            total_papers = sum(
                state.literature_findings_summary.get(g, {}).get('total_papers', 0)
                for g in genes
            )
            print(f"Total papers analysed: {total_papers}")
            print(f"{'='*70}\n")
    
            return InterpretAllGenes()

        finally:
            state.log_node_execution(self.__class__.__name__, round(time.perf_counter() - _t0, 3))

    async def _analyze_gene_literature(
        self,
        gene: str,
        papers: List[Dict[str, Any]],
        top_papers: List[Dict[str, Any]],
        recent_cutoff: int,
        user_query: str,
        context: Any,
        *,
        state=None
    ) -> Dict[str, Any]:
        """
        Analyse literature for a single gene.

        Args:
            gene: Gene symbol
            papers: All candidate papers
            top_papers: Top ranked papers for display
            recent_cutoff: Year cutoff for "recent" papers
            user_query: User's research question
            context: Experiment context
            state: Optional state object for token tracking

        Returns:
            Dictionary with literature analysis findings
        """
        # Basic counts
        total_papers = len(papers)

        # Publication timeline
        timeline = Counter()
        recent_count = 0
        for paper in papers:
            year = paper.get('year')
            if year:
                try:
                    year_int = int(year)
                    timeline[year_int] += 1
                    if year_int >= recent_cutoff:
                        recent_count += 1
                except (ValueError, TypeError):
                    pass

        # Publication types (if available)
        pub_types = Counter()
        for paper in papers:
            pub_type = paper.get('publication_type') or paper.get('pub_type')
            if pub_type:
                if isinstance(pub_type, list):
                    for pt in pub_type:
                        pub_types[pt] += 1
                else:
                    pub_types[pub_type] += 1

        # Infer publication types from titles/abstracts if not available
        if not pub_types:
            pub_types = self._infer_publication_types(papers)

        # Extract research themes
        if self.use_llm_themes and os.getenv('OPENAI_API_KEY'):
            themes = await self._extract_themes_with_llm(
                gene=gene,
                papers=papers[:30],  # Limit for LLM
                user_query=user_query,
                context=context,
                state=state,
                node_name=self.__class__.__name__
            )
        else:
            themes = self._extract_themes_simple(papers)

        # Format top papers for display
        formatted_top_papers = []
        for paper in top_papers[:5]:
            formatted_top_papers.append({
                'title': paper.get('title', 'Unknown title'),
                'year': paper.get('year', 'N/A'),
                'pmid': paper.get('pmid', 'N/A'),
                'journal': paper.get('journal', 'Unknown'),
                'relevance_score': paper.get('relevance_score', 0),
                'key_finding': paper.get('key_finding', '')
            })

        # Identify active research areas
        active_areas = self._identify_active_areas(papers, themes)

        return {
            'gene': gene,
            'total_papers': total_papers,
            'recent_papers_count': recent_count,
            'recent_years': f"{recent_cutoff}-{datetime.now().year}",
            'publication_types': dict(pub_types),
            'research_themes': themes,
            'top_papers': formatted_top_papers,
            'publication_timeline': dict(sorted(timeline.items())),
            'active_research_areas': active_areas,
            'analysis_status': 'complete'
        }

    def _infer_publication_types(self, papers: List[Dict[str, Any]]) -> Counter:
        """Infer publication types from titles and abstracts."""
        pub_types = Counter()

        review_keywords = ['review', 'meta-analysis', 'systematic review', 'overview']
        case_keywords = ['case report', 'case study', 'case series']
        clinical_keywords = ['clinical trial', 'randomized', 'cohort', 'prospective']

        for paper in papers:
            title = (paper.get('title') or '').lower()
            abstract = (paper.get('abstract') or '').lower()
            text = f"{title} {abstract}"

            if any(kw in text for kw in review_keywords):
                pub_types['Review'] += 1
            elif any(kw in text for kw in case_keywords):
                pub_types['Case Report'] += 1
            elif any(kw in text for kw in clinical_keywords):
                pub_types['Clinical Study'] += 1
            else:
                pub_types['Primary Research'] += 1

        return pub_types

    def _extract_themes_simple(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract research themes using simple keyword analysis."""
        # Collect all words from titles
        word_counts = Counter()

        stopwords = {
            'the', 'a', 'an', 'in', 'of', 'and', 'to', 'for', 'with', 'on',
            'is', 'are', 'by', 'from', 'that', 'this', 'be', 'as', 'at',
            'or', 'its', 'it', 'via', 'through', 'between', 'into', 'gene',
            'genes', 'protein', 'proteins', 'study', 'analysis', 'role'
        }

        for paper in papers:
            title = paper.get('title', '')
            # Simple word extraction
            words = title.lower().split()
            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum())
                if len(word) > 3 and word not in stopwords:
                    word_counts[word] += 1

        # Get top themes
        top_words = word_counts.most_common(10)
        themes = [word for word, count in top_words if count > 1]

        return themes[:5]  # Return top 5 themes

    async def _extract_themes_with_llm(
        self,
        gene: str,
        papers: List[Dict[str, Any]],
        user_query: str,
        context: Any,
        *,
        state=None,
        node_name: str = "AnalyzeLiteratureFindings"
    ) -> List[str]:
        """Extract research themes using LLM analysis.

        Args:
            gene: Gene symbol
            papers: List of paper dictionaries
            user_query: User's research question
            context: Experiment context
            state: Optional state object for token tracking
            node_name: Name of the calling node for token tracking

        Returns:
            List of research theme strings
        """
        from openai import OpenAI
        from src.config import get_active_model

        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Format papers for prompt
        papers_text = ""
        for i, paper in enumerate(papers[:20], 1):
            title = paper.get('title', 'No title')
            year = paper.get('year', 'N/A')
            papers_text += f"{i}. {title} ({year})\n"

        # Build context string
        context_parts = []
        if hasattr(context, 'cell_type') and context.cell_type:
            context_parts.append(f"Cell type: {context.cell_type}")
        if hasattr(context, 'tissue') and context.tissue:
            context_parts.append(f"Tissue: {context.tissue}")
        context_str = " | ".join(context_parts) if context_parts else "General"

        prompt = f"""Analyse these research papers about {gene} and identify the main research themes.

RESEARCH CONTEXT: {user_query}
EXPERIMENTAL CONTEXT: {context_str}

PAPER TITLES:
{papers_text}

TASK: Identify 3-5 distinct research themes from these papers. Each theme should be:
- A specific biological topic or research direction
- Relevant to understanding {gene}'s function
- Concise (2-5 words)

Return JSON:
{{
  "themes": ["theme1", "theme2", "theme3"]
}}

Examples of good themes:
- "cancer progression mechanisms"
- "immune cell regulation"
- "metabolic pathway control"
- "therapeutic target development"
- "protein-protein interactions"

Return ONLY valid JSON."""

        try:
            response = client.chat.completions.create(
                model=get_active_model(),
                messages=[
                    {"role": "system", "content": "You are a biomedical research analyst. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            # Track token usage
            usage = getattr(response, "usage", None)
            _accumulate_tokens(state, node_name, usage)

            result = json.loads(response.choices[0].message.content)
            themes = result.get('themes', [])

            return themes[:5]

        except Exception as e:
            logger.error(f"LLM theme extraction failed for {gene}: {e}")
            # Fallback to simple extraction
            return self._extract_themes_simple(papers)

    def _identify_active_areas(
        self,
        papers: List[Dict[str, Any]],
        themes: List[str]
    ) -> List[str]:
        """Identify currently active research areas based on recent papers."""
        current_year = datetime.now().year
        recent_cutoff = current_year - 3

        # Filter to recent papers
        recent_papers = []
        for paper in papers:
            try:
                year = int(paper.get('year', 0))
                if year >= recent_cutoff:
                    recent_papers.append(paper)
            except (ValueError, TypeError):
                pass

        if not recent_papers:
            return themes[:3] if themes else []

        # Extract keywords from recent papers
        recent_words = Counter()
        for paper in recent_papers:
            title = (paper.get('title') or '').lower()
            words = title.split()
            for word in words:
                word = ''.join(c for c in word if c.isalnum())
                if len(word) > 4:
                    recent_words[word] += 1

        # Return most common recent topics
        active = [word for word, _ in recent_words.most_common(5)]
        return active if active else themes[:3]
