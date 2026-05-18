"""
Python 3.14 Compatibility Patch for pydantic_graph

This module patches pydantic_graph to work with Python 3.13+'s stricter
forward reference evaluation.

IMPORTANT: This module must NOT import pydantic_graph at the top level,
as the patch must be applied BEFORE pydantic_graph is imported.
"""

import importlib
import pkgutil
import sys
from typing import get_type_hints as original_get_type_hints


# Cache for loaded node classes
_node_classes_cache: dict[str, type] | None = None


def _load_all_node_classes() -> dict[str, type]:
    """Dynamically import every node module and collect BaseNode subclasses."""
    global _node_classes_cache

    if _node_classes_cache is not None:
        return _node_classes_cache

    # Import pydantic_graph here (lazily) to avoid import-time issues
    from pydantic_graph.nodes import BaseNode, End

    discovered: dict[str, type] = {
        'End': End,  # Always include End from pydantic_graph
    }

    try:
        import src.nodes
    except ImportError:
        _node_classes_cache = discovered
        return discovered

    node_pkg = src.nodes  # type: ignore[attr-defined]

    for finder, module_name, is_pkg in pkgutil.iter_modules(node_pkg.__path__):
        full_module = f"{node_pkg.__name__}.{module_name}"
        try:
            module = importlib.import_module(full_module)
        except Exception:
            continue

        for attr_name, attr_value in vars(module).items():
            if isinstance(attr_value, type) and issubclass(attr_value, BaseNode):
                discovered[attr_name] = attr_value

    _node_classes_cache = discovered
    return discovered


def patched_get_type_hints(obj, globalns=None, localns=None, *, include_extras=False, format=None):
    """
    Patched version of get_type_hints that handles forward references more gracefully.

    In Python 3.13+, get_type_hints became stricter about resolving forward references.
    This patch attempts resolution but falls back gracefully on failure, building a
    localns with all available node classes only when needed.
    """
    # Try to build a comprehensive localns with all node classes
    if globalns is None:
        if hasattr(obj, '__globals__'):
            globalns = obj.__globals__
        else:
            globalns = {}

    if localns is None:
        localns = {}

    # First try without importing - this avoids circular imports
    try:
        return original_get_type_hints(obj, globalns, localns, include_extras=include_extras)
    except (NameError, AttributeError) as e:
        error_str = str(e)

        # Check if the error is about a node class or if we're in a node module
        is_node_module = hasattr(obj, '__module__') and obj.__module__ and 'src.nodes' in str(obj.__module__)
        is_node_error = any(name in error_str for name in [
            'FetchAllGeneData', 'ExtractGenesStateful', 'AnalyzeNetworkOverlap',
            'AnalyzeGoComparison', 'PopulateReportMetadata', 'BuildLiteratureQueryPlan',
            'FetchAdaptiveLiterature', 'RankPapersByRelevance', 'AnalyzeLiteratureFindings',
            'InterpretAllGenes', 'InterpretNetworkOverlap', 'InterpretGoPatterns',
            'GenerateCrossGeneSynthesis', 'GenerateGeneSummaries',
            'FinalSummary', 'End'
        ])

        if is_node_module or is_node_error:
            node_classes = _load_all_node_classes()
            if node_classes:
                # Update both globalns and localns for thorough resolution
                merged_localns = {**node_classes, **localns}
                try:
                    return original_get_type_hints(obj, globalns, merged_localns, include_extras=include_extras)
                except Exception:
                    pass

        # If we still can't resolve, re-raise
        raise e

def apply_python314_patch():
    """
    Apply the compatibility patch for Python 3.13+.

    Call this before importing pydantic_graph.
    """
    if sys.version_info >= (3, 13):
        import typing
        typing.get_type_hints = patched_get_type_hints
        print(f"✓ Applied Python {sys.version_info.major}.{sys.version_info.minor} compatibility patch for pydantic_graph")


def patch_pydantic_graph_modules():
    """
    Patch pydantic_graph modules after they've been imported.

    Call this AFTER importing pydantic_graph to ensure their cached
    get_type_hints references are also patched.
    """
    if sys.version_info >= (3, 13):
        try:
            import pydantic_graph.nodes
            pydantic_graph.nodes.get_type_hints = patched_get_type_hints
        except (ImportError, AttributeError):
            pass

        try:
            import pydantic_graph.graph
            pydantic_graph.graph.get_type_hints = patched_get_type_hints
        except (ImportError, AttributeError):
            pass

        _patch_pydantic_graph_node_serialization()


def _patch_pydantic_graph_node_serialization():
    """Patch persistence serialization/deserialization for nodes with fields.

    The issue: nodes with dataclass fields (like RankPapersByRelevance(top_n=10))
    serialize as plain dicts {'top_n': 10} without a type discriminator.
    On deserialization, pydantic_graph can't determine which node type to use.

    This patch ensures:
    1. Serialization always adds 'node_id' to the dict
    2. Deserialization can find 'node_id' in the dict
    """
    try:
        from pydantic_graph.persistence import _utils as persistence_utils
    except (ImportError, AttributeError):
        return

    # Store original methods for the wrap serializer
    _original_node_serializer = persistence_utils.CustomNodeSchema._node_serializer

    def _safe_node_discriminator(node_data):
        """Extract node_id for discriminated union deserialization.

        Returns node_id or None if not found. If we receive an empty dict,
        this is a serialization bug that should have been caught earlier.
        """
        if isinstance(node_data, dict):
            node_id = node_data.get('node_id')
            # Log warning if we got an empty dict without node_id
            if node_id is None and not node_data:
                # This indicates a bug in node serialization - the serializer
                # should have added node_id. But we can't fix it here, so we
                # let it fail naturally and LangGraph will repair the snapshot.
                pass
            return node_id
        if hasattr(node_data, 'get_node_id'):
            try:
                return node_data.get_node_id()
            except Exception:
                return None
        return None

    def _safe_node_serializer(node, handler):
        """Serialize node ensuring node_id is always included."""
        # Handle None or invalid nodes early
        if node is None:
            return {'node_id': 'None'}

        try:
            # First try the original serializer
            node_dict = handler(node)
        except Exception:
            # Fallback: manually serialize the node
            node_dict = {}
            if hasattr(node, '__dataclass_fields__'):
                for field_name in node.__dataclass_fields__:
                    if not field_name.startswith('_'):
                        try:
                            node_dict[field_name] = getattr(node, field_name, None)
                        except Exception:
                            continue

        # Ensure node_dict is a dict
        if not isinstance(node_dict, dict):
            node_dict = {}

        # CRITICAL: Ensure node_id is ALWAYS present, no matter what
        if 'node_id' not in node_dict:
            # Try multiple ways to get the node ID
            node_id = None

            # Method 1: get_node_id() method
            if hasattr(node, 'get_node_id'):
                try:
                    node_id = node.get_node_id()
                except Exception:
                    pass

            # Method 2: __class__.__name__
            if node_id is None and hasattr(node, '__class__'):
                try:
                    node_id = node.__class__.__name__
                except Exception:
                    pass

            # Method 3: type(node).__name__ as final fallback
            if node_id is None:
                try:
                    node_id = type(node).__name__
                except Exception:
                    node_id = 'UnknownNode'

            node_dict['node_id'] = node_id

        return node_dict

    persistence_utils.CustomNodeSchema._node_discriminator = staticmethod(_safe_node_discriminator)
    persistence_utils.CustomNodeSchema._node_serializer = staticmethod(_safe_node_serializer)
