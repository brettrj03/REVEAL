"""
Phoenix observability integration for the REVEAL pipeline.

Usage:
    from src.utils.phoenix_tracing import setup_phoenix, get_tracer, node_span

    # Once, at pipeline startup:
    setup_phoenix()

    # Inside any node, to add custom attributes to a span:
    with node_span("RankPapersByRelevance", gene=gene, n_candidates=30) as span:
        ...
        span.set_attribute("n_selected", 10)
        span.set_attribute("top_paper_score", 0.92)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Whether Phoenix was successfully set up this process
_phoenix_active = False


def setup_phoenix(project_name: str = "reveal-pipeline") -> bool:
    """
    Start Phoenix (if not already running) and register the OpenAI auto-instrumentor.

    Returns True if Phoenix is active, False if setup failed (non-fatal).

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _phoenix_active
    if _phoenix_active:
        return True

    # Opt-out via env var
    if os.getenv("PHOENIX_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        logger.info("Phoenix tracing disabled via PHOENIX_DISABLED env var.")
        return False

    try:
        import phoenix as px
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # Launch the embedded Phoenix server (opens http://localhost:6006)
        px.launch_app()

        # Set up OTel TracerProvider pointing at Phoenix's OTLP endpoint
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Auto-instrument all OpenAI calls (captures prompt, response, tokens, latency)
        OpenAIInstrumentor().instrument(tracer_provider=provider)

        _phoenix_active = True
        logger.info(
            "Phoenix tracing active. Dashboard: http://localhost:6006  "
            f"Project: {project_name}"
        )
        return True

    except ImportError:
        logger.warning(
            "Phoenix packages not installed. Run: "
            "pip install arize-phoenix openinference-instrumentation-openai"
        )
        return False
    except Exception as exc:
        logger.warning(f"Phoenix setup failed (non-fatal): {exc}")
        return False


def get_tracer(name: str = "reveal"):
    """Return an OTel tracer. Safe to call even if Phoenix is not active."""
    if not _phoenix_active:
        return _NoOpTracer()
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:
        return _NoOpTracer()


@contextmanager
def node_span(node_name: str, **attrs: Any):
    """
    Context manager that wraps a pipeline node in a named span.

    Example:
        with node_span("ValidateGeneSummaries", gene="MED12", iteration=2) as span:
            ...
            span.set_attribute("accuracy_score", 87.5)
            span.set_attribute("passed", False)

    If Phoenix is not active, this is a no-op context manager.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(node_name) as span:
        for k, v in attrs.items():
            try:
                span.set_attribute(k, v)
            except Exception:
                pass
        yield span


# ── No-op fallback (used when Phoenix is not installed / disabled) ──────────


class _NoOpSpan:
    def set_attribute(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoOpTracer:
    def start_as_current_span(self, *args, **kwargs):
        return _NoOpSpan()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
