"""OpenTelemetry tracing + per-request LLM cost. ADK emits its own spans into the same provider."""
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# USD per 1M tokens for your LLM_MODEL; set from the provider's price page (free tiers are 0)
PRICE_IN = float(os.environ.get("LLM_PRICE_IN", "0"))
PRICE_OUT = float(os.environ.get("LLM_PRICE_OUT", "0"))


def cost_usd(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6


def setup_tracing(path: str = "traces.jsonl"):
    """Appends every span as one JSON line. shortcut: local file, swap for a Cloud Trace exporter on GCP."""
    provider = TracerProvider()
    out = open(path, "a", encoding="utf-8")
    provider.add_span_processor(SimpleSpanProcessor(
        ConsoleSpanExporter(out=out, formatter=lambda span: span.to_json(indent=None) + "\n")))
    trace.set_tracer_provider(provider)
