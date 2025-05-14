import asyncio
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor, ReadableSpan
from opentelemetry.trace import NoOpTracerProvider
from ragmetrics.api import ragmetrics_client
import json

class RagMetricsSpanProcessor(SpanProcessor):
    def __init__(self, display_name="OpenAI Agent"):
        super().__init__()
        self.event_counter = 0
        self.display_name = display_name

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        print("[DEBUG] RagMetricsSpanProcessor.on_start called", span, parent_context)
        # No-op on start, parent_context supported in newer SDK
        return None

    def on_end(self, span: ReadableSpan) -> None:
        print("[DEBUG] RagMetricsSpanProcessor.on_end called", span)
        # Convert span to JSON and log immediately as its own trace
        span_json = span_to_json(span)
        import json
        print("[DEBUG] span_to_json output:")
        print(json.dumps(span_json, indent=2))

    async def on_handoff(self, context, agent, source) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {getattr(source, 'name', source)} handed off to {getattr(agent, 'name', agent)}"
        )

    async def on_tool_start(self, context, agent, tool) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {getattr(agent, 'name', agent)} started tool {getattr(tool, 'name', tool)}"
        )

    async def on_tool_end(self, context, agent, tool, result: str) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {getattr(agent, 'name', agent)} ended tool {getattr(tool, 'name', tool)} with result {result}"
        )
        ragmetrics_client._log_trace(
            input_messages=span.attributes.get("input"),
            response=span.attributes.get("output"),
            metadata_llm=span_json
        )



def span_to_json(span):
    print("[DEBUG] span_to_json called", span)
    span_dict = {
        "name": span.name,
        "context": {
            "trace_id": format(span.context.trace_id, "032x"),
            "span_id": format(span.context.span_id, "016x"),
            "trace_flags": int(span.context.trace_flags),
        },
        "parent": (
            {
                "trace_id": format(span.parent.trace_id, "032x"),
                "span_id": format(span.parent.span_id, "016x"),
            }
            if span.parent
            else None
        ),
        "attributes": dict(span.attributes),
        "events": [
            {
                "name": event.name,
                "attributes": dict(event.attributes),
                "timestamp": event.timestamp,
            }
            for event in span.events
        ],
        "links": [
            {
                "context": {
                    "trace_id": format(link.context.trace_id, "032x"),
                    "span_id": format(link.context.span_id, "016x"),
                },
                "attributes": dict(link.attributes),
            }
            for link in span.links
        ],
        "kind": str(span.kind),
        "status": {
            "status_code": str(span.status.status_code),
            "description": span.status.description,
        },
        "start_time": span.start_time,
        "end_time": span.end_time,
    }
    return span_dict



def monitor_openai_agents(target):
    print("[DEBUG] monitor_openai_agents called for target:", target)
    """
    Configure OpenTelemetry to push spans to RagMetrics and wrap the target's run method.
    """
    # Setup tracer provider with our span processor
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    provider.add_span_processor(RagMetricsSpanProcessor())
    tracer = trace.get_tracer("ragmetrics.openai_agents")
    orig_run = getattr(target, 'run')

    # Patch with an async wrapper for coroutine functions, sync otherwise
    if asyncio.iscoroutinefunction(orig_run):
        async def run_monitored(agent, *args, **kwargs):
            print(f"[DEBUG] run_monitored (async) called for agent: {getattr(agent, 'name', agent)} args: {args} kwargs: {kwargs}")
            with tracer.start_as_current_span(f"agent_{agent.name}"):
                result = await orig_run(agent, *args, **kwargs)
                print(f"[DEBUG] run_monitored (async) result: {result}")
                return result
    else:
        def run_monitored(agent, *args, **kwargs):
            print(f"[DEBUG] run_monitored (sync) called for agent: {getattr(agent, 'name', agent)} args: {args} kwargs: {kwargs}")
            with tracer.start_as_current_span(f"agent_{agent.name}"):
                result = orig_run(agent, *args, **kwargs)
                print(f"[DEBUG] run_monitored (sync) result: {result}")
                return result

    setattr(target, 'run', run_monitored)
    print("[DEBUG] monitor_openai_agents finished patching run method.")
    return target

# Deprecated alias kept for compatibility
setup_tracing = monitor_openai_agents
