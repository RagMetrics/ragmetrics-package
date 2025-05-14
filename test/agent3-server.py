# ragmetrics_agents.py

import ragmetrics
from agents import tracing
from agents.tracing.processor_interface import TracingProcessor
from agents import set_trace_processors, Runner, Agent, function_tool

class RagMetricsTracingProcessor(TracingProcessor):
    """A TracingProcessor that sends agent spans to RagMetrics."""

    def __init__(self):
        # Initialize RagMetrics (assumes ragmetrics.login was already called)
        self.client = ragmetrics.client  # ragmetrics.client is the API client

    def on_trace_start(self, trace):
        # Optionally mark the start of a new agent run in RagMetrics
        try:
            # For example, start a RagMetrics "trace" or run if API supports it
            self.run_id = str(trace.trace_id)
            self.client.create_run(
                run_id=self.run_id,
                name=trace.name or "AgentRun",
                inputs={}, outputs={}, extra={"agent_trace": True}
            )
        except Exception:
            pass

    def on_span_start(self, span):
        # No-op or record start time if needed
        pass

    def on_span_end(self, span):
        # When any span (LLM gen or tool) ends, log its I/O to RagMetrics
        span_data = span.span_data
        # Decide how to record: e.g., log function calls or agent handoff as RagMetrics interactions
        try:
            inputs = {}
            outputs = {}
            if hasattr(span_data, "input"):
                inputs = {"input": str(span_data.input)}
            if hasattr(span_data, "output"):
                outputs = {"output": str(span_data.output)}
            # Create an entry in RagMetrics (using run_id from trace_start)
            self.client.update_run(
                run_id=self.run_id,
                inputs=inputs,
                outputs=outputs,
                extra={"span_type": span_data.type}
            )
        except Exception:
            pass

    def on_trace_end(self, trace):
        # Optionally finalize the run in RagMetrics
        try:
            self.client.update_run(run_id=self.run_id, end_time=True)
        except Exception:
            pass

    def shutdown(self):
        # Flush any queued data to RagMetrics
        try:
            ragmetrics.client.flush()
        except Exception:
            pass

    def force_flush(self):
        ragmetrics.client.flush()

def monitor_agents(openai_client=None, ragmetrics_key=None):
    """Set up RagMetrics tracing for OpenAI Agents SDK.

    Call this once before running agents.  If openai_client is provided, we
    monitor it with ragmetrics.monitor() and set it as default for Agents.
    """
    # Log in to RagMetrics (user should have set RAGMETRICS_API_KEY or passed key)
    if ragmetrics_key:
        ragmetrics.login(key=ragmetrics_key)
    else:
        ragmetrics.login()  # assumes ENV var or prior login

    # Optionally instrument the OpenAI client so all LLM calls are logged
    if openai_client is not None:
        ragmetrics.monitor(openai_client)
        try:
            from agents import set_default_openai_client
            set_default_openai_client(openai_client)
        except ImportError:
            pass

    # Install our trace processor into the Agents SDK
    set_trace_processors([RagMetricsTracingProcessor()])
