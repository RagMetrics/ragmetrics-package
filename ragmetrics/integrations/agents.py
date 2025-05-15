"""
Integration with OpenAI Agents SDK for RagMetrics.
"""

import ragmetrics
from agents.tracing.processor_interface import TracingProcessor
from agents import set_trace_processors
from ragmetrics.api import ragmetrics_client, default_callback

class RagMetricsTracingProcessor(TracingProcessor):
    """A TracingProcessor that sends agent spans to RagMetrics."""

    def __init__(self):
        # Initialize RagMetrics (assumes ragmetrics.login was already called)
        self.trace_id = None
        self.spans = []
        self.conversation_id = None

    def on_trace_start(self, trace):
        try:
            self.conversation_id = ragmetrics_client.new_conversation(trace.trace_id)
        except Exception as e:
            print(f"Error in on_trace_start: {e}")


    def on_span_start(self, span):
        # Just collect the start time
        pass

    def on_span_end(self, span):
        """Called when a span ends (operation completes)."""
        try:
            # Extract input_messages and response from span
            input_messages, response = self.extract_raw_io_from_span(span)

            try:
                span_type = span.span_data.type
            except:
                span_type = "unknown"
            
            # Serialize span to JSON
            json_data = self.serialize_span_to_json(span)
            
            # Create metadata
            metadata = {
                "agent_sdk": True,
                "trace_id": self.trace_id,
                "span_type": span_type
            }
            metadata.update(json_data)
            
            callback_result = default_callback(input_messages, response)
            
            #Log to RagMetrics
            try:
                ragmetrics_client._log_trace(
                    input_messages=input_messages,
                    response=response,
                    metadata_llm=metadata,
                    callback_result=callback_result,
                    conversation_id=self.conversation_id,
                    duration=0.0
                )
            except Exception as e:
                print(f"Error logging span: {e}")

        except Exception as e:
            print(f"Error in on_span_end: {e}")

    def serialize_trace_to_json(self, trace):
        """Convert a trace object to a JSON-serializable dictionary.
        
        Args:
            trace: The trace object to serialize
            
        Returns:
            dict: A JSON-serializable dictionary representing the trace
        """
        if trace is None:
            return {}
            
        trace_data = {
            "trace_id": str(getattr(trace, "trace_id", "unknown")),
            "name": str(getattr(trace, "name", "unknown")),
            "event_type": "trace"
        }
        
        # Add any other trace attributes that might be useful
        for attr in ["status", "start_time", "end_time", "parent_id"]:
            if hasattr(trace, attr):
                trace_data[attr] = str(getattr(trace, attr))
                
        return trace_data
    
    def serialize_span_to_json(self, span):
        """Convert a span object to a JSON-serializable dictionary.
        
        Args:
            span: The span object to serialize
            
        Returns:
            dict: A JSON-serializable dictionary representing the span
        """
        if span is None:
            return {}

        # Get all span attributes from __dict__
        span_attrs = {}
        for k, v in span.__dict__.items():
            if not k.startswith("_"):
                try:
                    # Try to convert to string, but handle potential serialization issues
                    span_attrs[k] = str(v)
                except:
                    span_attrs[k] = repr(v)
        
        # Create base span data
        span_data = {
            "span_id": str(getattr(span, "span_id", "unknown")),
            "event_type": "span"
        }
        
        # Add span attributes to the data
        span_data.update(span_attrs)
        
        # Get span_data attributes if available
        if hasattr(span, "span_data"):
            try:
                sd = span.span_data
                span_data["type"] = getattr(sd, "type", "unknown")
                
                # Add span_data attributes
                sd_attrs = {}
                for k, v in sd.__dict__.items():
                    if not k.startswith("_"):
                        try:
                            # Prefix with sd_ to avoid conflicts with span attributes
                            sd_attrs[f"sd_{k}"] = str(v)
                        except:
                            sd_attrs[f"sd_{k}"] = repr(v)
                            
                span_data.update(sd_attrs)
            except Exception as e:
                span_data["sd_error"] = str(e)
        
        return span_data

    def extract_messages_from_trace(self, trace):
        """Extract input messages and response from a trace object.
        
        Args:
            trace: The trace object to extract from
            
        Returns:
            tuple: (input_messages, response)
        """
        # Default values
        input_messages = [{"role": "system", "content": f"Trace: {getattr(trace, 'name', 'Agent trace')}"}]
        response = {"role": "system", "content": f"Trace ID: {getattr(trace, 'trace_id', 'unknown')}"}  
        
        return input_messages, response
        
    def extract_raw_io_from_span(self, span):
        """Extract input messages and response from a span object.
        
        Args:
            span: The span object to extract from
            
        Returns:
            tuple: (input_messages, response)
        """
        # Default values
        try:
            raw_input = span.span_data.input
        except:
            raw_input = None
        
        try:
            raw_output = span.span_data.response
        except:
            raw_output = None
        
        return raw_input, raw_output



    def on_trace_end(self, trace):
        """Called when the agent trace completes."""
        pass

    def shutdown(self):
        """Called when the trace processor is shutting down."""
        pass

    def force_flush(self):
        pass

def monitor_agents(openai_client=None):
    """Set up RagMetrics tracing for OpenAI Agents SDK.

    Call this once before running agents. If openai_client is provided, we
    monitor it with ragmetrics.monitor() and set it as default for Agents.
    
    The OpenAI client can be either a synchronous (OpenAI) or asynchronous 
    (AsyncOpenAI) client. For use with the agents SDK, the async client is
    recommended.
    
    Args:
        openai_client: Optional OpenAI or AsyncOpenAI client instance to set as default
        ragmetrics_key: Optional RagMetrics API key for authentication
        
    Returns:
        The configured tracing processor or None if setup failed
    """
    try:
        # Determine if we have an async or sync client
        client_type = str(type(openai_client).__name__)
        is_async_client = "Async" in client_type

        # Set up the OpenAI client
        if openai_client is not None:
            # For agents SDK, we need to set the default client
            try:
                from agents import set_default_openai_client
                set_default_openai_client(openai_client)
            except ImportError:
                print("Please install OpenAI Agents SDK with pip install openai-agents")
                
            # For a sync client, we can also monitor it with ragmetrics
            if not is_async_client:
                try:
                    ragmetrics.monitor(openai_client)
                except Exception as e:
                    print(f"Could not monitor OpenAI client: {e}")
        
        # Install our trace processor into the Agents SDK
        processor = RagMetricsTracingProcessor()
        set_trace_processors([processor])
        return processor
    except Exception as e:
        print(f"Error setting up RagMetrics monitoring for agents: {e}")
        # Provide a fallback processor that does nothing
        class NoOpProcessor(TracingProcessor):
            def on_trace_start(self, trace): pass
            def on_span_start(self, span): pass
            def on_span_end(self, span): pass
            def on_trace_end(self, trace): pass
            def shutdown(self): pass
            def force_flush(self): pass
        
        set_trace_processors([NoOpProcessor()])
        print("Using no-op processor due to error")
        return None
