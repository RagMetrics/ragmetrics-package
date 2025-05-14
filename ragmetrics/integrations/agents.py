"""
Integration with OpenAI Agents SDK for RagMetrics.
"""

import ragmetrics
from agents import tracing
from agents.tracing.processor_interface import TracingProcessor
from agents import set_trace_processors, Runner, Agent, function_tool
import os
import uuid
import json
import time
import logging
from typing import Optional, Dict, Any, Union, List
from ragmetrics.api import ragmetrics_client, default_callback

class RagMetricsTracingProcessor(TracingProcessor):
    """A TracingProcessor that sends agent spans to RagMetrics."""

    def __init__(self):
        # Initialize RagMetrics (assumes ragmetrics.login was already called)
        self.trace_id = None
        self.spans = []
        self.conversation_id = None

    def on_trace_start(self, trace):
        """Called when a new agent trace starts."""
        try:
            self.trace_id = str(trace.trace_id)
            print(f"Starting agent trace {self.trace_id}")
            
            # Create a new conversation for this trace
            try:
                self.conversation_id = ragmetrics_client.new_conversation()
                print(f"Created new conversation for trace {self.trace_id}")
            except Exception as e:
                print(f"Could not create new conversation: {e}")
            
            # Log trace start using _log_trace_safely with the trace object
            self._log_trace_safely(trace=trace)
        except Exception as e:
            print(f"Error in on_trace_start: {e}")

    def on_span_start(self, span):
        # Just collect the start time
        pass

    def on_span_end(self, span):
        """Called when a span ends (operation completes)."""
        try:
            # Get basic span information for logging
            if hasattr(span, "span_data"):
                span_data = span.span_data
                span_type = getattr(span_data, "type", "unknown")
                span_id = getattr(span, "span_id", "unknown")
                print(f"Span ended: {span_type} (ID: {span_id})")
                
                # Extract input and output for our spans collection
                input_data = "[No input data]"
                if hasattr(span_data, "input"):
                    input_data = str(span_data.input)
                    
                output_data = "[No output data]"
                if hasattr(span_data, "output"):
                    output_data = str(span_data.output)
                
                # Store span info for our summary later
                self.spans.append({
                    "type": span_type,
                    "span_id": span_id,
                    "input": input_data,
                    "output": output_data,
                    "timestamp": time.time()
                })
            else:
                print(f"Span ended: (ID: {getattr(span, 'span_id', 'unknown')})")
            
            # Log the span using our simplified method
            self._log_trace_safely(span=span)

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
            
        # Basic span info
        span_data = {
            "span_id": str(getattr(span, "span_id", "unknown")),
            "event_type": "span"
        }
        
        # Get span_data if available
        if hasattr(span, "span_data"):
            sd = span.span_data
            span_data["type"] = getattr(sd, "type", "unknown")
            
            # Add any available properties from span_data
            for attr in ["status", "start_time", "end_time", "parent_id"]:
                if hasattr(sd, attr):
                    span_data[attr] = str(getattr(sd, attr))
        
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
        
    def extract_messages_from_span(self, span):
        """Extract input messages and response from a span object.
        
        Args:
            span: The span object to extract from
            
        Returns:
            tuple: (input_messages, response)
        """
        # Default values
        input_messages = [{"role": "user", "content": "[No input data]"}]
        response = {"role": "assistant", "content": "[No response data]"}
        
        if hasattr(span, "span_data"):
            sd = span.span_data
            span_type = getattr(sd, "type", "unknown")
            
            # Extract input if available
            if hasattr(sd, "input"):
                input_data = str(sd.input)
                input_messages = [{"role": "user", "content": f"[Agent {span_type}]: {input_data[:300]}" if len(input_data) > 300 else input_data}]
            
            # Extract output if available
            if hasattr(sd, "output"):
                output_data = str(sd.output)
                response = {"role": "assistant", "content": f"Result: {output_data[:300]}" if len(output_data) > 300 else output_data}
        
        return input_messages, response

    def _log_trace_safely(self, trace=None, span=None):
        """Log a trace or span to RagMetrics, extracting relevant data.
        
        This method accepts either a trace or span object (or both), extracts the relevant
        information, and logs it to RagMetrics. If both trace and span are None, it's a no-op.
        
        Args:
            trace: Optional trace object to log
            span: Optional span object to log
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        # If both trace and span are None, it's a no-op
        if trace is None and span is None:
            return True
            
        try:
            # Determine if we're logging a trace or span
            is_trace = trace is not None
            target = trace if is_trace else span
            
            # Extract input_messages and response
            if is_trace:
                input_messages, response = self.extract_messages_from_trace(trace)
                json_data = self.serialize_trace_to_json(trace)
            else:
                input_messages, response = self.extract_messages_from_span(span)
                json_data = self.serialize_span_to_json(span)
            
            # Create metadata
            metadata = {
                "agent_sdk": True,
                "trace_id": self.trace_id,
            }
            
            # Add the serialized JSON data to metadata
            metadata.update(json_data)
            
            # Use the default_callback to process inputs and outputs
            callback_result = default_callback(input_messages, response)
                
            # Call _log_trace with all required parameters
            ragmetrics_client._log_trace(
                input_messages=input_messages,
                response=response,
                metadata_llm=metadata,
                callback_result=callback_result,
                duration=0.0
            )
            return True
        except Exception as e:
            print(f"Error during _log_trace: {e}")
            return False

    def on_trace_end(self, trace):
        """Called when the agent trace completes."""
        try:
            print(f"Agent trace {self.trace_id} completed with {len(self.spans)} spans")
            
            # Add span count to the trace object
            if not hasattr(trace, "span_count"):
                setattr(trace, "span_count", len(self.spans))
            
            # Log trace completion
            self._log_trace_safely(trace=trace)
                
        except Exception as e:
            print(f"Error in on_trace_end: {e}")

    def shutdown(self):
        """Called when the trace processor is shutting down."""
        print("RagMetrics agent tracing shutdown")

    def force_flush(self):
        print("RagMetrics agent tracing force flush")

def monitor_agents(openai_client=None, ragmetrics_key=None):
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
        # Log in to RagMetrics (user should have set RAGMETRICS_API_KEY or passed key)
        #if ragmetrics_key:
        #    ragmetrics.login(key=ragmetrics_key)
        #else:
        #    ragmetrics.login()  # assumes ENV var or prior login

        # Determine if we have an async or sync client
        client_type = str(type(openai_client).__name__)
        is_async_client = "Async" in client_type
        print(f"Monitoring {client_type} client for RagMetrics")

        # Set up the OpenAI client
        if openai_client is not None:
            # For agents SDK, we need to set the default client
            try:
                from agents import set_default_openai_client
                set_default_openai_client(openai_client)
                print(f"Set {client_type} as default for agents SDK")
            except ImportError:
                print("Could not set default OpenAI client for agents SDK.")
                
            # For a sync client, we can also monitor it with ragmetrics
            if not is_async_client:
                try:
                    ragmetrics.monitor(openai_client)
                    print("OpenAI client is being monitored by RagMetrics")
                except Exception as e:
                    print(f"Could not monitor OpenAI client: {e}")
        
        # Install our trace processor into the Agents SDK
        processor = RagMetricsTracingProcessor()
        set_trace_processors([processor])
        print("RagMetrics monitoring enabled for OpenAI Agents SDK")
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
