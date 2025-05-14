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
            
            # Store basic information about the trace
            trace_name = getattr(trace, "name", "AgentRun")
            
            # Create messages and response
            input_messages = [{"role": "system", "content": f"Agent trace started: {trace_name}"}]
            response = {"role": "system", "content": f"Trace ID: {self.trace_id}"}
            
            # Add metadata
            metadata = {
                "agent_trace": True,
                "trace_id": self.trace_id,
                "event": "trace_start",
                "name": trace_name
            }
            
            # Log trace start using _log_trace
            self._log_trace_safely(input_messages, response, metadata)
        except Exception as e:
            print(f"Error in on_trace_start: {e}")

    def on_span_start(self, span):
        # Just collect the start time
        pass

    def on_span_end(self, span):
        """Called when a span ends (operation completes)."""
        try:
            # Get span information
            span_data = span.span_data
            span_type = getattr(span_data, "type", "unknown")
            span_id = getattr(span, "span_id", "unknown")
            
            print(f"Span ended: {span_type} (ID: {span_id})")
            
            # Extract input and output
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
            
            # Format input and output for RagMetrics in the expected format
            input_messages = [{"role": "user", "content": f"[Agent {span_type}]: {input_data[:300]}" if len(input_data) > 300 else input_data}]  
            response = {"role": "assistant", "content": f"Result: {output_data[:300]}" if len(output_data) > 300 else output_data}
            
            # Add metadata
            metadata = {
                "agent_span": True,
                "span_type": span_type,
                "span_id": str(span_id),
                "trace_id": self.trace_id
            }
            
            # Log the span directly using _log_trace
            try:
                self._log_trace_safely(input_messages, response, metadata)
            except Exception as e:
                print(f"Error during _log_trace: {e}")

        except Exception as e:
            print(f"Error in on_span_end: {e}")

    def _log_trace_safely(self, input_messages, response, metadata=None):
        """Call _log_trace with proper handling of all parameters to avoid NoneType errors."""
        try:
            # Ensure input_messages is a valid list
            if not isinstance(input_messages, list) or not input_messages:
                input_messages = [{"role": "system", "content": "[No input data]"}]
            
            # Ensure response is a dictionary with role and content
            if not isinstance(response, dict) or "role" not in response or "content" not in response:
                response = {"role": "assistant", "content": str(response) if response else "[No response data]"}
            
            # Ensure metadata is a dict
            if not isinstance(metadata, dict):
                metadata = {"agent_sdk": True}
            else:
                metadata["agent_sdk"] = True
            
            # Use the default_callback to process inputs and outputs
            callback_result = default_callback(input_messages, response)
                
            # Call _log_trace with all required parameters to avoid NoneType errors
            ragmetrics_client._log_trace(
                input_messages=input_messages,
                response=response,
                metadata_llm=metadata,
                callback_result=callback_result,  # Use processed callback result
                contexts=[],                      # Empty list, not None
                tools=[],                         # Empty list, not None
                duration=0.0                      # Default duration
            )
            return True
        except Exception as e:
            print(f"Error during _log_trace: {e}")
            return False

    def on_trace_end(self, trace):
        """Called when the agent trace completes."""
        try:
            print(f"Agent trace {self.trace_id} completed with {len(self.spans)} spans")
            
            # Create messages and response
            input_messages = [{"role": "system", "content": "Agent trace completed"}]
            response = {"role": "assistant", "content": f"Processed {len(self.spans)} operations in trace {self.trace_id}"}
            
            # Add metadata
            metadata = {
                "agent_trace": True,
                "trace_id": self.trace_id,
                "span_count": len(self.spans),
                "event": "trace_completed"
            }
            
            # Log trace completion with _log_trace
            self._log_trace_safely(input_messages, response, metadata)
                
        except Exception as e:
            print(f"Error in on_trace_end: {e}")

    def shutdown(self):
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
        if ragmetrics_key:
            ragmetrics.login(key=ragmetrics_key)
        else:
            ragmetrics.login()  # assumes ENV var or prior login

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
