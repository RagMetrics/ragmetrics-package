"""
Integration with OpenAI Agents SDK for RagMetrics.
"""

import iso8601
from typing import Any
import json
import types
import types
from datetime import datetime
from ragmetrics.api import ragmetrics_client, default_callback
# Note: Do not import the full ragmetrics package to avoid circular dependencies
from ragmetrics.utils import format_function_signature

# Define TracingProcessor as a base class in case import fails
class TracingProcessor:
    """Base class if agents SDK is not installed"""
    def on_trace_start(self, trace): pass
    def on_span_start(self, span): pass
    def on_span_end(self, span): pass
    def on_trace_end(self, trace): pass
    def shutdown(self): pass
    def force_flush(self): pass

# Try to import agents SDK
try:
    from agents.tracing.processor_interface import TracingProcessor
    from agents import set_trace_processors
    AGENTS_SDK_AVAILABLE = True
except ImportError:
    # Define a simple function for when agents SDK is not available
    def set_trace_processors(processors):
        """Fallback function when agents SDK is not available"""
        print("Warning: agents SDK not available, tracing will not work")
    AGENTS_SDK_AVAILABLE = False


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
            print(self.conversation_id)
        except Exception as e:
            print(f"Error in on_trace_start: {e}")


    def on_span_start(self, span):
        # Just collect the start time
        pass

    def _raw_input_from_span(self, span):
        if 'span_data' in span:
            span_data = span['span_data']
            span_type = span_data.get('type', None)
            if span_type == 'function':
                func_sig = format_function_signature(
                    func_name=span_data['name'], 
                    args_dict=json.loads(span_data['input'])
                )
                raw_input = {
                    'content': func_sig,
                    'tool_call': True
                }
            elif span_type == 'agent':
                raw_input = None
            elif 'input' in span_data:
                raw_input = span_data['input']
            else:
                raw_input = span_data
        else:
            raw_input = None
        
        return raw_input        
    
    def on_span_end(self, span):
        """Called when a span ends (operation completes)."""

        # Raw IO
        raw_output = self._to_dict(span)
        raw_input = self._raw_input_from_span(raw_output)     

        # Formatted IO
        # TODO: Support alternative callbacks from .monitor()
        callback_result = default_callback(raw_input, raw_output)
        
        # Duration
        end_time = iso8601.parse_date(span.ended_at)
        start_time = iso8601.parse_date(span.started_at)
        duration = (end_time - start_time).total_seconds()
        
        #Log to RagMetrics
        ragmetrics_client._log_trace(
            input_messages=raw_input,
            response=raw_output,
            callback_result=callback_result,
            conversation_id=self.conversation_id,
            duration=duration
        )

    def _to_dict(self, obj: Any, visited=None) -> Any:
        """Convert an object to a JSON-serializable dictionary.
        
        Args:
            obj: The object to serialize
            visited: Set of object IDs to detect circular references
                
        Returns:
            A JSON-serializable representation of the input object
        """
        if visited is None:
            visited = set()

        # Handle None
        if obj is None:
            return None
            
        # Handle circular references
        obj_id = id(obj)
        if obj_id in visited:
            return f"<circular ref id={obj_id}>"
        visited.add(obj_id)

        # Handle basic types
        if isinstance(obj, (str, int, float, bool)):
            return obj
            
        # Handle datetime objects
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()

        # Handle containers
        if isinstance(obj, (list, tuple, set)):
            return [self._to_dict(x, visited) for x in obj]
            
        if isinstance(obj, dict):
            return {k: self._to_dict(v, visited) for k, v in obj.items()}

        # Handle span objects specially
        if hasattr(obj, 'span_data') or hasattr(obj, 'started_at') or hasattr(obj, 'ended_at'):
            result = {'type': type(obj).__name__}
            
            # Get all attributes of the span
            for attr_name in dir(obj):
                if attr_name.startswith('_') or callable(getattr(obj, attr_name)):
                    continue
                    
                try:
                    attr_value = getattr(obj, attr_name)
                    if attr_value is not None:
                        result[attr_name] = self._to_dict(attr_value, visited)
                except Exception as e:
                    result[f'error_serializing_{attr_name}'] = str(e)
            
            # Special handling for span_data to ensure we capture everything
            if hasattr(obj, 'span_data'):
                try:
                    span_data = obj.span_data
                    if span_data is not None:
                        # First get all attributes from __dict__ if available
                        if hasattr(span_data, '__dict__'):
                            result['span_data'] = {}
                            for k, v in vars(span_data).items():
                                if not k.startswith('_') and v is not None:
                                    result['span_data'][k] = self._to_dict(v, visited)
                        
                        # Then ensure we get any properties that might not be in __dict__
                        for attr_name in dir(span_data):
                            if (not attr_name.startswith('_') and 
                                not callable(getattr(span_data, attr_name)) and 
                                attr_name not in result.get('span_data', {})):
                                try:
                                    val = getattr(span_data, attr_name)
                                    if val is not None:
                                        if 'span_data' not in result:
                                            result['span_data'] = {}
                                        result['span_data'][attr_name] = self._to_dict(val, visited)
                                except Exception:
                                    continue
                except Exception as e:
                    result['span_data_error'] = str(e)
            
            return result

        # Handle other objects with __dict__
        if hasattr(obj, '__dict__'):
            try:
                obj_dict = {}
                for k, v in vars(obj).items():
                    if not k.startswith('_') and v is not None:
                        obj_dict[k] = self._to_dict(v, visited)
                return {'type': type(obj).__name__, **obj_dict}
            except Exception:
                pass

        # Handle objects with __slots__
        slots = getattr(type(obj), "__slots__", None)
        if slots:
            data = {}
            for slot in slots:
                if isinstance(slot, str) and hasattr(obj, slot):
                    try:
                        val = getattr(obj, slot)
                        if val is not None:
                            data[slot] = self._to_dict(val, visited)
                    except Exception:
                        continue
            if data:
                return {'type': type(obj).__name__, **data}

        # For any other type, try to get its string representation
        try:
            return str(obj)
        except Exception:
            return f'<unserializable {type(obj).__name__}>'

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
            raw_output = span.export()
        
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
        
    Returns:
        The configured tracing processor or None if setup failed
    """
    if not AGENTS_SDK_AVAILABLE:
        print("OpenAI Agents SDK is not installed. Please install with pip install openai-agents")
        return None

    try:
        # Import here to avoid circular import
        import ragmetrics
        
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


def trace(mcp_client_session):
    """
    Wraps the send_request method of a client session with tracing logic.
    """
    original_send_request = mcp_client_session.send_request

    async def traced_send_request(self_instance, request, *args, **kwargs):
        try:
            result = await original_send_request(self_instance, request, *args, **kwargs)

            request_data = extract_dict(request)
            response_data = extract_dict(result)

            method = request_data.get("method")
            params = request_data.get("params")

            raw_input, raw_output = process_trace(method, params, response_data)
            callback_result = default_callback(raw_input, raw_output)

            ragmetrics_client._log_trace(
                input_messages=raw_input,
                response=raw_output,
                callback_result=callback_result,
                conversation_id=ragmetrics_client.new_conversation(),
            )
            return result

        except Exception as e:
            print(f"Error during traced send_request: {e}")
            traceback.print_exc()

    mcp_client_session.send_request = traced_send_request


def extract_dict(obj):
    """
    Safely extract a dictionary from an object, using .dict() or .model_dump() if available.
    """
    if hasattr(obj, 'dict'):
        return obj.dict()
    elif hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return obj
    return {}


def process_trace(method, params, response_data):
    """
    Determine raw input and output data to log based on the request method.
    """
    if method == "initialize":
        return params, "response_data"

    elif method == "tools/list":
        return params, response_data

    elif method == "tools/call":
        func_sig = format_function_signature(
            func_name=params.get("name"),
            args_dict=params.get("arguments", {})
        )
        return {
            'content': func_sig,
            'tool_call': True
        }, response_data

    return params or {}, response_data or {}

def monitor_mcp_server(mcp_client_session=None):
    trace(mcp_client_session)
