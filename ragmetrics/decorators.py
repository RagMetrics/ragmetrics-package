import time
import inspect
from typing import Callable

# This will be the global client instance, to be imported from a central place like config.py or api.py
# from .config import ragmetrics_client # Placeholder for now

def trace_function_call(func: Callable) -> Callable:
    '''
    Decorator to trace function execution and log structured input/output.
    
    Wrap a function with this decorator to automatically log its execution
    details to RagMetrics, including inputs, outputs, and timing information.
    This is particularly useful for tracking retrieval functions in RAG applications.

    Example - Tracing a weather API function:
        
        .. code-block:: python
        
            import requests
            # Assuming ragmetrics and its global client are initialized
            # from ragmetrics.api import ragmetrics_client
            from ragmetrics.decorators import trace_function_call
            
            # Apply the decorator to your function
            @trace_function_call
            def get_weather(latitude, longitude):
                api_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
                response = requests.get(api_url)
                data = response.json()
                return data['current']['temperature_2m']
                
            # Now when you call the function, it's automatically traced
            # temperature = get_weather(48.8566, 2.3522)  # Paris coordinates
        
    Args:
        func: The function to be traced.

    Returns:
        Callable: A wrapped version of the function that logs execution details.
    '''
    def wrapper(*args, **kwargs):
        # Import here to avoid circular dependency issues at module load time
        # This assumes api.py will define and instantiate ragmetrics_client
        from ragmetrics.api import ragmetrics_client
        
        if ragmetrics_client.logging_off or not ragmetrics_client.access_token:
            # If logging is off or client not authenticated, just run the original function
            return func(*args, **kwargs)

        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        params = []
        try:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            for i, arg_val in enumerate(args):
                if i < len(param_names):
                    params.append(f"{param_names[i]}={repr(arg_val)}")
                else:
                    params.append(f"{repr(arg_val)}")
        except Exception: # Fallback if inspect fails
            for arg_val in args:
                params.append(f"{repr(arg_val)}")
        
        for k, v_val in kwargs.items():
            params.append(f"{k}={repr(v_val)}")
        
        formatted_call = f"={func.__name__}({', '.join(params)})"

        function_input_for_log_trace = [
            {
                "role": "user",
                "content": formatted_call,
                "tool_call": True # Indicate this is a traced function, not a direct LLM call
            }
        ]

        function_output_for_log_trace = result 

        callback_payload = {
            "input": formatted_call,
            "output": str(result) 
        }

        ragmetrics_client._log_trace(
            input_messages=function_input_for_log_trace,
            response=function_output_for_log_trace, 
            metadata_llm={"function_name": func.__name__, "decorated_call": True},
            contexts=None, 
            expected=None, 
            duration=duration,
            tools=None,  
            callback_result=callback_payload
        )
        return result

    return wrapper 