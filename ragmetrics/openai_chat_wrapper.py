import types
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

def create_wrapper(original_method):
    """
    Create a wrapper for OpenAI's synchronous chat completion method that logs to RagMetrics.
    
    Args:
        original_method: The original OpenAI chat completion method to wrap.
        
    Returns:
        A wrapped version of the method that logs to RagMetrics.
    """
    def wrapper(self_instance, *args, **kwargs):
        from ragmetrics import ragmetrics_client
        
        start_time = None
        try:
            import time
            start_time = time.time()
            
            # Extract RagMetrics-specific parameters
            metadata_llm = kwargs.get('metadata', None)
            # Make a copy to avoid modifying the original kwargs
            kwargs_copy = deepcopy(kwargs)
            
            # Extract and remove RagMetrics parameters
            contexts = kwargs_copy.pop('contexts', None)
            expected = kwargs_copy.pop('expected', None)
            
            # Call the original method with cleaned kwargs
            response = original_method(self_instance, *args, **kwargs_copy)
            
            # Calculate duration
            duration = time.time() - start_time if start_time else None
            
            # Extract other parameters needed for logging
            input_messages = kwargs.get('messages', None)
            tools = kwargs.get('tools', None)
            
            # Default callback for processing input/output
            from ragmetrics import default_callback
            cb_result = default_callback(input_messages, response)
            
            # Log the trace
            ragmetrics_client._log_trace(
                input_messages=input_messages,
                response=response,
                metadata_llm=metadata_llm,
                contexts=contexts,
                expected=expected,
                duration=duration,
                tools=tools,
                callback_result=cb_result
            )
            
            return response
        except Exception as e:
            logger.error(f"Error in RagMetrics wrapper: {str(e)}")
            # If wrapping fails, fall back to the original method
            return original_method(self_instance, *args, **kwargs)
    
    return wrapper

def acreate_wrapper(original_method):
    """
    Create a wrapper for OpenAI's asynchronous chat completion method that logs to RagMetrics.
    
    Args:
        original_method: The original OpenAI async chat completion method to wrap.
        
    Returns:
        A wrapped version of the async method that logs to RagMetrics.
    """
    async def async_wrapper(self_instance, *args, **kwargs):
        from ragmetrics import ragmetrics_client
        
        start_time = None
        try:
            import time
            start_time = time.time()
            
            # Extract RagMetrics-specific parameters
            metadata_llm = kwargs.get('metadata', None)
            # Make a copy to avoid modifying the original kwargs
            kwargs_copy = deepcopy(kwargs)
            
            # Extract and remove RagMetrics parameters
            contexts = kwargs_copy.pop('contexts', None)
            expected = kwargs_copy.pop('expected', None)
            
            # Call the original async method with cleaned kwargs
            response = await original_method(self_instance, *args, **kwargs_copy)
            
            # Calculate duration
            duration = time.time() - start_time if start_time else None
            
            # Extract other parameters needed for logging
            input_messages = kwargs.get('messages', None)
            tools = kwargs.get('tools', None)
            
            # Default callback for processing input/output
            from ragmetrics import default_callback
            cb_result = default_callback(input_messages, response)
            
            # Log the trace
            ragmetrics_client._log_trace(
                input_messages=input_messages,
                response=response,
                metadata_llm=metadata_llm,
                contexts=contexts,
                expected=expected,
                duration=duration,
                tools=tools,
                callback_result=cb_result
            )
            
            return response
        except Exception as e:
            logger.error(f"Error in RagMetrics async wrapper: {str(e)}")
            # If wrapping fails, fall back to the original method
            return await original_method(self_instance, *args, **kwargs)
    
    return async_wrapper

def patch_openai_client(client):
    """
    Patch an OpenAI client to log interactions to RagMetrics.
    
    Args:
        client: An OpenAI client instance.
        
    Returns:
        The patched client.
    """
    try:
        # Store original methods
        original_create = type(client.chat.completions).create
        
        # Patch synchronous method
        client.chat.completions.create = types.MethodType(
            create_wrapper(original_create), 
            client.chat.completions
        )
        
        # If async method exists, patch it too
        if hasattr(client.chat.completions, "acreate"):
            original_acreate = type(client.chat.completions).acreate
            client.chat.completions.acreate = types.MethodType(
                acreate_wrapper(original_acreate), 
                client.chat.completions
            )
        
        logger.info("Successfully patched OpenAI client for RagMetrics monitoring")
        return client
    except Exception as e:
        logger.error(f"Failed to patch OpenAI client: {str(e)}")
        return client 