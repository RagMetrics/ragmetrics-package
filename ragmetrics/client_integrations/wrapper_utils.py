import time
from typing import Callable, Optional, Any, Tuple, Dict
import inspect # Added for checking awaitable callback
import logging # Import logging
import types # Added

# Forward declaration for RagMetricsClient to avoid circular import if type hinting directly
# from ragmetrics.api import RagMetricsClient 

logger = logging.getLogger(__name__) # Setup logger

# Helper 1: Pop standard RM kwargs
def _pop_ragmetrics_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pops standard RagMetrics parameters from kwargs and returns them."""
    return {
        "user_call_metadata": kwargs.pop("metadata", None),
        "contexts": kwargs.pop("contexts", None),
        "expected": kwargs.pop("expected", None),
        "force_new_conversation": kwargs.pop("force_new_conversation", False)
    }

# Helper 2: Execute sync callback
def _execute_callback(callback: Optional[Callable], log_input: Any, log_response: Any) -> Optional[Any]:
    """Executes the callback synchronously with error handling."""
    callback_result = None
    if callback:
        try:
            callback_result = callback(log_input, log_response)
        except Exception as cb_err:
            logger.error(f"RagMetrics: Error in callback function: {cb_err}", exc_info=True)
    return callback_result

# Helper 3: Execute async callback
async def _aexecute_callback(callback: Optional[Callable], log_input: Any, log_response: Any) -> Optional[Any]:
    """Executes the callback (potentially async) with error handling."""
    callback_result = None
    if callback:
        try:
            potential_result = callback(log_input, log_response)
            if inspect.isawaitable(potential_result):
                 callback_result = await potential_result
            else:
                callback_result = potential_result
        except Exception as cb_err:
            logger.error(f"RagMetrics: Error in async callback function: {cb_err}", exc_info=True)
    return callback_result

def _extract_final_log_parameters(
    dynamic_details: Dict[str, Any],
    static_model_name: Optional[str],
    static_tools: Optional[Any],
    rm_client_metadata: Optional[Dict[str, Any]],
    user_call_metadata: Optional[Dict[str, Any]],
    original_call_remaining_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Helper to compute final logging parameters from various sources."""
    final_model_name = dynamic_details.get("model_name", static_model_name)
    final_tools = dynamic_details.get("tools", static_tools)
    final_tool_choice = dynamic_details.get("tool_choice") 

    final_metadata_for_log = {
        **(rm_client_metadata or {}), 
        **(dynamic_details.get("additional_llm_metadata") or {}), 
        **(user_call_metadata or {})
    }
    
    passthrough_kwargs = { 
        k: v for k, v in original_call_remaining_kwargs.items() 
        if k not in ['model', 'tools', 'tool_choice']
    }
    
    return {
        "final_model_name": final_model_name,
        "final_tools": final_tools,
        "final_tool_choice": final_tool_choice,
        "final_metadata_for_log": final_metadata_for_log,
        "passthrough_kwargs": passthrough_kwargs
    }

def create_sync_wrapper(
    rm_client: Any,  # Should be RagMetricsClient, using Any to avoid import cycle
    original_method: Callable,
    is_target_instance_method: bool, 
    callback: Optional[Callable],
    input_extractor: Callable[[Tuple[Any, ...], Dict[str, Any]], Any], 
    output_extractor: Callable[[Any], Any],
    dynamic_llm_details_extractor: Callable[[Dict[str, Any]], Dict[str, Any]],
    static_model_name: Optional[str] = None,
    static_tools: Optional[Any] = None
):
    """
    Creates a synchronous wrapper around an original method for RagMetrics logging.
    is_target_instance_method: True if the wrapper will be bound to an instance, False otherwise.
    """
    # logger.debug(f"create_sync_wrapper: method {original_method.__name__ if hasattr(original_method, '__name__') else 'unknown'}, is_target_instance_method: {is_target_instance_method}")
    
    def wrapper(*args, **kwargs):
        # logger.debug(f"wrapper entered for {original_method.__name__ if hasattr(original_method, '__name__') else 'unknown'}. is_target_instance_method: {is_target_instance_method}. Args: {args}")
        if rm_client.logging_off:
            if is_target_instance_method:
                return original_method(*(args[1:]), **kwargs) 
            else:
                return original_method(*args, **kwargs)

        start_time = time.time()

        rm_params = _pop_ragmetrics_kwargs(kwargs) # Use helper
        # kwargs now contains only params for original_method/dynamic_extractor

        user_args = args[1:] if is_target_instance_method else args
        log_input = input_extractor(user_args, kwargs)
        dynamic_details = dynamic_llm_details_extractor(kwargs)

        error_obj, result, duration = None, None, 0.0
        try:
            if is_target_instance_method:
                result = original_method(*(args[1:]), **kwargs)
            else:
                result = original_method(*args, **kwargs)
        except Exception as e:
            error_obj = e
        finally:
            duration = time.time() - start_time

        log_response = output_extractor(result) if error_obj is None else None

        final_log_params = _extract_final_log_parameters(
            dynamic_details=dynamic_details,
            static_model_name=static_model_name,
            static_tools=static_tools,
            rm_client_metadata=rm_client.metadata,
            user_call_metadata=rm_params["user_call_metadata"], # From helper result
            original_call_remaining_kwargs=kwargs 
        )

        callback_result = _execute_callback(callback, log_input, log_response) # Use helper

        explicit_trace_args = {
            "input_messages": log_input,
            "response": log_response,
            "expected": rm_params["expected"], # From helper result
            "contexts": rm_params["contexts"], # From helper result
            "metadata_llm": final_log_params["final_metadata_for_log"] or None,
            "error": error_obj,
            "duration": duration,
            "model_name": final_log_params["final_model_name"],
            "tools": final_log_params["final_tools"],
            "tool_choice": final_log_params["final_tool_choice"],
            "callback_result": callback_result,
            "force_new_conversation": rm_params["force_new_conversation"], # From helper result
        }
        rm_client._log_trace(**explicit_trace_args, **final_log_params["passthrough_kwargs"])

        if error_obj:
            raise error_obj

        return result
    return wrapper 

def create_async_wrapper(
    rm_client: Any,  # RagMetricsClient
    original_method: Callable, 
    is_target_instance_method: bool, 
    callback: Optional[Callable], 
    input_extractor: Callable[[Tuple[Any, ...], Dict[str, Any]], Any],
    output_extractor: Callable[[Any], Any],
    dynamic_llm_details_extractor: Callable[[Dict[str, Any]], Dict[str, Any]],
    static_model_name: Optional[str] = None,
    static_tools: Optional[Any] = None
):
    """
    Creates an asynchronous wrapper.
    is_target_instance_method: True if the wrapper will be bound to an instance, False otherwise.
    """
    # logger.debug(f"create_async_wrapper: method {original_method.__name__ if hasattr(original_method, '__name__') else 'unknown'}, is_target_instance_method: {is_target_instance_method}")

    async def wrapper(*args, **kwargs):
        # logger.debug(f"async wrapper entered for {original_method.__name__ if hasattr(original_method, '__name__') else 'unknown'}. is_target_instance_method: {is_target_instance_method}. Args: {args}")
        if rm_client.logging_off:
            if is_target_instance_method:
                 return await original_method(*(args[1:]), **kwargs) 
            else:
                 return await original_method(*args, **kwargs)

        start_time = time.time()

        rm_params = _pop_ragmetrics_kwargs(kwargs) # Use helper
        
        user_args = args[1:] if is_target_instance_method else args
        log_input = input_extractor(user_args, kwargs)
        dynamic_details = dynamic_llm_details_extractor(kwargs)

        error_obj, result, duration = None, None, 0.0
        try:
            if is_target_instance_method:
                 result = await original_method(*(args[1:]), **kwargs)
            else:
                 result = await original_method(*args, **kwargs)
        except Exception as e:
            error_obj = e
        finally:
            duration = time.time() - start_time

        log_response = output_extractor(result) if error_obj is None else None

        final_log_params = _extract_final_log_parameters(
            dynamic_details=dynamic_details,
            static_model_name=static_model_name,
            static_tools=static_tools,
            rm_client_metadata=rm_client.metadata,
            user_call_metadata=rm_params["user_call_metadata"], # From helper result
            original_call_remaining_kwargs=kwargs 
        )

        callback_result = await _aexecute_callback(callback, log_input, log_response) # Use async helper
        
        explicit_trace_args = {
            "input_messages": log_input,
            "response": log_response,
            "expected": rm_params["expected"], # From helper result
            "contexts": rm_params["contexts"], # From helper result
            "metadata_llm": final_log_params["final_metadata_for_log"] or None,
            "error": error_obj,
            "duration": duration,
            "model_name": final_log_params["final_model_name"],
            "tools": final_log_params["final_tools"],
            "tool_choice": final_log_params["final_tool_choice"],
            "callback_result": callback_result,
            "force_new_conversation": rm_params["force_new_conversation"], # From helper result
        }

        if not hasattr(rm_client, '_alog_trace'):
            logger.error("RagMetrics: _alog_trace not found, using sync _log_trace.")
            rm_client._log_trace(**explicit_trace_args, **final_log_params["passthrough_kwargs"])
        else:
            await rm_client._alog_trace(**explicit_trace_args, **final_log_params["passthrough_kwargs"])

        if error_obj:
            raise error_obj

        return result
    return wrapper