import time
from typing import Callable, Optional, Any, Tuple, Dict
import inspect # Added for checking awaitable callback
import logging # Import logging
import types # Added


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

    # Log metadata sources for debugging
    logger.info(f"Merging metadata from sources:")
    logger.info(f"- rm_client_metadata: {rm_client_metadata}")
    logger.info(f"- dynamic_details additional_llm_metadata: {dynamic_details.get('additional_llm_metadata')}")
    logger.info(f"- user_call_metadata: {user_call_metadata}")

    # Important: The order here determines precedence (later sources override earlier ones)
    final_metadata_for_log = {
        **(rm_client_metadata or {}), 
        **(dynamic_details.get("additional_llm_metadata") or {}), 
        **(user_call_metadata or {})
    }
    
    logger.info(f"Final merged metadata: {final_metadata_for_log}")
    
    passthrough_kwargs = { 
        k: v for k, v in original_call_remaining_kwargs.items() 
        if k not in ['model', 'tools', 'tool_choice', 'endpoint', 'method']
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
    method_name = getattr(original_method, '__name__', 'unknown_method')
    logger.info(f"Creating sync wrapper for {method_name}, is_instance_method={is_target_instance_method}")
    
    def wrapper(*args, **kwargs):
        logger.info(f"RagMetrics SYNC wrapper for {method_name} CALLED with {len(args)} args")
        if rm_client.logging_off:
            logger.debug(f"RagMetrics logging is OFF, calling original method directly")
            return original_method(*args, **kwargs)

        start_time = time.time()

        rm_params = _pop_ragmetrics_kwargs(kwargs) # Use helper
        # kwargs now contains only params for original_method/dynamic_extractor

        # For both instance and class methods, use all args for extraction
        # The extractors need to be able to handle the different formats
        log_input = input_extractor(args, kwargs)
        dynamic_details = dynamic_llm_details_extractor(kwargs)

        error_obj, result, duration = None, None, 0.0
        try:
            # Call original method with all original args
            logger.debug(f"Calling original method {method_name}")
            if is_target_instance_method and len(args) > 0:
                # For instance methods, we need to ensure we're not passing the instance twice
                # args[0] is the instance (self)
                if len(args) > 1:
                    # If there are more args beyond self, call with those
                    result = original_method(args[0], *args[1:], **kwargs)
                else:
                    # Only self was passed
                    result = original_method(args[0], **kwargs)
            else:
                # For non-instance methods or when no args are provided
                result = original_method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in original method {method_name}: {e}")
            error_obj = e
        finally:
            duration = time.time() - start_time
            logger.debug(f"Original method {method_name} completed in {duration:.2f}s")

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
        
        logger.info(f"Calling _log_trace for {method_name}")
        try:
            trace_result = rm_client._log_trace(**explicit_trace_args, **final_log_params["passthrough_kwargs"])
            if trace_result:
                logger.info(f"Successfully logged trace for {method_name}")
            else:
                logger.warning(f"Failed to log trace for {method_name}")
        except Exception as e:
            logger.error(f"Error logging trace for {method_name}: {e}", exc_info=True)

        if error_obj:
            raise error_obj

        return result
    
    # Add identifier for debugging
    wrapper.__wrapped_method__ = method_name
    wrapper.__ragmetrics_wrapped__ = True
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
    method_name = getattr(original_method, '__name__', 'unknown_method')
    logger.info(f"Creating async wrapper for {method_name}, is_instance_method={is_target_instance_method}")

    async def wrapper(*args, **kwargs):
        logger.info(f"RagMetrics ASYNC wrapper for {method_name} CALLED with {len(args)} args")
        if rm_client.logging_off:
            logger.debug(f"RagMetrics logging is OFF, calling original method directly")
            return await original_method(*args, **kwargs)

        start_time = time.time()

        rm_params = _pop_ragmetrics_kwargs(kwargs) # Use helper
        
        # For both instance and class methods, use all args for extraction
        # The extractors need to be able to handle the different formats
        log_input = input_extractor(args, kwargs)
        dynamic_details = dynamic_llm_details_extractor(kwargs)

        error_obj, result, duration = None, None, 0.0
        try:
            # Call original method with all original args
            logger.debug(f"Calling original async method {method_name}")
            if is_target_instance_method and len(args) > 0:
                # For instance methods, we need to ensure we're not passing the instance twice
                # args[0] is the instance (self)
                if len(args) > 1:
                    # If there are more args beyond self, call with those
                    result = await original_method(args[0], *args[1:], **kwargs)
                else:
                    # Only self was passed
                    result = await original_method(args[0], **kwargs)
            else:
                # For non-instance methods or when no args are provided
                result = await original_method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in original async method {method_name}: {e}")
            error_obj = e
        finally:
            duration = time.time() - start_time
            logger.debug(f"Original async method {method_name} completed in {duration:.2f}s")

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

        logger.info(f"Logging async trace for {method_name}")
        try:
            if not hasattr(rm_client, '_alog_trace'):
                logger.error("RagMetrics: _alog_trace not found, using sync _log_trace.")
                trace_result = rm_client._log_trace(**explicit_trace_args, **final_log_params["passthrough_kwargs"])
            else:
                trace_result = await rm_client._alog_trace(**explicit_trace_args, **final_log_params["passthrough_kwargs"])
                
            if trace_result:
                logger.info(f"Successfully logged async trace for {method_name}")
            else:
                logger.warning(f"Failed to log async trace for {method_name}")
        except Exception as e:
            logger.error(f"Error logging async trace for {method_name}: {e}", exc_info=True)
        
        if error_obj:
            raise error_obj
            
        return result

    # Add identifier for debugging
    wrapper.__wrapped_method__ = method_name
    wrapper.__ragmetrics_wrapped__ = True
    return wrapper