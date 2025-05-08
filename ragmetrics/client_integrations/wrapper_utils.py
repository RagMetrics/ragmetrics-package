import time
from typing import Callable, Optional, Any, Tuple, Dict
import inspect # Added for checking awaitable callback
import logging # Import logging
import types # Added

# Forward declaration for RagMetricsClient to avoid circular import if type hinting directly
# from ragmetrics.api import RagMetricsClient 

logger = logging.getLogger(__name__) # Setup logger

def create_sync_wrapper(
    rm_client: Any,  # Should be RagMetricsClient, using Any to avoid import cycle
    original_method: Callable,
    is_target_instance_method: bool, # New parameter
    callback: Optional[Callable],
    input_extractor: Callable[[Tuple[Any, ...], Dict[str, Any]], Any], 
    output_extractor: Callable[[Any], Any],
    # Extracts dynamic details from call_kwargs.
    # Returns a dict like: 
    # {
    #   "model_name": "model_from_kwargs_if_any", 
    #   "tools": "tools_from_kwargs_if_any", 
    #   "tool_choice": "tool_choice_from_kwargs_if_any",
    #   "additional_llm_metadata": {"param1": value1, ...}
    # }
    dynamic_llm_details_extractor: Callable[[Dict[str, Any]], Dict[str, Any]],
    static_model_name: Optional[str] = None,
    static_tools: Optional[Any] = None
    # static_tool_choice is less common as a static attribute, usually dynamic
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

        user_call_metadata = kwargs.pop("metadata", None)
        contexts = kwargs.pop("contexts", None)
        expected = kwargs.pop("expected", None)
        force_new_conversation = kwargs.pop("force_new_conversation", False)
        
        user_args = args[1:] if is_target_instance_method else args
        # logger.debug(f"About to call input_extractor. is_target_instance_method: {is_target_instance_method}. user_args: {user_args}, kwargs: {kwargs}")
        log_input = input_extractor(user_args, kwargs)
        # logger.debug(f"input_extractor returned: {log_input}")
        
        dynamic_details = dynamic_llm_details_extractor(kwargs)

        error_obj = None
        result = None
        try:
            if is_target_instance_method:
                result = original_method(*(args[1:]), **kwargs)
            else:
                result = original_method(*args, **kwargs)
        except Exception as e:
            error_obj = e
        finally:
            end_time = time.time()
            duration = end_time - start_time

        log_response = output_extractor(result) if error_obj is None else None

        # Determine final model_name, tools, tool_choice
        # Prioritize dynamic values from the call over static ones from the client
        final_model_name = dynamic_details.get("model_name", static_model_name)
        final_tools = dynamic_details.get("tools", static_tools)
        final_tool_choice = dynamic_details.get("tool_choice") # No static_tool_choice usually

        # Prepare combined metadata for logging
        # Order of precedence: client.metadata < dynamic_details.additional_llm_metadata < user_call_metadata
        final_metadata_for_log = {**(rm_client.metadata or {}), **(dynamic_details.get("additional_llm_metadata") or {}), **(user_call_metadata or {})}

        # kwargs in the current scope are those remaining after popping RagMetrics-specific ones.
        # These were passed to the original_method and seen by dynamic_llm_details_extractor.
        # We want to pass any *other* kwargs to _log_trace, avoiding duplication with explicitly mapped params.
        log_trace_passthrough_kwargs = { 
            k: v for k, v in kwargs.items() 
            if k not in ['model', 'tools', 'tool_choice'] # These are handled by final_model_name, final_tools, etc.
        }

        callback_result = None
        if callback:
            try:
                # Pass only input and response to standard callback signature
                callback_result = callback(log_input, log_response)
            except Exception as cb_err:
                # Log error with logger, not print
                logger.error(f"RagMetrics: Error in callback function: {cb_err}", exc_info=True)

        rm_client._log_trace(
            input_messages=log_input,
            response=log_response,
            expected=expected,
            contexts=contexts,
            metadata_llm=final_metadata_for_log or None, # _log_trace expects None if empty
            error=error_obj,
            duration=duration,
            model_name=final_model_name,
            tools=final_tools,
            tool_choice=final_tool_choice,
            callback_result=callback_result,
            force_new_conversation=force_new_conversation,
            **log_trace_passthrough_kwargs 
        )

        if error_obj:
            raise error_obj

        return result
    return wrapper 

# Corrected: create_async_wrapper is a sync function that returns an async function
def create_async_wrapper(
    rm_client: Any,  # RagMetricsClient
    original_method: Callable, # This will be an async method
    is_target_instance_method: bool, # New parameter
    callback: Optional[Callable], # Can be sync or async
    input_extractor: Callable[[Tuple[Any, ...], Dict[str, Any]], Any],
    output_extractor: Callable[[Any], Any],
    dynamic_llm_details_extractor: Callable[[Dict[str, Any]], Dict[str, Any]],
    static_model_name: Optional[str] = None,
    static_tools: Optional[Any] = None
):
    """
    Creates an asynchronous wrapper around an original async method for RagMetrics logging.
    This function itself is synchronous, but it returns an `async def` wrapper.
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

        user_call_metadata = kwargs.pop("metadata", None)
        contexts = kwargs.pop("contexts", None)
        expected = kwargs.pop("expected", None)
        force_new_conversation = kwargs.pop("force_new_conversation", False)
        
        user_args = args[1:] if is_target_instance_method else args
        # logger.debug(f"About to call async input_extractor. is_target_instance_method: {is_target_instance_method}. user_args: {user_args}, kwargs: {kwargs}")
        log_input = input_extractor(user_args, kwargs)
        # logger.debug(f"Async input_extractor returned: {log_input}")
        dynamic_details = dynamic_llm_details_extractor(kwargs)

        error_obj = None
        result = None
        try:
            if is_target_instance_method:
                 result = await original_method(*(args[1:]), **kwargs)
            else:
                 result = await original_method(*args, **kwargs)
        except Exception as e:
            error_obj = e
        finally:
            end_time = time.time()
            duration = end_time - start_time

        log_response = output_extractor(result) if error_obj is None else None

        final_model_name = dynamic_details.get("model_name", static_model_name)
        final_tools = dynamic_details.get("tools", static_tools)
        final_tool_choice = dynamic_details.get("tool_choice")

        final_metadata_for_log = {**(rm_client.metadata or {}), **(dynamic_details.get("additional_llm_metadata") or {}), **(user_call_metadata or {})}

        # Similar logic for async
        alog_trace_passthrough_kwargs = { 
            k: v for k, v in kwargs.items() 
            if k not in ['model', 'tools', 'tool_choice']
        }

        callback_result = None
        if callback:
            try:
                # Pass only input and response to standard callback signature
                # Await if the callback itself is async or returns an awaitable
                potential_result = callback(log_input, log_response)
                if inspect.isawaitable(potential_result):
                     callback_result = await potential_result
                else:
                    callback_result = potential_result
            except Exception as cb_err:
                # Log error with logger, not print
                logger.error(f"RagMetrics: Error in callback function: {cb_err}", exc_info=True)

        if not hasattr(rm_client, '_alog_trace'):
            logger.error("RagMetrics: _alog_trace not found, using sync _log_trace.")
            rm_client._log_trace(
                input_messages=log_input,
                response=log_response,
                expected=expected,
                contexts=contexts,
                metadata_llm=final_metadata_for_log or None,
                error=error_obj,
                duration=duration,
                model_name=final_model_name,
                tools=final_tools,
                tool_choice=final_tool_choice,
                callback_result=callback_result,
                force_new_conversation=force_new_conversation,
                **alog_trace_passthrough_kwargs 
            )
        else:
            await rm_client._alog_trace(
                input_messages=log_input,
                response=log_response,
                expected=expected,
                contexts=contexts,
                metadata_llm=final_metadata_for_log or None,
                error=error_obj,
                duration=duration,
                model_name=final_model_name,
                tools=final_tools,
                tool_choice=final_tool_choice,
                callback_result=callback_result,
                force_new_conversation=force_new_conversation,
                **alog_trace_passthrough_kwargs 
            )

        if error_obj:
            raise error_obj

        return result
    return wrapper 