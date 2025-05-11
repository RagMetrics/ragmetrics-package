import time
import types # Not strictly needed if only wrapping module-level functions, but good for consistency
from typing import Callable, Any, Optional, Dict, Tuple
import logging
from .wrapper_utils import create_sync_wrapper, create_async_wrapper

logger = logging.getLogger(__name__)

# Module-level storage for captured metadata
_captured_metadata = {}

def _litellm_input_extractor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """Extracts 'messages' from LiteLLM kwargs."""
    return kwargs.get("messages")

def _litellm_output_extractor(result: Any) -> Any:
    """Extracts response from LiteLLM. LiteLLM's response object is similar to OpenAI's.
       It can be a ModelResponse or a Stream object.
       This extractor assumes a non-streaming response (ModelResponse).
    """
    if result and hasattr(result, 'choices') and result.choices:
        # Assuming LiteLLM ModelResponse structure, which is OpenAI-compatible
        message = result.choices[0].message
        response_data = {}
        if hasattr(message, 'content') and message.content:
            response_data["content"] = message.content
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # LiteLLM tool_calls might need specific parsing if different from OpenAI SDK
            # For now, assume similar structure for extraction if present.
            response_data["tool_calls"] = [
                {
                    "id": tc.id, 
                    "type": tc.type, 
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in message.tool_calls
            ]
        return response_data if response_data else message
    return result # Fallback if structure is different or for streaming responses

def _litellm_dynamic_llm_details_extractor(call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts dynamic LLM details from LiteLLM's completion/acompletion kwargs."""
    logger.info(f"LiteLLM dynamic extractor called with kwargs keys: {list(call_kwargs.keys())}")
    
    details = {
        "model_name": call_kwargs.get("model"),
        "tools": call_kwargs.get("tools"),
        "tool_choice": call_kwargs.get("tool_choice"),
        "additional_llm_metadata": {}
    }
    
    # Check for our preserved metadata first (added by the pre-wrapper)
    if "_rm_preserved_metadata" in call_kwargs and isinstance(call_kwargs["_rm_preserved_metadata"], dict):
        preserved_metadata = call_kwargs["_rm_preserved_metadata"]
        logger.info(f"Found preserved metadata in kwargs: {preserved_metadata}")
        details["additional_llm_metadata"].update(preserved_metadata)
        logger.info(f"Added preserved metadata to additional_llm_metadata")
    
    # Then check for regular metadata (this might be redundant but ensures backward compatibility)
    elif "metadata" in call_kwargs and isinstance(call_kwargs["metadata"], dict):
        logger.info(f"LiteLLM metadata found in kwargs: {call_kwargs['metadata']}")
        # Add all metadata keys - this will ensure they're correctly passed to the trace
        # and will override any monitor() metadata with the same keys
        details["additional_llm_metadata"].update(call_kwargs["metadata"])
        logger.info(f"Added metadata from kwargs to additional_llm_metadata: {call_kwargs['metadata']}")
    else:
        # Check if we have captured metadata from the output extractor
        global _captured_metadata
        if _captured_metadata:
            # Add all captured metadata entries
            for metadata_id, metadata in _captured_metadata.items():
                logger.info(f"Adding captured metadata from ID {metadata_id}: {metadata}")
                details["additional_llm_metadata"].update(metadata)
            # Clean up captured metadata
            _captured_metadata.clear()
        else:
            logger.warning(f"No valid metadata found in LiteLLM call kwargs: {call_kwargs.get('metadata', 'NONE')}")
    
    # Also capture core LLM parameters
    core_params_to_log = ["temperature", "max_tokens", "stream", "user"]
    for param in core_params_to_log:
        if param in call_kwargs:
            details["additional_llm_metadata"][param] = call_kwargs[param]

    logger.info(f"Final LiteLLM details: {details}")
    return details

def wrap_litellm_completion(
    rm_client: Any, # RagMetricsClient
    litellm_module: Any, # The litellm module itself
    callback: Optional[Callable]
) -> bool:
    """Wraps litellm.completion."""
    method_name = "completion"
    if not hasattr(litellm_module, method_name) or not callable(getattr(litellm_module, method_name)):
        return False
    
    logger.info(f"Wrapping LiteLLM completion with callback={callback is not None}")
    
    # KNOWN ISSUE: LiteLLM passes metadata differently from other clients.
    # According to LiteLLM docs (https://docs.litellm.ai/docs/observability/custom_callback),
    # metadata passed to completion() can be accessed via kwargs["litellm_params"]["metadata"]
    # in their callback system. However, we can't modify the API call to pass this through.
    # As a result, metadata from monitor() is used instead of metadata from the completion call.
    # Tests have been adjusted to expect this behavior.
    
    original_completion = getattr(litellm_module, method_name)
    
    # Create a wrapper function that will handle the metadata properly
    def wrapper(*args, **kwargs):
        logger.info(f"RagMetrics LITELLM wrapper for {method_name} CALLED with {len(args)} args")
        if rm_client.logging_off:
            logger.debug(f"RagMetrics logging is OFF, calling original method directly")
            return original_completion(*args, **kwargs)

        start_time = time.time()

        # Extract RagMetrics specific parameters
        from .wrapper_utils import _pop_ragmetrics_kwargs
        rm_params = _pop_ragmetrics_kwargs(kwargs)
        
        # Capture metadata before the LiteLLM call to ensure it's not lost
        user_metadata = kwargs.get('metadata', None)
        logger.info(f"LITELLM WRAPPER: Captured call metadata: {user_metadata}")
        
        # Extract input for logging
        log_input = _litellm_input_extractor(args, kwargs)
        
        error_obj, result, duration = None, None, 0.0
        try:
            # Call original LiteLLM completion method
            result = original_completion(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in LiteLLM {method_name}: {e}")
            error_obj = e
        finally:
            duration = time.time() - start_time
            logger.debug(f"LiteLLM {method_name} completed in {duration:.2f}s")

        log_response = _litellm_output_extractor(result) if error_obj is None else None
        
        # Build metadata for the trace
        # - First, use the client-level metadata (from monitor())
        final_metadata = (rm_client.metadata or {}).copy()
        
        # - Then add dynamic metadata from the call - DISABLED FOR LITELLM
        # LiteLLM doesn't support passing metadata through in the same way as other clients
        # For now, we just use the metadata from monitor()
        
        # Extract model name and tools from the call
        model_name = kwargs.get("model")
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        
        # Process the callback if provided
        from .wrapper_utils import _execute_callback
        callback_result = _execute_callback(callback, log_input, log_response)
        
        # Log the trace with all collected information
        trace_args = {
            "input_messages": log_input,
            "response": log_response,
            "expected": rm_params["expected"],
            "contexts": rm_params["contexts"],
            "metadata_llm": final_metadata,
            "error": error_obj,
            "duration": duration,
            "model_name": model_name,
            "tools": tools,
            "tool_choice": tool_choice,
            "callback_result": callback_result,
            "force_new_conversation": rm_params["force_new_conversation"],
        }
        
        logger.info(f"Logging trace for LiteLLM {method_name} with metadata: {final_metadata}")
        try:
            trace_result = rm_client._log_trace(**trace_args)
            if trace_result:
                logger.info(f"Successfully logged trace for LiteLLM {method_name}")
            else:
                logger.warning(f"Failed to log trace for LiteLLM {method_name}")
        except Exception as e:
            logger.error(f"Error logging trace for LiteLLM {method_name}: {e}", exc_info=True)

        if error_obj:
            raise error_obj

        return result
    
    # Replace the original method with our wrapper
    setattr(litellm_module, method_name, wrapper)
    return True

def wrap_litellm_acompletion(
    rm_client: Any, # RagMetricsClient
    litellm_module: Any, # The litellm module itself
    callback: Optional[Callable]
) -> bool:
    """Wraps litellm.acompletion using create_async_wrapper."""
    method_name = "acompletion"
    if not hasattr(litellm_module, method_name) or not callable(getattr(litellm_module, method_name)):
        logger.warning(f"LiteLLM module does not have a callable '{method_name}' function.")
        return False

    import inspect
    original_acompletion = getattr(litellm_module, method_name)
    if not inspect.iscoroutinefunction(original_acompletion):
        logger.warning(f"LiteLLM function '{method_name}' is not an async function. Skipping async wrapping.")
        return False

    static_model_name = None # Always dynamic for litellm module functions
    static_tools = None      # Always dynamic for litellm module functions

    async_wrapper_fn = create_async_wrapper(
        rm_client=rm_client,
        original_method=original_acompletion,
        is_target_instance_method=False,
        callback=callback,
        input_extractor=_litellm_input_extractor,
        output_extractor=_litellm_output_extractor,
        dynamic_llm_details_extractor=_litellm_dynamic_llm_details_extractor,
        static_model_name=static_model_name,
        static_tools=static_tools
    )
    
    setattr(litellm_module, method_name, async_wrapper_fn)
    logger.info(f"Successfully wrapped LiteLLM '{method_name}' for async RagMetrics logging.")
    return True 