import time
import types
from typing import Callable, Any, Optional, Dict, Tuple, List
import logging
import inspect
from ragmetrics.utils import default_output
from .wrapper_utils import create_sync_wrapper, create_async_wrapper

# Set up more verbose logging for debugging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create console handler if not already present
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.debug("Enabled debug logging for OpenAI wrapper")

def is_openai_v1():
    """Check if we're using OpenAI v1.x"""
    try:
        import openai
        openai_version = getattr(openai, '__version__', '0.0.0')
        return openai_version.startswith('1.')
    except ImportError:
        return False

def _extract_openai_call_details(kwargs: Dict[str, Any]) -> Tuple[Optional[List[Any]], Optional[Any], Dict[str, Any], Dict[str, Any]]:
    """Extracts RagMetrics-specific args and prepares kwargs for the actual LLM call.
    Gathers user-supplied metadata and specific OpenAI params (tools, tool_choice) for logging.
    """
    llm_call_kwargs = kwargs.copy() # Work on a copy

    # Extract RagMetrics-specific parameters (remove them from llm_call_kwargs)
    contexts = llm_call_kwargs.pop("contexts", None)
    expected = llm_call_kwargs.pop("expected", None)
    user_call_metadata = llm_call_kwargs.pop("metadata", None) 

    additional_llm_metadata = {}
    if isinstance(user_call_metadata, dict):
        additional_llm_metadata.update(user_call_metadata)
    
    # Only add 'tools' and 'tool_choice' from the llm_call_kwargs to additional_llm_metadata
    # Other llm params (model, temp, etc.) are handled by _openai_chat_dynamic_llm_details_extractor for top-level fields.
    if "tools" in llm_call_kwargs:
        additional_llm_metadata["tools"] = llm_call_kwargs["tools"]
    if "tool_choice" in llm_call_kwargs:
        additional_llm_metadata["tool_choice"] = llm_call_kwargs["tool_choice"]
            
    return contexts, expected, llm_call_kwargs, additional_llm_metadata

def _openai_chat_input_extractor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """Extracts the 'messages' payload from OpenAI chat completion kwargs."""
    return kwargs.get("messages")

def _openai_chat_output_extractor(result: Any) -> Any:
    """Extracts and formats the response from an OpenAI chat completion result.
       Specifically, extracts content and tool calls from the first choice.
    """
    if result and hasattr(result, 'choices') and result.choices:
        message = result.choices[0].message
        response_data = {}
        if message.content:
            response_data["content"] = message.content
        if hasattr(message, 'tool_calls') and message.tool_calls:
            response_data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in message.tool_calls
            ]
        # If there are other fields in message (e.g. role) that should be logged, add them.
        # For now, focusing on content and tool_calls as primary output.
        return response_data if response_data else message # Return raw message if no content/tool_calls
    return result # Fallback to raw result if structure is unexpected

def _openai_chat_dynamic_llm_details_extractor(call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts dynamic LLM details from OpenAI chat completions `create` kwargs."""
    details = {
        "model_name": call_kwargs.get("model"),
        "tools": call_kwargs.get("tools"),
        "tool_choice": call_kwargs.get("tool_choice"),
        "additional_llm_metadata": {}
    }
    # Add model, tools, and tool_choice to additional_llm_metadata as well, if present
    if call_kwargs.get("model"):
        details["additional_llm_metadata"]["model"] = call_kwargs["model"]
    if call_kwargs.get("tools"):
        details["additional_llm_metadata"]["tools"] = call_kwargs["tools"]
    if call_kwargs.get("tool_choice"):
        details["additional_llm_metadata"]["tool_choice"] = call_kwargs["tool_choice"]

    # Add other relevant OpenAI params to additional_llm_metadata if desired for logging
    # For example: temperature, max_tokens, etc.
    known_openai_params = ["frequency_penalty", "logit_bias", "logprobs", "max_tokens", 
                           "n", "presence_penalty", "response_format", "seed", "stop", 
                           "stream", "temperature", "top_p", "user"]
    for param in known_openai_params:
        if param in call_kwargs:
            details["additional_llm_metadata"][param] = call_kwargs[param]
    
    return details

def wrap_openai_chat_completions_create(
    rm_client: Any, # RagMetricsClient
    # client_or_class is the `openai.resources.chat.completions.Completions` or `AsyncCompletions` object (instance)
    # or the `openai.ChatCompletion` class (older SDK versions) - we assume modern SDK structure.
    # For modern SDK, `client.chat.completions` is an instance of `Completions`.
    completions_obj: Any, 
    callback: Optional[Callable]
) -> bool:
    """Wraps the `create` method of OpenAI's Chat Completions object."""
    method_name_to_wrap = "create"
    if not hasattr(completions_obj, method_name_to_wrap) or not callable(getattr(completions_obj, method_name_to_wrap)):
        logger.warning(f"OpenAI Completions object does not have a callable '{method_name_to_wrap}' method.")
        return False

    original_create_method = getattr(completions_obj, method_name_to_wrap)

    # In OpenAI, the model can be set on the client (e.g. `OpenAI(model=...)`)
    # or passed directly to `create(model=...)`.
    # `create_sync_wrapper` handles dynamic overriding static, so we try to get static here.
    # However, the `completions_obj` itself (e.g., `client.chat.completions`) doesn't typically store the model.
    # The model is usually on the main `OpenAI()` client instance.
    # This wrapper receives `client.chat.completions`, not the root `client`.
    # For simplicity, we'll rely on `model` kwarg in `create` or assume user sets it in `rm_client.metadata` or `metadata` kwarg if needed globally.
    # Alternatively, the monitor function could pass the root client if it can determine it.
    static_model_name = None # Cannot easily get from `completions_obj`
    static_tools = None      # Tools are always passed to `create`

    # OpenAI v1's create method has a different parameter binding model than expected.
    # Instead of trying to rebind it with types.MethodType (which causes issues),
    # we'll directly create a wrapper that properly handles the binding.
    
    # Define wrapper with appropriate signature for OpenAI v1 client methods
    def create_wrapper(*args, **kwargs):
        logger.debug("===== RAGMETRICS WRAPPER CALLED =====")
        logger.info(f"RagMetrics: OpenAI v1 chat.completions.create wrapper called")
        
        # Create a copy of kwargs to avoid modifying the original
        openai_kwargs = kwargs.copy()
        
        # Extract RagMetrics-specific parameters from kwargs
        contexts = openai_kwargs.pop("contexts", None)
        expected = openai_kwargs.pop("expected", None)
        metadata = openai_kwargs.pop("metadata", {}).copy() if "metadata" in openai_kwargs else {}
        force_new_conversation = openai_kwargs.pop("force_new_conversation", False)
        
        logger.debug(f"Extracted contexts: {contexts}")
        logger.debug(f"Cleaned kwargs keys: {list(openai_kwargs.keys())}")
        
        # Get important parameters for the trace (not for metadata)
        model_name = openai_kwargs.get("model", None)
        messages = openai_kwargs.get("messages", [])
        tools = openai_kwargs.get("tools", None)
        tool_choice = openai_kwargs.get("tool_choice", None)
        
        # Measure request time
        start_time = time.time()
        
        # Call the original create method
        try:
            # Call original method with all original args and cleaned kwargs (without contexts)
            response = original_create_method(*args, **openai_kwargs)
            error = None
        except Exception as e:
            error = e
            logger.error(f"Error in OpenAI create call: {str(e)}")
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Don't log if error and no response
            if error is not None and 'response' not in locals():
                return
            
            # Process callback if provided
            callback_result = None
            if callback and 'response' in locals():
                try:
                    callback_result = callback(messages, response)
                except Exception as e:
                    logger.error(f"Callback error in OpenAI v1 wrapper: {e}")
            
            # Log trace with RagMetrics
            rm_client._log_trace(
                input_messages=messages,
                response=response if 'response' in locals() else None,
                metadata_llm=metadata,
                contexts=contexts,
                expected=expected,
                duration=duration,
                model_name=model_name,
                tools=tools,
                tool_choice=tool_choice,
                callback_result=callback_result,
                force_new_conversation=force_new_conversation,
                error=error
            )
            
        return response
    
    # Replace the original method. In OpenAI v1, this is cleaner than using types.MethodType
    setattr(completions_obj, method_name_to_wrap, create_wrapper)
    logger.info(f"Successfully wrapped OpenAI Completions '{method_name_to_wrap}' for RagMetrics logging.")
    return True

# Async wrapper for OpenAI chat completions (achronous create method)
# Placeholder: Needs a proper create_async_wrapper utility.

def wrap_openai_chat_completions_acreate(
    rm_client: Any, # RagMetricsClient
    async_completions_obj: Any, # e.g., openai.resources.chat.AsyncCompletions
    callback: Optional[Callable]
) -> bool:
    """Wraps the `create` method of OpenAI's Async Chat Completions object."""
    method_name_to_wrap = "create"
    if not hasattr(async_completions_obj, method_name_to_wrap) or \
       not callable(getattr(async_completions_obj, method_name_to_wrap)):
        logger.warning(f"OpenAI AsyncCompletions object does not have a callable '{method_name_to_wrap}' method.")
        return False

    original_acreate_method = getattr(async_completions_obj, method_name_to_wrap)
    
    is_truly_async = False
    if inspect.iscoroutinefunction(original_acreate_method):
        is_truly_async = True
    elif hasattr(original_acreate_method, '__func__') and inspect.iscoroutinefunction(original_acreate_method.__func__):
        is_truly_async = True

    if not is_truly_async:
        logger.warning(f"OpenAI AsyncCompletions method '{method_name_to_wrap}' is not an async function. Skipping async wrapping.")
        return False

    static_model_name = None
    static_tools = None

    # Define async wrapper with appropriate signature for OpenAI v1 async client methods
    async def acreate_wrapper(*args, **kwargs):
        logger.debug("===== RAGMETRICS ASYNC WRAPPER CALLED =====")
        logger.info(f"RagMetrics: OpenAI v1 chat.completions.acreate wrapper called")
        
        # Create a copy of kwargs to avoid modifying the original
        openai_kwargs = kwargs.copy()
        
        # Extract RagMetrics-specific parameters from kwargs
        contexts = openai_kwargs.pop("contexts", None)
        expected = openai_kwargs.pop("expected", None)
        metadata = openai_kwargs.pop("metadata", {}).copy() if "metadata" in openai_kwargs else {}
        force_new_conversation = openai_kwargs.pop("force_new_conversation", False)
        
        logger.debug(f"Extracted contexts: {contexts}")
        logger.debug(f"Cleaned kwargs keys: {list(openai_kwargs.keys())}")
        
        # Get important parameters for the trace (not for metadata)
        model_name = openai_kwargs.get("model", None)
        messages = openai_kwargs.get("messages", [])
        tools = openai_kwargs.get("tools", None)
        tool_choice = openai_kwargs.get("tool_choice", None)
        
        # Measure request time
        start_time = time.time()
        
        # Call the original async create method
        try:
            # Call original method with all original args and cleaned kwargs (without contexts)
            response = await original_acreate_method(*args, **openai_kwargs)
            error = None
        except Exception as e:
            error = e
            logger.error(f"Error in OpenAI acreate call: {str(e)}")
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Don't log if error and no response
            if error is not None and 'response' not in locals():
                return
            
            # Process callback if provided
            callback_result = None
            if callback and 'response' in locals():
                try:
                    callback_result = callback(messages, response)
                except Exception as e:
                    logger.error(f"Callback error in OpenAI v1 async wrapper: {e}")
            
            # Log trace with RagMetrics (using async version if available)
            try:
                if hasattr(rm_client, '_alog_trace'):
                    await rm_client._alog_trace(
                        input_messages=messages,
                        response=response if 'response' in locals() else None,
                        metadata_llm=metadata,
                        contexts=contexts,
                        expected=expected,
                        duration=duration,
                        model_name=model_name,
                        tools=tools,
                        tool_choice=tool_choice,
                        callback_result=callback_result,
                        force_new_conversation=force_new_conversation,
                        error=error
                    )
                else:
                    # Fallback to sync logging if async not available
                    rm_client._log_trace(
                        input_messages=messages,
                        response=response if 'response' in locals() else None,
                        metadata_llm=metadata,
                        contexts=contexts,
                        expected=expected,
                        duration=duration,
                        model_name=model_name,
                        tools=tools,
                        tool_choice=tool_choice,
                        callback_result=callback_result,
                        force_new_conversation=force_new_conversation,
                        error=error
                    )
            except Exception as e:
                logger.error(f"Error logging trace in async wrapper: {e}")
            
        return response
    
    # Replace the original method
    setattr(async_completions_obj, method_name_to_wrap, acreate_wrapper)
    logger.info(f"Successfully wrapped OpenAI AsyncCompletions '{method_name_to_wrap}' for async RagMetrics logging.")
    return True

# The old top-level wrap_openai_chat_completions function is removed as it's superseded by the specific create/acreate wrappers.
# If it was meant for older SDK versions, that logic would need to be re-evaluated or placed in a separate legacy handler. 

def wrap_openai_module_v1(rm_client: Any, openai_module: Any, callback: Optional[Callable]) -> bool:
    """
    Wrap the OpenAI v1.x module for monitoring.
    This is for cases when the module itself is passed to monitor() rather than a client instance.
    
    Args:
        rm_client: RagMetricsClient instance
        openai_module: The openai module (not a class instance)
        callback: Optional callback function for custom processing
        
    Returns:
        bool: True if wrapping was successful
    """
    if not is_openai_v1():
        logger.debug("OpenAI v1.x module wrapper skipped - not running on v1.x")
        return False
        
    # Create an OpenAI client if the module is passed
    if not hasattr(openai_module, 'OpenAI'):
        logger.warning("OpenAI module does not have OpenAI client class")
        return False

    # Create a client instance
    try:
        client = openai_module.OpenAI()
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            # Now wrap the client's chat.completions.create method
            success = wrap_openai_chat_completions_create(rm_client, client.chat.completions, callback)
            if success:
                logger.info("Successfully wrapped OpenAI v1.x module by creating and wrapping a client instance")
                # Store the wrapped client on the module for future use
                if not hasattr(openai_module, '_ragmetrics_wrapped_client'):
                    setattr(openai_module, '_ragmetrics_wrapped_client', client)
                return True
    except Exception as e:
        logger.error(f"Error wrapping OpenAI v1.x module: {e}")
        
    return False 