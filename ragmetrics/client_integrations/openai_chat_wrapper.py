import time
import types
from typing import Callable, Any, Optional, Dict, Tuple, List
import logging
import inspect
from ragmetrics.utils import default_output
from .wrapper_utils import create_sync_wrapper, create_async_wrapper

logger = logging.getLogger(__name__)

def _extract_openai_call_details(kwargs: Dict[str, Any]) -> Tuple[Optional[List[Any]], Optional[Any], Dict[str, Any], Dict[str, Any]]:
    """Extracts RagMetrics-specific args and prepares kwargs for the actual LLM call.
    Gathers user-supplied metadata and specific OpenAI params (tools, tool_choice) for logging.
    """
    llm_call_kwargs = kwargs.copy() # Work on a copy

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

    wrapper_fn = create_sync_wrapper(
        rm_client=rm_client,
        original_method=original_create_method,
        is_target_instance_method=True, # completions_obj is an instance
        callback=callback,
        input_extractor=_openai_chat_input_extractor,
        output_extractor=_openai_chat_output_extractor,
        dynamic_llm_details_extractor=_openai_chat_dynamic_llm_details_extractor,
        static_model_name=static_model_name, # Typically None here, relies on dynamic or higher-level config
        static_tools=static_tools # Tools are dynamic for chat completions
    )
    
    # `completions_obj` (e.g., `client.chat.completions`) is an instance, so we bind the method.
    setattr(completions_obj, method_name_to_wrap, types.MethodType(wrapper_fn, completions_obj))
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

    async_wrapper_fn = create_async_wrapper(
        rm_client=rm_client,
        original_method=original_acreate_method,
        is_target_instance_method=True, # async_completions_obj is an instance
        callback=callback,
        input_extractor=_openai_chat_input_extractor,
        output_extractor=_openai_chat_output_extractor,
        dynamic_llm_details_extractor=_openai_chat_dynamic_llm_details_extractor,
        static_model_name=static_model_name,
        static_tools=static_tools
    )
    
    setattr(async_completions_obj, method_name_to_wrap, types.MethodType(async_wrapper_fn, async_completions_obj))
    logger.info(f"Successfully wrapped OpenAI AsyncCompletions '{method_name_to_wrap}' for async RagMetrics logging.")
    return True

# The old top-level wrap_openai_chat_completions function is removed as it's superseded by the specific create/acreate wrappers.
# If it was meant for older SDK versions, that logic would need to be re-evaluated or placed in a separate legacy handler. 