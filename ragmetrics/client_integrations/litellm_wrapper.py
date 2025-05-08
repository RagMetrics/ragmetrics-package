import time
import types # Not strictly needed if only wrapping module-level functions, but good for consistency
from typing import Callable, Any, Optional, Dict, Tuple
import logging
from .wrapper_utils import create_sync_wrapper, create_async_wrapper

logger = logging.getLogger(__name__)

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
    details = {
        "model_name": call_kwargs.get("model"),
        "tools": call_kwargs.get("tools"),
        "tool_choice": call_kwargs.get("tool_choice"),
        "additional_llm_metadata": {}
    }
    # LiteLLM passes many provider-specific parameters directly in kwargs.
    # We can capture some common ones or let users pass them via 'metadata' kwarg.
    # For example, temperature, max_tokens, etc.
    common_litellm_params = [
        "api_base", "api_key", "api_version", "aws_access_key_id", "aws_region_name", 
        "aws_secret_access_key", "azure_ad_token", "base_url", "custom_llm_provider",
        "drop_params", "extra_body", "frequency_penalty", "function_call", "headers",
        "input_cost_per_token", "litellm_logging_obj", "logit_bias", "logprobs", 
        "max_tokens", "metadata", "mock_response", "n", "num_retries", "optional_params",
        "organization", "output_cost_per_token", "presence_penalty", "project", 
        "proxy", "response_format", "rpm", "safety_settings", "seed", "stop", 
        "stream", "temperature", "timeout", "top_k", "top_p", "tpm", "user"
    ]
    # We should be selective about what goes into additional_llm_metadata to avoid excessive data.
    # Perhaps only a few key ones, or let user specify via main `metadata` kwarg.
    # For now, let's grab a few common ones if present.
    core_params_to_log = ["temperature", "max_tokens", "stream", "user"]
    for param in core_params_to_log:
        if param in call_kwargs:
            details["additional_llm_metadata"][param] = call_kwargs[param]
    
    # LiteLLM has a 'metadata' kwarg for its own logging. We should not conflict.
    # Our 'metadata' for RagMetrics is popped before this extractor is called.
    # If user wants LiteLLM's own metadata also logged by RagMetrics, they should pass it
    # within the RagMetrics 'metadata' dictionary, e.g. metadata={"litellm_own_meta": call_kwargs.get("metadata")}

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
    
    original_completion = getattr(litellm_module, method_name)
    
    # LiteLLM is a module, model/tools are always dynamic (passed in kwargs)
    static_model_name = None
    static_tools = None

    wrapper_fn = create_sync_wrapper(
        rm_client=rm_client,
        original_method=original_completion,
        is_target_instance_method=False,
        callback=callback,
        input_extractor=_litellm_input_extractor,
        output_extractor=_litellm_output_extractor,
        dynamic_llm_details_extractor=_litellm_dynamic_llm_details_extractor,
        static_model_name=static_model_name, # Always None for LiteLLM module methods
        static_tools=static_tools           # Always None for LiteLLM module methods
    )
    
    setattr(litellm_module, method_name, wrapper_fn)
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