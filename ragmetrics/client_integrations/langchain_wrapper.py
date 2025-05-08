import time
import types
from typing import Callable, Any, Optional, Dict, Tuple
import logging
from ragmetrics.utils import default_input, default_output
from .wrapper_utils import create_sync_wrapper, create_async_wrapper

logger = logging.getLogger(__name__)

def _langchain_input_extractor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """Extracts input for LangChain's invoke/ainvoke.
       Assumes input is typically the first positional argument."""
    if args:
        return args[0] # Now args[0] will be the actual first user arg for instance calls
    return kwargs.get("input", None) # Fallback for named input

def _langchain_output_extractor(result: Any) -> Any:
    """Extracts output for LangChain. Typically the result itself."""
    return result

def _langchain_dynamic_llm_details_extractor(call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts dynamic LLM details from LangChain's invoke/ainvoke kwargs."""
    details = {
        "additional_llm_metadata": {},
        "model_name": None, # Usually not in invoke kwargs
        "tools": None,      # Usually not in invoke kwargs
        "tool_choice": None # Usually not in invoke kwargs
    }
    if "config" in call_kwargs:
        details["additional_llm_metadata"]["langchain_config"] = call_kwargs["config"]
    
    # If LangChain starts passing model/tools in invoke's kwargs, extract them here.
    # E.g., details["model_name"] = call_kwargs.get("model")
    return details

def wrap_langchain_invoke(
    rm_client: Any, # RagMetricsClient
    client_or_class: Any, 
    callback: Optional[Callable]
) -> bool:
    """Wraps the LangChain client's `invoke` method."""
    if not hasattr(client_or_class, "invoke") or not callable(getattr(client_or_class, "invoke")):
        # TODO: Log a warning if invoke is not found
        return False

    original_invoke = getattr(client_or_class, "invoke")
    
    # For LangChain, model_name is often an attribute of the runnable/chain instance itself,
    # rather than passed in `invoke`. Attempt to get it if available.
    # This is a best-effort guess; specific chains might store it differently or not at all.
    static_model_name = None
    if not isinstance(client_or_class, type): # If it's an instance
        # Common attribute names for model in LangChain components (LCEL Runnables)
        possible_model_attrs = ["model", "llm", "client"]
        for attr_name in possible_model_attrs:
            if hasattr(client_or_class, attr_name):
                model_attr = getattr(client_or_class, attr_name)
                if isinstance(model_attr, str):
                    static_model_name = model_attr
                    break
                if hasattr(model_attr, "model_name") and isinstance(getattr(model_attr, "model_name"), str):
                    static_model_name = getattr(model_attr, "model_name")
                    break
                if hasattr(model_attr, "model") and isinstance(getattr(model_attr, "model"), str): # e.g. ChatOpenAI has .model
                    static_model_name = getattr(model_attr, "model")
                    break

    # TODO: Add similar logic for static_tools if applicable and identifiable

    wrapper_fn = create_sync_wrapper(
        rm_client=rm_client,
        original_method=original_invoke,
        is_target_instance_method=not isinstance(client_or_class, type),
        callback=callback,
        input_extractor=_langchain_input_extractor, # Using default_input might also work
        output_extractor=_langchain_output_extractor, # Using default_output might also work
        dynamic_llm_details_extractor=_langchain_dynamic_llm_details_extractor,
        static_model_name=static_model_name
        # static_tools=None # Add if identifiable
    )

    if isinstance(client_or_class, type):
        setattr(client_or_class, "invoke", wrapper_fn)
    else:
        setattr(client_or_class, "invoke", types.MethodType(wrapper_fn, client_or_class))
    return True

async def _dummy_async_langchain_wrapper(*args, **kwargs):
    # This is a placeholder for the actual async wrapper implementation
    # It should call the original async method and log the trace asynchronously
    # For now, it just indicates that the async wrapping is not yet fully implemented
    logger.warning("Async LangChain wrapper called, but not fully implemented for logging.")
    # In a real scenario, you'd fetch the original method and call it:
    # original_method = getattr(self, "_original_ainvoke_for_" + id(self)) 
    # For testing, let's assume there is an original ainvoke to call if this dummy is set
    if hasattr(args[0], '_original_ainvoke_method'): # args[0] would be self (the client_or_class instance)
        return await args[0]._original_ainvoke_method(*(args[1:]), **kwargs)
    raise NotImplementedError("Async LangChain original method not found for dummy wrapper.")

def wrap_langchain_ainvoke(
    rm_client: Any, # RagMetricsClient
    client_or_class: Any, 
    callback: Optional[Callable]
) -> bool:
    """Wraps the LangChain client's `ainvoke` method using create_async_wrapper."""
    method_name_to_wrap = "ainvoke"
    if not hasattr(client_or_class, method_name_to_wrap) or not callable(getattr(client_or_class, method_name_to_wrap)):
        logger.warning(f"LangChain object does not have a callable '{method_name_to_wrap}' method.")
        return False
    
    # Ensure the method is actually async
    import inspect
    original_ainvoke = getattr(client_or_class, method_name_to_wrap)
    if not inspect.iscoroutinefunction(original_ainvoke) and not inspect.isasyncgenfunction(original_ainvoke):
        # If it's an instance method that's async, inspect.iscoroutinefunction might be false for the bound method directly.
        # We might need to check original_ainvoke.__func__ if it's a bound method.
        # However, getattr on a classmethod or staticmethod that is async should return true for iscoroutinefunction.
        # For an instance, original_ainvoke is already a bound method. If its __func__ is a coroutine function, it's async.
        is_truly_async = False
        if hasattr(original_ainvoke, '__func__'): # Bound method
            if inspect.iscoroutinefunction(original_ainvoke.__func__):
                is_truly_async = True
        if not is_truly_async:
             logger.warning(f"LangChain method '{method_name_to_wrap}' is not an async function. Skipping async wrapping.")
             return False

    # Static model name extraction (same as sync version)
    static_model_name = None
    if not isinstance(client_or_class, type):
        possible_model_attrs = ["model", "llm", "client"]
        for attr_name in possible_model_attrs:
            if hasattr(client_or_class, attr_name):
                model_attr = getattr(client_or_class, attr_name)
                if isinstance(model_attr, str):
                    static_model_name = model_attr
                    break
                if hasattr(model_attr, "model_name") and isinstance(getattr(model_attr, "model_name"), str):
                    static_model_name = getattr(model_attr, "model_name")
                    break
                if hasattr(model_attr, "model") and isinstance(getattr(model_attr, "model"), str):
                    static_model_name = getattr(model_attr, "model")
                    break
    
    # Create the async wrapper
    # The extractors (_langchain_input_extractor, etc.) are assumed to be compatible with async calls (i.e., they don't block)
    async_wrapper_fn = create_async_wrapper(
        rm_client=rm_client,
        original_method=original_ainvoke, 
        is_target_instance_method=not isinstance(client_or_class, type),
        callback=callback,
        input_extractor=_langchain_input_extractor,
        output_extractor=_langchain_output_extractor,
        dynamic_llm_details_extractor=_langchain_dynamic_llm_details_extractor,
        static_model_name=static_model_name
    )

    # Apply the wrapper
    if isinstance(client_or_class, type):
        setattr(client_or_class, method_name_to_wrap, async_wrapper_fn) # For class/static methods
    else:
        setattr(client_or_class, method_name_to_wrap, types.MethodType(async_wrapper_fn, client_or_class)) # For instance methods
    
    logger.info(f"Successfully wrapped LangChain '{method_name_to_wrap}' for async RagMetrics logging.")
    return True

# Placeholder for async LangChain wrapper if/when _original_async_llm_invoke supports it for LangChain
# def wrap_langchain_ainvoke(ragmetrics_client_instance: Any, client_to_wrap: Any, original_ainvoke_method: Callable, callback_to_use: Callable):
#     async def ainvoke_async_wrapper(*args_call, **kwargs_call):
#         # ... similar logic to invoke_sync_wrapper but async ...
#         pass
#     try:
#         if isinstance(client_to_wrap, type):
#             setattr(client_to_wrap, "ainvoke", ainvoke_async_wrapper)
#         else:
#             client_to_wrap.ainvoke = types.MethodType(ainvoke_async_wrapper, client_to_wrap)
#         logger.info("Successfully wrapped LangChain client .ainvoke() for RagMetrics.")
#         return True
#     except Exception as e:
#         logger.error(f"Failed to wrap LangChain client .ainvoke(): {e}")
#         return False 