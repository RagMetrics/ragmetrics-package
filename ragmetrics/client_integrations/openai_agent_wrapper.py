import time
from typing import Callable, Any, Optional, Dict, Tuple
import logging
import types
import inspect
from .wrapper_utils import create_sync_wrapper, create_async_wrapper

# Assuming these are the correct import paths for OpenAI Agent SDK types
# from agents import Runner # This was for the old wrapper, may not be needed directly here
# from agents.agents.run_config import RunConfig # if used by extractors
# from agents.agents.tools.tool_protocol import ToolDefinition # if used by extractors

logger = logging.getLogger(__name__)

def _openai_agent_input_extractor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """Extracts input context for OpenAI Agent's run_sync/run.
       Logs relevant identifiers like thread_id and assistant_id from kwargs primarily.
       `args` typically only contains `self` for these instance methods.
    """
    # Relevant identifiers are usually in kwargs for run_sync/run
    # e.g., runner.run_sync(thread_id="t_123", assistant_id="asst_abc", instructions="Override")
    # The actual "user message" input is part of the thread, not directly in this call signature.
    # We log the kwargs as a proxy for the run configuration.
    input_context = {}
    if len(args) > 0:
        # args[0] is 'self' (the Runner instance). We don't need to log it here.
        # If there were other positional args for run_sync/run, they'd be args[1:]
        pass 
    input_context.update(kwargs)
    return input_context

def _openai_agent_output_extractor(result: Any) -> Any:
    """Extracts output for OpenAI Agent. Typically the Run object itself."""
    # The result of run_sync/run is usually a `Run` object.
    # We can log it as is, or extract key fields like status, id, output (if available).
    # For now, logging the raw Run object is simple and provides all details.
    if hasattr(result, "to_dict") and callable(result.to_dict):
        try:
            return result.to_dict() # Convert to dict if possible (OpenAI models often have this)
        except Exception:
            pass # Fallback to raw object if to_dict fails
    return result

def _openai_agent_dynamic_llm_details_extractor(call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts dynamic details from OpenAI Agent's run_sync/run kwargs.
       `assistant_id` is crucial. `model`, `tools`, `instructions` can be overrides.
    """
    details = {
        # Model/tools specified in run_sync/run override the Assistant's defaults.
        "model_name": call_kwargs.get("model"), 
        "tools": call_kwargs.get("tools"),
        # tool_choice is not a direct param for run_sync, but for specific steps if using Assistants API v1 style
        # For v2, it's usually managed by the assistant or passed via `tool_configurations`.
        "tool_choice": call_kwargs.get("tool_choice"), 
        "additional_llm_metadata": {}
    }
    # Log other relevant parameters passed to run_sync/run
    # assistant_id is vital, instructions, additional_instructions, etc.
    # These are not LLM model params but control the agent execution.
    agent_run_params = ["assistant_id", "instructions", "additional_instructions", 
                          "max_prompt_tokens", "max_completion_tokens", "truncation_strategy",
                          "temperature", "top_p", "stream"]
    for param in agent_run_params:
        if param in call_kwargs:
            details["additional_llm_metadata"][param] = call_kwargs[param]

    return details

def _get_static_agent_details(runner_instance: Any) -> Dict[str, Any]:
    """Tries to get static details like assistant_id from the Runner instance if assistants v1 style.
       For v2, assistant_id is always passed to run/run_sync.
    """
    # In Assistants API v1, Runner was initialized with assistant_id, thread_id.
    # In v2, Runner is more generic, and these are passed to run/create_thread_and_run.
    # This function is less relevant for v2 Runners unless they are subclassed with static IDs.
    # For now, assume v2 where these are dynamic and handled by the extractor.
    return {
        "static_model_name": None, # Model is on Assistant, not Runner
        "static_tools": None       # Tools are on Assistant, not Runner
    }

def wrap_openai_agent_runner_sync(
    rm_client: Any, # RagMetricsClient
    runner_instance: Any, # This will be an instance of openai.beta.threads.runs.Runs (the client.beta.threads.runs object)
                         # or a similar object that has run_sync, not the top-level client.
                         # Actually, it's more likely the Runner class from `from openai.beta.threads.runs import AssistantEventHandler, OpenAI()`
                         # No, it's `openai.resources.beta.threads.runs.Runs` which has `create`, `retrieve`, `list`, `submit_tool_outputs`
                         # The monitored object is `openai.resources.beta.threads.runs.Runs` if we wrap `create` or `create_and_poll`
                         # OR if we are wrapping a custom Runner class as per user's original setup. The user's test script used `from assistants import Runner`.
                         # Let's assume the user monitors an instance of a class that *has* a `run_sync` method.
    callback: Optional[Callable]
) -> bool:
    """Wraps the `run_sync` method of an OpenAI Agent Runner instance."""
    method_name = "run_sync"
    if not hasattr(runner_instance, method_name) or not callable(getattr(runner_instance, method_name)):
        logger.warning(f"OpenAI Agent Runner instance does not have a callable '{method_name}' method.")
        return False

    original_run_sync = getattr(runner_instance, method_name)
    
    # Static details are less common for v2 agent runners directly on the runner instance.
    # They are usually on the Assistant object itself, or passed dynamically to run_sync.
    static_details = _get_static_agent_details(runner_instance)

    wrapper_fn = create_sync_wrapper(
        rm_client=rm_client,
        original_method=original_run_sync,
        is_target_instance_method=True,
        callback=callback,
        input_extractor=_openai_agent_input_extractor,
        output_extractor=_openai_agent_output_extractor,
        dynamic_llm_details_extractor=_openai_agent_dynamic_llm_details_extractor,
        static_model_name=static_details["static_model_name"],
        static_tools=static_details["static_tools"]
    )
    
    # runner_instance is an instance, so bind the method
    setattr(runner_instance, method_name, types.MethodType(wrapper_fn, runner_instance))
    logger.info(f"Successfully wrapped OpenAI Agent Runner '{method_name}' for RagMetrics logging.")
    return True

def wrap_openai_agent_runner_async(
    rm_client: Any, # RagMetricsClient
    runner_instance: Any, 
    callback: Optional[Callable]
) -> bool:
    """Wraps the `run` (async) method of an OpenAI Agent Runner instance using create_async_wrapper."""
    method_name = "run" 
    if not hasattr(runner_instance, method_name) or not callable(getattr(runner_instance, method_name)):
        logger.warning(f"OpenAI Agent Runner instance does not have a callable '{method_name}' method.")
        return False

    original_run_method = getattr(runner_instance, method_name)

    # Verify it's an async method. For instance methods, check __func__.
    is_truly_async = False
    if inspect.iscoroutinefunction(original_run_method): 
        is_truly_async = True
    elif hasattr(original_run_method, '__func__') and inspect.iscoroutinefunction(original_run_method.__func__):
        is_truly_async = True

    if not is_truly_async:
        logger.warning(f"OpenAI Agent Runner method '{method_name}' is not an async function. Skipping async wrapping.")
        return False

    static_details = _get_static_agent_details(runner_instance) # Reusing the same logic as sync

    async_wrapper_fn = create_async_wrapper(
        rm_client=rm_client,
        original_method=original_run_method,
        is_target_instance_method=True,
        callback=callback,
        input_extractor=_openai_agent_input_extractor,       
        output_extractor=_openai_agent_output_extractor,      
        dynamic_llm_details_extractor=_openai_agent_dynamic_llm_details_extractor,
        static_model_name=static_details["static_model_name"],
        static_tools=static_details["static_tools"]
    )
    
    setattr(runner_instance, method_name, types.MethodType(async_wrapper_fn, runner_instance))
    logger.info(f"Successfully wrapped OpenAI Agent Runner '{method_name}' for async RagMetrics logging.")
    return True

# The old wrap_openai_agent_runner function is removed. 