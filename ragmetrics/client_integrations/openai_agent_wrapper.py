from typing import Callable, Any, Optional, Dict, Tuple
import logging
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
       For class methods, the first arg is typically the agent to run.
    """
    # For run_sync as class method, args likely contains: (agent, message, **kwargs)
    logger.debug(f"OpenAI Agent input extractor called with args: {args[:1]} and kwargs keys: {list(kwargs.keys())}")
    
    input_context = {}
    
    # Get the agent info if available (typically the first arg for class methods)
    if len(args) > 0:
        # For class methods, first arg is typically the agent
        agent = args[0]
        logger.debug(f"Agent object type: {type(agent).__name__}")
        if hasattr(agent, 'name'):
            input_context['agent_name'] = agent.name
        if hasattr(agent, 'id'):
            input_context['agent_id'] = agent.id
        if hasattr(agent, 'instructions') and isinstance(agent.instructions, str):
            # Include just a preview of instructions to avoid large payloads
            input_context['agent_instructions_preview'] = agent.instructions[:100] + ('...' if len(agent.instructions) > 100 else '')
    
    # For class methods, second arg is typically the user message
    if len(args) > 1:
        logger.debug(f"Second arg type: {type(args[1]).__name__}")
        if isinstance(args[1], str):
            input_context['message'] = args[1]
    
    # Include any kwargs like thread_id, run_config, etc.
    input_context.update(kwargs)
    
    return input_context

def _openai_agent_output_extractor(result: Any) -> Any:
    """Extracts output for OpenAI Agent. Typically the Run object itself."""
    logger.debug(f"OpenAI Agent output extractor called with result type: {type(result).__name__}")
    
    # The result of run_sync/run is usually a `Run` object.
    # We can log it as is, or extract key fields like status, id, output (if available).
    # For now, logging the raw Run object is simple and provides all details.
    if hasattr(result, "to_dict") and callable(result.to_dict):
        try:
            return result.to_dict() # Convert to dict if possible (OpenAI models often have this)
        except Exception as e:
            logger.debug(f"Error calling to_dict(): {e}")
            pass # Fallback to raw object if to_dict fails
    
    # Extract specific fields for cleaner logging if to_dict isn't available
    output = {}
    if hasattr(result, "final_output"):
        # If final_output is a Pydantic model, convert it to dict
        final_output = result.final_output
        if hasattr(final_output, "model_dump"):
            output["final_output"] = final_output.model_dump()
        else:
            output["final_output"] = final_output
    if hasattr(result, "status"):
        output["status"] = result.status
    if hasattr(result, "id"):
        output["id"] = result.id
    
    # If we found any fields, return them, otherwise return the whole result
    return output if output else result

def _openai_agent_dynamic_llm_details_extractor(call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts dynamic details from OpenAI Agent's run_sync/run kwargs.
       `assistant_id` is crucial. `model`, `tools`, `instructions` can be overrides.
    """
    logger.debug(f"OpenAI Agent dynamic details extractor called with kwargs keys: {list(call_kwargs.keys())}")
    
    details = {
        # Model/tools specified in run_sync/run override the Assistant's defaults.
        "model_name": call_kwargs.get("model"), 
        "tools": call_kwargs.get("tools"),
        # tool_choice is not a direct param for run_sync, but for specific steps if using Assistants API v1 style
        # For v2, it's usually managed by the assistant or passed via `tool_configurations`.
        "tool_choice": call_kwargs.get("tool_choice"), 
        "additional_llm_metadata": {}
    }
    
    # If config is present, extract metadata from it
    config = call_kwargs.get("config", {})
    if config and hasattr(config, "metadata") and config.metadata:
        details["additional_llm_metadata"]["run_config_metadata"] = config.metadata
    
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

def _get_static_agent_details(runner_class_or_instance: Any) -> Dict[str, Any]:
    """Tries to get static details from the Runner class if available.
       For most cases with the OpenAI Agent SDK, these details are passed dynamically.
    """
    # For OpenAI Agents SDK, class is generic and doesn't have config
    return {
        "static_model_name": None, 
        "static_tools": None
    }

def wrap_openai_agent_runner_sync(
    rm_client: Any, # RagMetricsClient
    runner_class: Any, # This should be the Runner class itself
    callback: Optional[Callable]
) -> bool:
    """Wraps the `run_sync` class method of the OpenAI Agent Runner class."""
    method_name = "run_sync"
    
    # Log more details about the runner class
    logger.info(f"Attempting to wrap {method_name} on {runner_class.__name__ if hasattr(runner_class, '__name__') else type(runner_class).__name__}")
    logger.info(f"Runner class module: {runner_class.__module__ if hasattr(runner_class, '__module__') else 'unknown'}")
    
    if not hasattr(runner_class, method_name) or not callable(getattr(runner_class, method_name)):
        logger.warning(f"OpenAI Agent Runner class does not have a callable '{method_name}' method.")
        return False

    original_run_sync = getattr(runner_class, method_name)
    logger.info(f"Original {method_name} function: {original_run_sync} from module {original_run_sync.__module__ if hasattr(original_run_sync, '__module__') else 'unknown'}")
    
    static_details = _get_static_agent_details(runner_class)

    # Create a wrapper that can handle being called as a class method
    wrapper_fn = create_sync_wrapper(
        rm_client=rm_client,
        original_method=original_run_sync,
        is_target_instance_method=False,  # Class method, not instance method
        callback=callback,
        input_extractor=_openai_agent_input_extractor,
        output_extractor=_openai_agent_output_extractor,
        dynamic_llm_details_extractor=_openai_agent_dynamic_llm_details_extractor,
        static_model_name=static_details["static_model_name"],
        static_tools=static_details["static_tools"]
    )
    
    # Add a debug identifier to the wrapper function
    wrapper_fn.__ragmetrics_wrapped__ = True
    wrapper_fn.__name__ = f"ragmetrics_wrapped_{method_name}"
    
    # Replace the class method
    setattr(runner_class, method_name, wrapper_fn)
    
    # Verify the replacement worked
    current_method = getattr(runner_class, method_name)
    if hasattr(current_method, '__ragmetrics_wrapped__'):
        logger.info(f"Successfully wrapped OpenAI Agent Runner class method '{method_name}' for RagMetrics logging.")
        return True
    else:
        logger.error(f"Failed to wrap {method_name} - method was not replaced properly")
        return False

def wrap_openai_agent_runner_async(
    rm_client: Any, # RagMetricsClient
    runner_class: Any, # Should be the Runner class itself
    callback: Optional[Callable]
) -> bool:
    """Wraps the `run` (async) class method of the OpenAI Agent Runner class."""
    method_name = "run" 
    
    # Log more details about the runner class
    logger.info(f"Attempting to wrap async {method_name} on {runner_class.__name__ if hasattr(runner_class, '__name__') else type(runner_class).__name__}")
    logger.info(f"Runner class module: {runner_class.__module__ if hasattr(runner_class, '__module__') else 'unknown'}")
    
    if not hasattr(runner_class, method_name) or not callable(getattr(runner_class, method_name)):
        logger.warning(f"OpenAI Agent Runner class does not have a callable '{method_name}' method.")
        return False

    original_run_method = getattr(runner_class, method_name)
    logger.info(f"Original async {method_name} function: {original_run_method} from module {original_run_method.__module__ if hasattr(original_run_method, '__module__') else 'unknown'}")

    # Verify it's an async method
    is_truly_async = False
    if inspect.iscoroutinefunction(original_run_method): 
        is_truly_async = True
    elif hasattr(original_run_method, '__func__') and inspect.iscoroutinefunction(original_run_method.__func__):
        is_truly_async = True

    if not is_truly_async:
        logger.warning(f"OpenAI Agent Runner method '{method_name}' is not an async function. Skipping async wrapping.")
        return False

    static_details = _get_static_agent_details(runner_class)

    async_wrapper_fn = create_async_wrapper(
        rm_client=rm_client,
        original_method=original_run_method,
        is_target_instance_method=False,  # Class method, not instance method
        callback=callback,
        input_extractor=_openai_agent_input_extractor,       
        output_extractor=_openai_agent_output_extractor,      
        dynamic_llm_details_extractor=_openai_agent_dynamic_llm_details_extractor,
        static_model_name=static_details["static_model_name"],
        static_tools=static_details["static_tools"]
    )
    
    # Add a debug identifier to the wrapper function
    async_wrapper_fn.__ragmetrics_wrapped__ = True
    async_wrapper_fn.__name__ = f"ragmetrics_wrapped_{method_name}"
    
    # Replace the class method with our wrapper
    setattr(runner_class, method_name, async_wrapper_fn)
    
    # Verify the replacement worked
    current_method = getattr(runner_class, method_name)
    if hasattr(current_method, '__ragmetrics_wrapped__'):
        logger.info(f"Successfully wrapped OpenAI Agent Runner class method '{method_name}' for async RagMetrics logging.")
        return True
    else:
        logger.error(f"Failed to wrap async {method_name} - method was not replaced properly")
        return False

# The old wrap_openai_agent_runner function is removed. 