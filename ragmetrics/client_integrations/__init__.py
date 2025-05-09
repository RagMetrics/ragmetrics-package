# This file makes client_integrations a Python package
# It can also be used for a registry or to expose wrapper functions
import logging
import types # For isinstance checks if needed
from typing import Any # Import Any

logger = logging.getLogger(__name__)

# --- Import specific wrapper functions --- 
from .openai_chat_wrapper import wrap_openai_chat_completions_create, wrap_openai_chat_completions_acreate
from .langchain_wrapper import wrap_langchain_invoke, wrap_langchain_ainvoke
from .litellm_wrapper import wrap_litellm_completion, wrap_litellm_acompletion
from .openai_agent_wrapper import wrap_openai_agent_runner_sync, wrap_openai_agent_runner_async

# Import the OpenAI Agents trace processor
try:
    from .openai_agents_trace_processor import register_trace_processor, AGENTS_TRACING_AVAILABLE
except ImportError:
    logger.warning("Failed to import openai_agents_trace_processor, tracing integration will not be available")
    register_trace_processor = None
    AGENTS_TRACING_AVAILABLE = False

# Helper to safely check for Runner type if openai-agents is optional
# Try multiple import paths for better compatibility
OpenAIAgentRunner = None
try:
    from agents import Runner as OpenAIAgentRunner  # First attempt - direct import
    logger.debug("Successfully imported Runner from agents")
except ImportError:
    try:
        from agents.run import Runner as OpenAIAgentRunner  # Second attempt - from run module
        logger.debug("Successfully imported Runner from agents.run")
    except ImportError:
        logger.debug("OpenAI Agents SDK not available or Runner not found")
        OpenAIAgentRunner = None

# Helper to safely check for LiteLLM module if optional
try:
    import litellm
except ImportError:
    litellm = None

# Helper to safely import OpenAI and AsyncOpenAI from openai package v1.0+
try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    OpenAI = None
    AsyncOpenAI = None

INTEGRATION_REGISTRY = [
    {
        "name": "OpenAI API Client (SYNC Chat Completions)",
        "client_type_check": lambda client: OpenAI is not None and isinstance(client, OpenAI) and \
                                         hasattr(client, "chat") and hasattr(client.chat, "completions"),
        "target_object_path": "chat.completions",
        "methods_to_wrap": {
            "create": wrap_openai_chat_completions_create,
        },
        # No async_methods_to_wrap for the purely sync client
    },
    {
        "name": "OpenAI API Client (ASYNC Chat Completions)",
        "client_type_check": lambda client: AsyncOpenAI is not None and isinstance(client, AsyncOpenAI) and \
                                         hasattr(client, "chat") and hasattr(client.chat, "completions"),
        "target_object_path": "chat.completions", # client.chat.completions will be AsyncCompletions
        "async_methods_to_wrap": {
            "create": wrap_openai_chat_completions_acreate,
        }
        # No methods_to_wrap for 'create' as we are targeting the async 'create'
    },
    {
        "name": "OpenAI Agent SDK Runner",
        # Check if the client is the Runner class itself (not an instance)
        "client_type_check": lambda client: OpenAIAgentRunner is not None and (
            client is OpenAIAgentRunner or  # Direct reference check
            (hasattr(client, '__name__') and client.__name__ == 'Runner' and  # Class name check
             (hasattr(client, '__module__') and (
                 client.__module__ == 'agents' or  # Check from direct import
                 client.__module__ == 'agents.run' or  # Check from agents.run import
                 client.__module__.startswith('agents.') # Check for any other agents submodule
             ))
            )
        ),
        "target_object_path": None, # The class itself is the target
        "methods_to_wrap": {
            "run_sync": wrap_openai_agent_runner_sync,
        },
        "async_methods_to_wrap": {
            "run": wrap_openai_agent_runner_async,
        },
        "is_class_method": True,  # Flag to indicate we're wrapping class methods, not instance methods
        "register_trace_processor": register_trace_processor  # Function to register the trace processor
    },
    {
        "name": "LangChain Runnable/Client",
        # Check for common LangChain runnable attributes/methods.
        # isinstance(client, Runnable) would be ideal if Runnable is easily importable and universal.
        "client_type_check": lambda client: hasattr(client, "invoke") or hasattr(client, "ainvoke"),
        "target_object_path": None, # The client (runnable instance or class) is the target
        "methods_to_wrap": {
            "invoke": wrap_langchain_invoke,
        },
        "async_methods_to_wrap": {
            "ainvoke": wrap_langchain_ainvoke,
        }
    },
    {
        "name": "LiteLLM Module",
        # Check if the client is the LiteLLM module itself
        "client_type_check": lambda client: litellm is not None and client is litellm,
        "target_object_path": None, # The module itself is the target
        "methods_to_wrap": {
            "completion": wrap_litellm_completion,
            # Potentially others like `embedding` if we add wrappers for them
        },
        "async_methods_to_wrap": {
            "acompletion": wrap_litellm_acompletion,
            # "aembedding": wrap_litellm_aembedding,
        }
    },
]

def find_integration(client: Any):
    """Finds the first matching integration in the registry."""
    for integration in INTEGRATION_REGISTRY:
        try:
            if integration["client_type_check"](client):
                logger.debug(f"RagMetrics: Found matching integration for client: {integration['name']}")
                
                # If this is the OpenAI Agents Runner, also register the trace processor
                if AGENTS_TRACING_AVAILABLE and "register_trace_processor" in integration and integration["register_trace_processor"]:
                    # Import ragmetrics_client here to avoid circular imports
                    from ragmetrics.api import ragmetrics_client
                    logger.info("Registering OpenAI Agents trace processor")
                    success = integration["register_trace_processor"](ragmetrics_client)
                    if success:
                        logger.info("Successfully registered OpenAI Agents trace processor")
                    else:
                        logger.warning("Failed to register OpenAI Agents trace processor")
                
                return integration
        except Exception as e:
             logger.debug(f"RagMetrics: Error checking integration '{integration['name']}' for client type {type(client).__name__}: {e}")
             continue
    
    # Log class info for debugging if no match found
    if isinstance(client, type):  # If it's a class
        logger.debug(f"No integration found for class: {client.__name__} from module {client.__module__ if hasattr(client, '__module__') else 'unknown'}")
    elif hasattr(client, '__class__'):
        logger.debug(f"No integration found for instance type: {client.__class__.__name__}")
    else:
        logger.debug(f"No integration found for object: {type(client).__name__}")
    
    return None

__all__ = [
    # Specific wrappers can be exposed if users need to apply them manually, though `monitor` is preferred.
    "wrap_openai_chat_completions_create", "wrap_openai_chat_completions_acreate",
    "wrap_langchain_invoke", "wrap_langchain_ainvoke",
    "wrap_litellm_completion", "wrap_litellm_acompletion",
    "wrap_openai_agent_runner_sync", "wrap_openai_agent_runner_async",
    "find_integration",
    "INTEGRATION_REGISTRY",
    # OpenAI Agents trace processor
    "register_trace_processor", "AGENTS_TRACING_AVAILABLE"
] 