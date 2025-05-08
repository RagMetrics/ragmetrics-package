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

# Helper to safely check for Runner type if openai-agents is optional
try:
    from agents import Runner as OpenAIAgentRunner # Alias to avoid name clash if used elsewhere
except ImportError:
    OpenAIAgentRunner = None

# Helper to safely check for LiteLLM module if optional
try:
    import litellm
except ImportError:
    litellm = None

INTEGRATION_REGISTRY = [
    {
        "name": "OpenAI API Client (Chat Completions)",
        # Checks if it's a modern OpenAI client (sync or async) that has chat.completions
        "client_type_check": lambda client: hasattr(client, "chat") and hasattr(client.chat, "completions"),
        "target_object_path": "chat.completions", # Path to client.chat.completions
        # async_target_object_path is the same as target_object_path for OpenAI v1.x SDK
        "methods_to_wrap": {
            "create": wrap_openai_chat_completions_create,
        },
        "async_methods_to_wrap": {
            "create": wrap_openai_chat_completions_acreate,
        }
    },
    {
        "name": "OpenAI Agent SDK Runner",
        # Check if the client is an instance of the openai-agents Runner
        "client_type_check": lambda client: OpenAIAgentRunner is not None and isinstance(client, OpenAIAgentRunner),
        "target_object_path": None, # The client instance itself is the target
        "methods_to_wrap": {
            "run_sync": wrap_openai_agent_runner_sync,
        },
        "async_methods_to_wrap": {
            "run": wrap_openai_agent_runner_async,
        }
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
                return integration
        except Exception as e:
             logger.debug(f"RagMetrics: Error checking integration '{integration['name']}' for client type {type(client).__name__}: {e}")
             continue
    return None

__all__ = [
    # Specific wrappers can be exposed if users need to apply them manually, though `monitor` is preferred.
    "wrap_openai_chat_completions_create", "wrap_openai_chat_completions_acreate",
    "wrap_langchain_invoke", "wrap_langchain_ainvoke",
    "wrap_litellm_completion", "wrap_litellm_acompletion",
    "wrap_openai_agent_runner_sync", "wrap_openai_agent_runner_async",
    "find_integration",
    "INTEGRATION_REGISTRY"
] 