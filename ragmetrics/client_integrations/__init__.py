# This file makes client_integrations a Python package
# It can also be used for a registry or to expose wrapper functions
import logging
import types # For isinstance checks if needed
from typing import Any, Dict, List, Optional, Callable, Type, Union
from importlib import import_module
from functools import partial

logger = logging.getLogger(__name__)

# Define a registry for client integrations
INTEGRATION_REGISTRY = []

class Registry:
    """Client integration registry with simplified wrapper management."""
    
    def __init__(self):
        self.integrations = []
        
    def register(self, client_type=None, target_path=None):
        """Decorator to register a client integration handler."""
        def decorator(func):
            self.integrations.append({
                'name': func.__name__,
                'handler': func,
                'client_type': client_type,
                'target_path': target_path
            })
            return func
        return decorator
        
    def wrapper(self, *methods):
        """Create a standard wrapper configuration for methods."""
        return {'methods_to_wrap': {m: globals()[f'wrap_{m}'] for m in methods if globals().get(f'wrap_{m}')}}
        
    def async_wrapper(self, *methods):
        """Create a standard wrapper configuration for async methods."""
        return {'async_methods_to_wrap': {m: globals()[f'wrap_{m}'] for m in methods if globals().get(f'wrap_{m}')}}
        
    def combined(self, sync_methods=None, async_methods=None, **extras):
        """Create a combined configuration with both sync and async wrappers."""
        result = {}
        
        if sync_methods:
            result['methods_to_wrap'] = {
                m: globals()[f'wrap_{m}'] for m in sync_methods if globals().get(f'wrap_{m}')
            }
            
        if async_methods:
            result['async_methods_to_wrap'] = {
                m: globals()[f'wrap_{m}'] for m in async_methods if globals().get(f'wrap_{m}')
            }
            
        # Add any extra configuration
        result.update(extras)
        return result
        
    def find_handler(self, client):
        """Find the appropriate handler for a client."""
        for integration in self.integrations:
            client_type = integration['client_type']
            
            # Check if client type matches
            if client_type is not None:
                try:
                    # Import client_type if it's a string (lazy import)
                    if isinstance(client_type, str):
                        module_path, class_name = client_type.rsplit('.', 1)
                        try:
                            module = import_module(module_path)
                            client_type = getattr(module, class_name)
                        except (ImportError, AttributeError):
                            continue
                    
                    # Skip if client doesn't match type
                    if not (client is client_type or isinstance(client, client_type)):
                        continue
                except Exception:
                    continue
            
            try:
                # Call the integration handler to get wrappers
                result = integration['handler'](client)
                if result:
                    # Add standard info to the result
                    result['name'] = integration['name']
                    if 'target_object_path' not in result and integration['target_path']:
                        result['target_object_path'] = integration['target_path']
                    return result
            except Exception:
                continue
        
        return None

# Create the global registry
registry = Registry()

# --- Import common wrapper functions ---
for wrapper_name in [
    'openai_chat_completions_create', 'openai_chat_completions_acreate', 
    'langchain_invoke', 'langchain_ainvoke',
    'litellm_completion', 'litellm_acompletion',
    'openai_agent_runner_sync', 'openai_agent_runner_async',
    'openai_module_v1'
]:
    try:
        # Import all wrappers from appropriate modules (will be populated into globals())
        exec(f"from .{wrapper_name.split('_')[0]}_wrapper import wrap_{wrapper_name}")
    except (ImportError, AttributeError):
        # Create dummy function if import fails
        exec(f"def wrap_{wrapper_name}(*args, **kwargs): return False")

# Import OpenAI v0.x wrappers if available
try:
    from .openai_v0_wrapper import wrap_openai_v0_chat_completion, wrap_openai_v0_completion, is_openai_v0
    HAS_OPENAI_V0 = True
except ImportError:
    wrap_openai_v0_chat_completion = lambda *args: False
    wrap_openai_v0_completion = lambda *args: False
    is_openai_v0 = lambda: False
    HAS_OPENAI_V0 = False

# Import other utility functions
try:
    from .openai_chat_wrapper import is_openai_v1
except ImportError:
    is_openai_v1 = lambda: False

# Import the OpenAI Agents trace processor if available
try:
    from .openai_agents_trace_processor import register_trace_processor
    HAS_AGENTS_TRACING = True
except ImportError:
    register_trace_processor = lambda *args: False
    HAS_AGENTS_TRACING = False

# --- Register integrations using decorators ---

@registry.register(client_type='openai.OpenAI', target_path='chat.completions')
def openai_client(client):
    """OpenAI V1 Client integration."""
    return registry.wrapper('openai_chat_completions_create')

@registry.register(client_type='openai.AsyncOpenAI', target_path='chat.completions')
def async_openai_client(client):
    """Async OpenAI V1 Client integration."""
    return registry.async_wrapper('openai_chat_completions_acreate')

@registry.register()
def openai_module(client):
    """OpenAI module (v1 or v0) integration."""
    # Check if it's a module named 'openai'
    if not (isinstance(client, types.ModuleType) and client.__name__ == 'openai'):
        return None
        
    # V0 integration (has ChatCompletion)
    if HAS_OPENAI_V0 and is_openai_v0() and hasattr(client, 'ChatCompletion'):
        return {
            'methods_to_wrap': {
                'ChatCompletion.create': wrap_openai_v0_chat_completion,
                'Completion.create': wrap_openai_v0_completion
            }
        }
    
    # V1 integration
    if is_openai_v1():
        return registry.wrapper('openai_module_v1')
    
    return None

@registry.register()
def litellm_module(client):
    """LiteLLM module integration."""
    try:
        import litellm
        if client is litellm:
            return registry.combined(
                sync_methods=['litellm_completion'],
                async_methods=['litellm_acompletion']
            )
    except ImportError:
        pass
    return None

@registry.register()
def openai_agent_runner(client):
    """OpenAI Agent Runner integration."""
    try:
        # Try both import paths
        Runner = None
        for import_path in ['agents', 'agents.run']:
            try:
                module = import_module(import_path)
                Runner = getattr(module, 'Runner')
                break
            except (ImportError, AttributeError):
                continue
                
        if Runner and (isinstance(client, Runner) or client is Runner):
            result = registry.combined(
                sync_methods=['openai_agent_runner_sync'],
                async_methods=['openai_agent_runner_async']
            )
            
            # Add trace processor if available
            if HAS_AGENTS_TRACING:
                result['register_trace_processor'] = register_trace_processor
                
            return result
    except Exception:
        pass
    return None

@registry.register()
def langchain_runnable(client):
    """LangChain Runnable integration."""
    # Check for invoke/ainvoke methods
    if not (hasattr(client, 'invoke') or hasattr(client, 'ainvoke')):
        return None
        
    # Try to get Runnable for more specific type check
    try:
        for import_path in ['langchain.schema.runnable', 'langchain.schema']:
            try:
                module = import_module(import_path)
                Runnable = getattr(module, 'Runnable')
                # If we found Runnable but client isn't an instance, rely on method check
                if not isinstance(client, Runnable):
                    pass
                break
            except (ImportError, AttributeError):
                continue
    except Exception:
        pass
        
    # Return appropriate wrappers based on available methods
    methods = {}
    if hasattr(client, 'invoke'):
        methods['sync_methods'] = ['langchain_invoke']
    if hasattr(client, 'ainvoke'):
        methods['async_methods'] = ['langchain_ainvoke']
        
    return registry.combined(**methods)

def find_integration(client: Any):
    """Find the first matching integration for the given client."""
    # Use the registry to find a handler
    integration = registry.find_handler(client)
    
    # If found and has trace processor, register it
    if integration and HAS_AGENTS_TRACING and 'register_trace_processor' in integration:
        from ragmetrics.api import ragmetrics_client
        integration['register_trace_processor'](ragmetrics_client)
    
    # For backwards compatibility
    if integration and integration not in INTEGRATION_REGISTRY:
        INTEGRATION_REGISTRY.append(integration)
        
    return integration

# Export public API
__all__ = ["find_integration", "registry"] 