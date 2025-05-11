import logging
import os
import sys
import time
import uuid
import requests
from typing import Any, Optional, Dict, List, Callable

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
    logger.debug("Enabled debug logging for RagMetrics client")

# Import exceptions and utility functions
from .api import (
    RagMetricsError,
    RagMetricsConfigError, 
    RagMetricsAuthError, 
    RagMetricsAPIError,
    _serialize_default
)

try:
    from .utils import default_callback
except ImportError:
    # Fallback for standalone testing
    default_callback = lambda input_msg, output_msg: {"input": input_msg, "output": output_msg}

class RagMetricsClient:
    """
    Client for interacting with the RagMetrics API.
    
    This class handles authentication, request management, and logging of LLM interactions.
    It provides the core functionality for monitoring LLMs and RAG systems, including:
    
    * Authenticating with the RagMetrics API
    * Logging LLM interactions (inputs, outputs, context, metadata)
    * Tracking conversation sessions
    * Wrapping various LLM clients (OpenAI, LangChain, etc.) for monitoring
    """

    def __init__(self):
        """
        Initialize a new RagMetricsClient instance.
        
        Creates an unauthenticated client. Call the login() method to authenticate
        before using other functionality.
        """
        self.access_token: Optional[str] = None
        # Initialize base_url from env var or default, login can override
        self.base_url: str = os.environ.get('RAGMETRICS_BASE_URL', 'https://ragmetrics.ai') 
        self.logging_off: bool = False
        self.metadata: Optional[dict] = None 
        self.conversation_id: str = str(uuid.uuid4())
        self.trace_ids: List[str] = [] # For capturing trace IDs during tests
        logger.info(f"RagMetricsClient initialized. Default Base URL: {self.base_url}")
    
    def new_conversation(self, id: Optional[str] = None):
        """
        Reset the conversation ID to a new UUID or use the provided ID.
        
        Call this method to start a new conversation thread. All subsequent
        interactions will be logged under the new conversation ID until this
        method is called again.
        
        Args:
            id: Optional custom conversation ID. If not provided, a new UUID will be generated.
        """
        self.conversation_id = id if id is not None else str(uuid.uuid4())
        logger.debug(f"RagMetrics: Started new conversation: {self.conversation_id}")

    def _find_external_caller(self) -> str:
        """
        Find the first non-ragmetrics function in the call stack.
        
        Used internally to identify which user function triggered a logging event.

        Returns:
            str: The name of the first external function that called into ragmetrics,
                 or an empty string if none is found.
        """
        external_caller = ""
        try:
            frame = sys._getframe().f_back # Start one frame back to exclude this method itself
            # Traverse up to a certain limit to avoid excessively deep stack walks
            for _ in range(10): # Limit depth to 10 frames for performance/safety
                if not frame: break
                module_name = frame.f_globals.get("__name__", "")
                if (not module_name.startswith("ragmetrics.") and 
                    module_name != "ragmetrics" and 
                    not module_name.startswith("__main__") and 
                    # Exclude common interactive environments
                    not module_name.startswith("ipykernel.") and 
                    not module_name.startswith("IPython.") and 
                    "<module>" not in frame.f_code.co_name):
                    external_caller = frame.f_code.co_name
                    break
                frame = frame.f_back
        except Exception as e:
            logger.debug(f"RagMetrics: Error finding external caller: {e}")
        return external_caller

    def _log_trace(
            self, 
            input_messages: Any, # Can be list of dicts, string, etc.
            response: Any,       # Can be dict, object, string, etc.
            metadata_llm: Optional[dict],
            contexts: Optional[list],
            expected: Optional[Any],
            duration: float,
            tools: Optional[list],
            callback_result: Optional[dict] = None,
            conversation_id: Optional[str] = None,
            force_new_conversation: bool = False, # New flag for explicit control
            error: Optional[Exception] = None, # Added error to signature
            model_name: Optional[str] = None, # Added model_name
            tool_choice: Optional[Any] = None, # Added tool_choice
            **kwargs # For any additional passthrough data to the payload root
        ):
        """
        Log a trace of an LLM interaction to the RagMetrics API.
        
        This is the core logging method used by monitored LLM clients to record
        interactions with LLMs. It handles various formats and includes detailed
        metadata about the interaction.

        Args:
            input_messages: The input messages sent to the LLM (prompts, queries, etc.).
            response: The response received from the LLM.
            metadata_llm: Additional metadata about the LLM and the interaction.
            contexts: Context information or retrieved documents used in the interaction.
            expected: The expected output from the LLM (for evaluation).
            duration: The duration of the interaction in seconds.
            tools: Any tools or functions used/available during the interaction.
            callback_result: Processed results from a callback (e.g., default_callback).
            conversation_id: Specific conversation ID for this trace, overrides client's current.
            force_new_conversation: Flag to force a new conversation for the trace.
            error: Optional exception object if the interaction resulted in an error.
            model_name: Optional name of the LLM used for the interaction.
            tool_choice: Optional choice of tool used for the interaction.
            **kwargs: Additional keyword arguments to include at the root of the trace payload.
        Returns:
            Response: The API response from logging the trace, or None if logging is off.
        """
        if self.logging_off:
            logger.debug("RagMetrics: Logging is off, skipping trace.")
            return None
        if not self.access_token:
            logger.warning("RagMetrics: Not logged in. Trace not sent. Call login() first.")
            return None
        
        current_trace_conversation_id: str
        if force_new_conversation:
            self.new_conversation()
            current_trace_conversation_id = self.conversation_id
            logger.debug(f"RagMetrics: Forced new conversation for trace: {current_trace_conversation_id}")
        elif conversation_id is not None:
            current_trace_conversation_id = conversation_id # Use explicitly passed ID
        else:
            # Apply heuristic only if no explicit conversation ID and not forced
            current_trace_conversation_id = self.conversation_id # Start with current client ID
            is_likely_new_interaction = False
            if isinstance(input_messages, list) and len(input_messages) == 1:
                first_message = input_messages[0]
                is_continuation = False
                if isinstance(first_message, dict):
                    role = first_message.get("role")
                    is_continuation = first_message.get("tool_call_id") or first_message.get("tool_calls") or role in ["assistant", "tool", "system"]
                if not is_continuation:
                    is_likely_new_interaction = True
            elif isinstance(input_messages, str): 
                 is_likely_new_interaction = True
            
            if is_likely_new_interaction:
                self.new_conversation()
                current_trace_conversation_id = self.conversation_id # Use the new ID
                logger.debug(f"RagMetrics: Heuristically started new conversation for trace: {current_trace_conversation_id}")
            else:
                 logger.debug(f"RagMetrics: Continuing conversation for trace: {current_trace_conversation_id}")

        response_processed = response
        if hasattr(response, "model_dump") and callable(response.model_dump):
            response_processed = response.model_dump() 
        elif hasattr(response, "dict") and callable(response.dict):
            response_processed = response.dict()

        union_metadata = {}
        if isinstance(self.metadata, dict):
            union_metadata.update(self.metadata)
        if isinstance(metadata_llm, dict):
            union_metadata.update(metadata_llm)

        payload = {
            "raw": {
                "input": input_messages, # Keep raw input as is
                "output": response_processed, # Keep raw (processed) output as is
                "id": str(uuid.uuid4()), # Unique ID for this specific raw log entry
                "duration": duration,
                "caller": self._find_external_caller(),
                "error": str(error) if error else None
            },
            "metadata": union_metadata,
            "contexts": contexts,
            "expected": expected,
            "tools": tools,
            "tool_choice": tool_choice,
            "model_name": model_name,
            "input": None, # Placeholder, to be filled by callback_result
            "output": None, # Placeholder, to be filled by callback_result
            "scores": None, # Placeholder for potential future score passing
            "conversation_id": current_trace_conversation_id
        }

        # Apply callback_result if provided and valid
        if isinstance(callback_result, dict):
            for key in ["input", "output", "expected", "scores"]:
                if key in callback_result:
                    payload[key] = callback_result[key]
        
        # Merge any additional kwargs into the payload root
        if kwargs:
            payload.update(kwargs)

        # Make the API call to log the trace
        response_data = self._make_request(endpoint="/api/client/logtrace/", method="post", json=payload) 

        if response_data and isinstance(response_data, dict):
            trace_id = response_data.get("id") 
            if trace_id and hasattr(self, 'test_logged_trace_ids'):
                self.trace_ids.append(str(trace_id))
            return response_data 
        
        return None

    async def _alog_trace(
        self,
        input_messages: Any,
        response: Any,
        expected: Optional[Any] = None,
        contexts: Optional[List[Any]] = None,
        metadata_llm: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        duration: Optional[float] = None,
        model_name: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None, # Or Dict for specific function call
        callback_result: Optional[Any] = None,
        force_new_conversation: bool = False,
        **additional_kwargs
    ):
        """Placeholder for asynchronous trace logging."""
        # TODO: Implement true async HTTP request here (e.g., using httpx.AsyncClient)
        logger.error("RagMetrics: _alog_trace is using synchronous _log_trace. Implement full async logging.")
        # For now, delegate to sync version (not ideal for performance in async contexts)
        # Don't pass the endpoint parameter - _log_trace will add it
        return self._log_trace(
            input_messages=input_messages,
            response=response,
            expected=expected,
            contexts=contexts,
            metadata_llm=metadata_llm,
            error=error,
            duration=duration,
            model_name=model_name,
            tools=tools,
            tool_choice=tool_choice,
            callback_result=callback_result,
            force_new_conversation=force_new_conversation,
            **additional_kwargs
        )

    def login(self, key: Optional[str]=None, base_url: Optional[str]=None, off: bool=False) -> bool:
        """
        Authenticate with the RagMetrics API.
        Raises RagMetricsConfigError, RagMetricsAuthError, RagMetricsAPIError on failure.
        Returns True on success or if logging is off.
        """
        # Clear test_logged_trace_ids when login is called
        if hasattr(self, 'test_logged_trace_ids'):
            self.trace_ids = []
            
        if off:
            self.logging_off = True
            self.access_token = None 
            logger.info("RagMetrics: Logging explicitly disabled.")
            return True

        api_key_to_use = key or os.environ.get('RAGMETRICS_API_KEY')
        if not api_key_to_use:
            raise RagMetricsConfigError("Missing API key. Provide key to login() or set RAGMETRICS_API_KEY environment variable.")

        base_url_to_use = base_url or os.environ.get('RAGMETRICS_BASE_URL', self.base_url) 
        self.base_url = base_url_to_use
        self.access_token = None

        try:
            logger.info(f"RagMetrics: Validating API key via {self.base_url}/api/client/login/...")
            login_payload = {"key": api_key_to_use}
            # Call login endpoint purely to validate the key
            response_data = self._make_request(
                method='post', 
                endpoint='/api/client/login/', 
                json=login_payload
            )
            username = response_data['user']['username']
            logger.info(f"RagMetrics: {username} logged in.")           
            self.access_token = api_key_to_use
            self.new_conversation() 
            return True

        except (RagMetricsAuthError, RagMetricsAPIError) as e:
            # Handle validation failure (e.g., 401/403 from /api/client/login/)
            logger.error(f"RagMetrics: API key validation failed: {e}")
            self.access_token = None
            self.logging_off = True
            raise # Re-raise the specific error
        except Exception as e:
            logger.error(f"RagMetrics: An unexpected error occurred during API key validation: {e}", exc_info=True)
            self.access_token = None
            self.logging_off = True
            raise RagMetricsError(f"An unexpected error occurred during API key validation: {e}") from e

    def _make_request(self, endpoint: str, method: str ="post", **kwargs) -> Optional[Any]:
        """Internal helper to make authenticated requests to the RagMetrics API."""
        logger.debug(f"_make_request(method={method}")
        url = f"{self.base_url}{endpoint}"

        # Add Authorization header if we have an access token (except for login endpoint)
        if self.access_token:
            headers = kwargs.get('headers', {})
            headers['Authorization'] = f"Token {self.access_token}"
            kwargs['headers'] = headers
            logger.debug(f"Adding Authorization header to request for {endpoint}")

        # Add debug logging
        if kwargs.get('json'):
            import json
            try:
                # Log the JSON data being sent
                json_data = json.dumps(kwargs.get('json'), default=_serialize_default)
                logger.debug(f"DEBUG - Request JSON payload: {json_data}")
            except Exception as e:
                logger.error(f"DEBUG - Failed to serialize request payload: {e}")
                # Print the raw data to see what's causing the issue
                logger.error(f"DEBUG - Raw payload: {kwargs.get('json')}")

        response = requests.request(method, url, **kwargs)
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_message = f"HTTP Error: {e.response.status_code} for {url}"
            if e.response.status_code == 401:
                logger.error(f"{error_message} - Unauthorized. Check your API key.")
                raise RagMetricsAuthError(f"Authentication failed: {error_message}", status_code=e.response.status_code, response_text=e.response.text)
            elif e.response.status_code == 403:
                logger.error(f"{error_message} - Forbidden. Check permissions.")
                raise RagMetricsAuthError(f"Permission denied: {error_message}", status_code=e.response.status_code, response_text=e.response.text)
            else:
                logger.error(f"{error_message}")
                # Log the response content in case of error
                logger.error(f"DEBUG - Response content: {e.response.text}")
                raise RagMetricsAPIError(f"API request failed: {error_message}", status_code=e.response.status_code, response_text=e.response.text)
        except ValueError as e:
            error_message = f"Invalid JSON in response: {e}"
            logger.error(f"{error_message}")
            logger.error(f"DEBUG - Response content: {response.text}")
            raise RagMetricsAPIError(error_message, response_text=response.text)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise RagMetricsError(f"Request error: {e}")

    def monitor(self, client: Any, metadata: Optional[dict] = None, callback: Optional[Callable[[Any, Any], dict]] = None) -> Any:
        """Monitor an LLM client by applying the appropriate wrapper based on client type."""
        # Reset trace IDs and update metadata
        if hasattr(self, 'trace_ids'):
            self.trace_ids = []
            
        if metadata:
            self.metadata = self.metadata or {}
            self.metadata.update(metadata)
        
        # Import and find integration (lazy import to avoid circular references)
        from .client_integrations import find_integration
        integration = find_integration(client)
        self._apply_integration(client, integration, callback or default_callback)

        return client
        
    def _apply_integration(self, client, integration, callback):
        """Apply an integration to a client."""
        if not integration:
            raise ValueError(f"No integration found for client: {client}")

        # Process both synchronous and asynchronous methods with a common approach
        method_configs = [
            {
                "path": integration.get("target_object_path"),
                "methods": integration.get("methods_to_wrap", {})
            },
            {
                "path": integration.get("async_target_object_path", integration.get("target_object_path")),
                "methods": integration.get("async_methods_to_wrap", {})
            }
        ]
        
        for config in method_configs:
            if config["methods"]:
                target = self._get_target_object(client, config["path"])
                if target:
                    self._wrap_methods(client, target, config["methods"], callback)
    
    def _get_target_object(self, client, path):
        """Get a target object by path."""
        if not path:
            return client
            
        obj = client
        try:
            for part in path.split('.'):
                obj = getattr(obj, part)
            return obj
        except (AttributeError, TypeError):
            return None
            
    def _wrap_methods(self, client, target, methods, callback):
        """Wrap methods on a target object."""
        for method_name, wrapper_func in methods.items():
            try:
                if "." in method_name:
                    # Handle methods with dot notation (e.g., "ChatCompletion.create")
                    self._wrap_dotted_method(client, method_name, wrapper_func, callback)
                elif hasattr(target, method_name):
                    # Apply wrapper to direct method
                    wrapper_func(self, target, callback)
            except Exception as e:
                logger.debug(f"Error wrapping method {method_name}: {e}")
    
    def _wrap_dotted_method(self, client, method_name, wrapper_func, callback):
        """Handle wrapping methods with dot notation (e.g., "ChatCompletion.create")."""
        parts = method_name.split('.')
        obj = client  # Start from client for dotted paths
        
        # Navigate to parent object
        for part in parts[:-1]:
            if not hasattr(obj, part):
                return  # Exit if we can't find the path
            obj = getattr(obj, part)
        
        # Apply wrapper if method exists on parent
        if hasattr(obj, parts[-1]):
            wrapper_func(self, client, callback)

# Create a global client instance
ragmetrics_client = RagMetricsClient() 