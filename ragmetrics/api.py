import types
import requests
import sys
import os
import time
import json
import uuid
import logging
import functools # For functools.reduce
from typing import Any, Callable, Optional, List, Dict

# Imports for type hinting and specific client SDKs if needed for checks
try:
    from agents import Runner
except ImportError:
    Runner = None # Define as None if openai-agents is not installed

# Local project imports
from .utils import default_callback
from .client_integrations import find_integration

logger = logging.getLogger(__name__)

# --- Custom Exceptions --- 
class RagMetricsError(Exception):
    """Base exception class for RagMetrics errors."""
    pass

class RagMetricsConfigError(RagMetricsError):
    """Exception for configuration-related errors."""
    pass

class RagMetricsAuthError(RagMetricsError):
    """Exception for authentication failures."""
    def __init__(self, message, status_code=None, response_text=None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text

    def __str__(self):
        s = super().__str__()
        if self.status_code: s += f" (Status Code: {self.status_code})"
        return s

class RagMetricsAPIError(RagMetricsError):
    """Exception for errors returned by the RagMetrics API."""
    def __init__(self, message, status_code=None, response_text=None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text

    def __str__(self):
        s = super().__str__()
        if self.status_code: s += f" (Status Code: {self.status_code})"
        if self.response_text: s += f" Response: {self.response_text[:200]}..." 
        return s

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

        try:
            def serialize_default(o):
                if hasattr(o, "model_dump_json") and callable(o.model_dump_json): 
                    try: return json.loads(o.model_dump_json())
                    except: return o.model_dump_json() 
                if hasattr(o, "model_dump") and callable(o.model_dump): return o.model_dump()
                if hasattr(o, "dict") and callable(o.dict): return o.dict()
                if isinstance(o, uuid.UUID) or type(o).__name__ == 'datetime': return str(o)
                if isinstance(o, time.struct_time): return time.strftime("%Y-%m-%dT%H:%M:%SZ", o)
                if isinstance(o, Exception): return str(o) # Serialize exceptions
                try: return str(o)
                except: return f"<unserializable:{type(o).__name__}>"
            payload_final = json.loads(json.dumps(payload, indent=4, default=serialize_default))
        except Exception as e: # Catch broader exceptions during the complex serialization
            logger.error(f"RagMetrics: Error serializing payload to JSON: {e}. Payload (first 500 chars): {str(payload)[:500]}")
            return None 

        log_resp = self._make_request(
            method='post',
            endpoint='/api/client/logtrace/',
            json=payload_final,
            headers={
                "Authorization": f"Token {self.access_token}",
                "Content-Type": "application/json"
            }
        )

        if log_resp and log_resp.status_code >= 400:
             logger.warning(f"RagMetrics: API returned error on log trace: {log_resp.status_code} - {log_resp.text[:200]}") # Limit response text
        elif log_resp:
             logger.debug(f"RagMetrics: Trace logged successfully (Conv ID: {current_trace_conversation_id}).")
             
        return log_resp

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
        if off:
            self.logging_off = True
            self.access_token = None
            logger.info("RagMetrics: Logging explicitly disabled.")
            return True
        
        self.logging_off = False # Ensure logging is enabled if off=False

        # Determine API key: Arg > Env Var
        api_key_to_use = key or os.environ.get('RAGMETRICS_API_KEY')
        if not api_key_to_use:
            self.logging_off = True # Disable logging
            raise RagMetricsConfigError("Missing API key. Provide key to login() or set RAGMETRICS_API_KEY environment variable.")

        # Determine base URL: Arg > Env Var > Current/Default
        base_url_to_use = base_url or os.environ.get('RAGMETRICS_BASE_URL', self.base_url) 
        self.base_url = base_url_to_use # Update client's base_url

        try:
            logger.info(f"RagMetrics: Attempting login to {self.base_url}...")
            response = self._make_request(method='post', endpoint='/api/client/login/', json={"key": api_key_to_use})
            
            if response is None: # _make_request failed (e.g., network error)
                 raise RagMetricsAPIError("Login request failed. Check network connectivity or base URL.")
            
            if response.status_code in [401, 403]:
                 raise RagMetricsAuthError(f"Authentication failed (Status: {response.status_code}). Check API key.", status_code=response.status_code, response_text=response.text)
            
            response.raise_for_status() # Raise HTTPError for other bad responses (4xx client error, 5xx server error)

            # Success
            self.access_token = api_key_to_use
            self.new_conversation() 
            logger.info(f"RagMetrics: Successfully logged in. Base URL: {self.base_url}")
            return True

        except requests.exceptions.HTTPError as e:
            # Handle non-auth HTTP errors specifically
            logger.error(f"RagMetrics: Login request failed. HTTP Error {e.response.status_code} for {e.request.url}. Response: {e.response.text}")
            self.access_token = None
            self.logging_off = True
            raise RagMetricsAPIError(f"Login request failed (Status: {e.response.status_code}).", status_code=e.response.status_code, response_text=e.response.text) from e
        except RagMetricsError: # Re-raise our custom errors
             self.access_token = None
             self.logging_off = True
             raise
        except Exception as e:
            # Catch any other unexpected errors during login
            logger.error(f"RagMetrics: An unexpected error occurred during login: {e}", exc_info=True)
            self.access_token = None
            self.logging_off = True
            raise RagMetricsError(f"An unexpected error occurred during login: {e}") from e

    def _make_request(self, endpoint: str, method: str ="post", **kwargs) -> Optional[requests.Response]:
        """Internal method to make API requests, handling common errors."""
        if not self.access_token and "/api/client/login/" not in endpoint: # Allow login requests without token
             logger.error("RagMetrics: Attempted API call without access token. Please login first.")
             return None # Cannot make request without token (except login)
             
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        response = None
        try:
            kwargs.setdefault('timeout', 15)
            logger.debug(f"RagMetrics: Sending {method.upper()} request to {url} with timeout {kwargs['timeout']}s")
            response = requests.request(method, url, **kwargs) 
            # Check for API-level errors even if HTTP status is 2xx but response indicates issue
            # (This depends on API contract, but good practice to check)
            # if response.status_code < 400 and response.content and "error" in response.text.lower():
            #     logger.warning(f"RagMetrics: API request to {url} succeeded (HTTP {response.status_code}) but response contains potential error: {response.text[:200]}")

        except requests.exceptions.Timeout:
            logger.error(f"RagMetrics: Request to {url} timed out after {kwargs['timeout']} seconds.")
            # Return None, let caller handle timeout
        except requests.exceptions.ConnectionError as e:
             logger.error(f"RagMetrics: Connection error for {url}. Check network or base URL. Error: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"RagMetrics: Request to {url} failed: {e}")
            # Log specific HTTP errors if available in the exception context
            if e.response is not None:
                 logger.error(f"RagMetrics: Status Code: {e.response.status_code}, Response: {e.response.text[:200]}")
        except Exception as e:
             logger.error(f"RagMetrics: Unexpected error during request to {url}: {e}", exc_info=True)

        return response # Return the response object (or None on critical failure)

    def monitor(self, client: Any, metadata: Optional[dict] = None, callback: Optional[Callable[[Any, Any], dict]] = None) -> Any:
        """
        Monitor an LLM client by finding a matching integration in the registry and applying its wrapper.
        """
        if not hasattr(client, '__class__') and not isinstance(client, types.ModuleType):
            client_type = type(client).__name__
            logger.error(f"RagMetrics: Cannot monitor object of type '{client_type}'. Expected class instance, module, or specific known type like Runner.")
            return client # Return unmodified
            
        if not self.access_token and not self.logging_off:
            logger.warning("RagMetrics: Client not authenticated. Methods will be wrapped, but traces won't be logged until login() is successful.")
        
        if metadata is not None:
            if self.metadata is None: self.metadata = {}
            self.metadata.update(metadata)
            logger.debug(f"RagMetrics: Updated client session metadata: {self.metadata}")
        
        current_callback = callback if callback is not None else default_callback

        integration = find_integration(client)
        
        if not integration:
            client_name_for_log = getattr(client, '__name__', str(type(client)))
            logger.warning(
                f"RagMetrics: Client '{client_name_for_log}' was not matched by any integration in the registry. "
                f"Automatic monitoring may not be active."
            )
            return client

        logger.info(f"RagMetrics: Applying integration: {integration['name']}")
        
        wrapper_applied_count = 0
        
        # Resolve sync target object
        target_object = client
        target_object_path = integration.get("target_object_path")
        if target_object_path:
            path_parts = target_object_path.split('.')
            try:
                target_object = functools.reduce(getattr, path_parts, client)
            except AttributeError:
                logger.error(f"RagMetrics: Could not resolve target_object_path '{target_object_path}' for client type {type(client).__name__}. Monitoring may fail.")
                return client # Stop if target cannot be found

        # Apply synchronous wrappers
        for method_name, wrapper_func in integration.get("methods_to_wrap", {}).items():
            # Check if the target object actually has this method
            if hasattr(target_object, method_name):
                logger.debug(f"Attempting to wrap sync method: {method_name} on {type(target_object).__name__}")
                try:
                    # Call the specific wrapper function from the registry
                    if wrapper_func(self, target_object, current_callback):
                        wrapper_applied_count += 1
                    else:
                        logger.warning(f"Sync wrapper function for {method_name} reported failure.")
                except Exception as e:
                    logger.error(f"Error applying sync wrapper for {method_name}: {e}", exc_info=True)
            else:
                logger.debug(f"Sync method '{method_name}' not found on target {type(target_object).__name__}")

        # Resolve async target object
        async_target_object = client
        async_target_object_path = integration.get("async_target_object_path", target_object_path) # Default to sync path
        if async_target_object_path and async_target_object_path != target_object_path:
            # Only resolve again if path is different from sync path
            path_parts = async_target_object_path.split('.')
            try:
                async_target_object = functools.reduce(getattr, path_parts, client)
            except AttributeError:
                logger.error(f"RagMetrics: Could not resolve async_target_object_path '{async_target_object_path}'. Async monitoring may fail.")
                async_target_object = None # Mark as failed to resolve
        elif async_target_object_path == target_object_path:
             async_target_object = target_object # Use the already resolved sync target
        # else: async_target_object remains the original client if no path specified

        # Apply asynchronous wrappers
        if async_target_object: # Only proceed if async target was resolved
            for method_name, wrapper_func in integration.get("async_methods_to_wrap", {}).items():
                if hasattr(async_target_object, method_name):
                    logger.debug(f"Attempting to wrap async method: {method_name} on {type(async_target_object).__name__}")
                    try:
                        # Call the specific async wrapper function
                        if wrapper_func(self, async_target_object, current_callback):
                            wrapper_applied_count += 1
                        else:
                            logger.warning(f"Async wrapper function for {method_name} reported failure.")
                    except Exception as e:
                         logger.error(f"Error applying async wrapper for {method_name}: {e}", exc_info=True)
                else:
                     logger.debug(f"Async method '{method_name}' not found on target {type(async_target_object).__name__}")

        if wrapper_applied_count > 0:
            logger.info(f"RagMetrics: Monitoring applied for {wrapper_applied_count} method(s) on client '{getattr(client, '__name__', str(type(client)))}'.")
        else:
            logger.warning(f"RagMetrics: No wrappers were applied for client '{getattr(client, '__name__', str(type(client)))}' via integration '{integration['name']}'.")
            
        return client

class RagMetricsObject:
    """
    Base class for RagMetrics objects that can be stored on the platform.
    
    This abstract class provides common functionality for objects that can be
    serialized to and from the RagMetrics API, including saving, downloading,
    and conversions between Python objects and API representations.
    
    All RagMetrics object classes (Dataset, Criteria, etc.) inherit from this class.
    """

    object_type: str = None

    def to_dict(self):
        """
        Convert the object to a dictionary representation.
        
        This method must be implemented by subclasses to define how the object
        is serialized for API communication.

    
    Returns:
            dict: Dictionary representation of the object.
            
    
    Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create an object instance from a dictionary.
        
        This method creates a new instance of the class from data received
        from the API. Subclasses may override this to customize deserialization.

    
    Args:
            data: Dictionary containing object data.

    
    Returns:
            RagMetricsObject: A new instance of the object.
        """
        return cls(**data)

    def save(self):
        """
        Save the object to the RagMetrics API.
        
        This method sends the object to the RagMetrics API for storage and
        retrieves the assigned ID.

    
    Returns:
            Response: The API response from saving the object.
            
    
    Raises:
            ValueError: If object_type is not defined.
            Exception: If the API request fails.
        """
        if not self.object_type:
            raise ValueError("object_type must be defined.")
        payload = self.to_dict()
        # e.g. /api/client/task/save/ or /api/client/dataset/save/
        endpoint = f"/api/client/{self.object_type}/save/"
        headers = {"Authorization": f"Token {ragmetrics_client.access_token}"}
        response = ragmetrics_client._make_request(
            method="post", endpoint=endpoint, json=payload, headers=headers
        )
        if response.status_code == 200:
            json_resp = response.json()
            self.id = json_resp.get(self.object_type, {}).get("id")
        else:
            raise Exception(f"Failed to save {self.object_type}: {response.text}")

    @classmethod
    def download(cls, id=None, name=None):
        """
        Download an object from the RagMetrics API.
        
        This method retrieves an object from the RagMetrics API by its ID or name.

    
    Args:
            id: ID of the object to download (mutually exclusive with name).
            name: Name of the object to download (mutually exclusive with id).

    
    Returns:
            RagMetricsObject: The downloaded object instance.
            
    
    Raises:
            ValueError: If neither id nor name is provided, or if object_type is not defined.
            Exception: If the API request fails.
        """
        if not cls.object_type:
            raise ValueError("object_type must be defined.")
        if id is None and name is None:
            raise ValueError("Either id or name must be provided.")
        
        if id is not None:
            endpoint = f"/api/client/{cls.object_type}/download/?id={id}"
        else:
            endpoint = f"/api/client/{cls.object_type}/download/?name={name}"
        
        headers = {"Authorization": f"Token {ragmetrics_client.access_token}"}
        response = ragmetrics_client._make_request(
            method="get", endpoint=endpoint, headers=headers
        )
        if response.status_code == 200:
            json_resp = response.json()
            obj_data = json_resp.get(cls.object_type, {})
            obj = cls.from_dict(obj_data)
            obj.id = obj_data.get("id")
            return obj
        else:
            raise Exception(f"Failed to download {cls.object_type}: {response.text}")
        
# Global client instance
ragmetrics_client = RagMetricsClient()

def login(key: Optional[str]=None, base_url: Optional[str]=None, off: bool=False) -> bool:
    """
    Authenticate the global RagMetrics client. Convenience function.
    Raises RagMetrics specific exceptions on failure.
    """
    try:
        return ragmetrics_client.login(key=key, base_url=base_url, off=off)
    except RagMetricsError as e:
         # Log the error from the convenience function as well for visibility
         logger.error(f"RagMetrics login failed: {e}")
         raise # Re-raise the specific error

def monitor(client: Any, metadata: Optional[dict]=None, callback: Optional[Callable[[Any, Any], dict]]=None) -> Any:
    """
    Monitor an LLM client using the global RagMetrics client. Convenience function.
    """
    return ragmetrics_client.monitor(client, metadata=metadata, callback=callback)
