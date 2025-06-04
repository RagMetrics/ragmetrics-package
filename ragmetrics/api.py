import types
import requests
import sys
import os
import time
import json
import uuid
import logging
from typing import Any, Callable, Optional
from .utils import format_function_signature

logger = logging.getLogger(__name__)

def default_input(raw_input):
    """
    Format input messages into a standardized string format.
    
    Handles various input formats (list of messages, single message object, etc.)
    and converts them into a consistent string representation.
    
    Args:
        raw_input: The input to format. Can be a list of messages, a dictionary
                  with role/content keys, an object with role/content attributes,
                  or a primitive value.
    
    Returns:
        str: Formatted string representation of the input, or None if input is empty.
    """
    # OpenAI Completions API, Multiple Messages
    if isinstance(raw_input, list) and len(raw_input) > 0:
        raw_input = raw_input[-1]
    
    # OpenAI Completions API, Single Message
    if isinstance(raw_input, dict) and "content" in raw_input:
        return raw_input.get('content', '')

    elif hasattr(raw_input, "content"):
        return raw_input.content

    # OpenAI Responses API, function call output
    if isinstance(raw_input, dict) and \
        "type" in raw_input and \
        raw_input["type"] == "function_call_output" and \
        "output" in raw_input:
        return raw_input["output"]
    
    else:
        return str(raw_input)

def default_output(raw_response):
    """
    Extract content from various types of LLM responses.
    
    Handles different response formats from various LLM providers and APIs,
    extracting the actual content in a consistent way.
    
    Args:
        raw_response: The response object from the LLM. Can be OpenAI ChatCompletion,
                     object with text/content attributes, or another response format.
    
    Returns:
        str: The extracted content from the response, or the raw response if content
             cannot be extracted.
    """
    if not raw_response:
        return None
        
    # Handle tool_calls in the response (OpenAI function calling API)
    if isinstance(raw_response, dict) and "choices" in raw_response:
        try:
            message = raw_response["choices"][0]["message"]
            if message.get("tool_calls") and not message.get("content"):
                tool_call = message["tool_calls"][0]
                if tool_call["type"] == "function":
                    func_name = tool_call["function"]["name"]
                    # Parse the JSON arguments
                    args_dict = json.loads(tool_call["function"]["arguments"])
                    return format_function_signature(func_name, args_dict)
        except Exception as e:
            logger.error("Error formatting tool_calls from response: %s", e)

    #OpenAI Response API
    if isinstance(raw_response, dict) and \
        raw_response.get('type', None) == 'SpanImpl':
        
        #Tool selection
        try:
            output = raw_response['span_data']['response']['output']
            function_calls = []
            for o in output:
                if o['type'] == 'function_call':
                    func_name = o['name']
                    args_dict = json.loads(o['arguments'])
                    function_calls.append(format_function_signature(func_name, args_dict))
            if len(function_calls) > 0:
                function_calls = '\n'.join(function_calls)
                return function_calls
        except KeyError as e:
            pass

        #Tool response
        try:
            if raw_response['span_data']['type'] == 'function':
                return raw_response['span_data']['output']
        except KeyError as e:
            pass

        #Assistant response
        try:
            last_output = raw_response['span_data']['response']['output'][-1]
            if last_output['role'] == 'assistant':
                return last_output['content'][-1]['text']
        except KeyError as e:
            pass

        #Agent span
        try:
            if raw_response['span_data']['type'] == 'agent':
                return f"Agent: {raw_response['span_data']['name']} Completed"
        except KeyError as e:
            pass
    
    # OpenAI Completions API (Legacy)
    if hasattr(raw_response, "choices") and raw_response.choices:
        try:
            message = raw_response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls and (not message.content or message.content is None):
                tool_call = message.tool_calls[0]
                if tool_call.type == "function":
                    func_name = tool_call.function.name
                    # Parse the JSON arguments
                    args_dict = json.loads(tool_call.function.arguments)
                    return format_function_signature(func_name, args_dict)
            return message.content
        except Exception as e:
            logger.error("Error extracting content from response.choices: %s", e)
    
    # Handle other response types
    if hasattr(raw_response, "text"):
        content = raw_response.text
    if hasattr(raw_response, "content"):
        content = raw_response.content
    else:
        content = str(raw_response)
    return content

def default_callback(raw_input, raw_output) -> dict:
    """
    Create a standardized callback result dictionary.
    
    This is the default callback used by the monitor function when no custom
    callback is provided.


    Args:
        raw_input: The raw input to the LLM.
        raw_output: The raw output from the LLM.


    Returns:
        dict: A dictionary containing formatted input and output.
    """
    return {
        "input": default_input(raw_input),
        "output": default_output(raw_output)
    }

def trace_function_call(func):
    """
    Decorator to trace function execution and log structured input/output.
    
    Wrap a function with this decorator to automatically log its execution
    details to RagMetrics, including inputs, outputs, and timing information.
    This is particularly useful for tracking retrieval functions in RAG applications.

    Example - Tracing a weather API function:
        
        .. code-block:: python
        
            import requests
            import ragmetrics
            from ragmetrics import trace_function_call
            
            # First, login to RagMetrics
            ragmetrics.login("your-api-key")
            
            # Apply the decorator to your function
            @trace_function_call
            def get_weather(latitude, longitude):
                response = requests.get(
                    f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
                )
                data = response.json()
                return data['current']['temperature_2m']
                
            # Now when you call the function, it's automatically traced
            temperature = get_weather(48.8566, 2.3522)  # Paris coordinates
        
    Example - Tracing a document retrieval function:
    
        .. code-block:: python
        
            @trace_function_call
            def retrieve_documents(query, top_k=3):
                # Connect to your vector database
                results = vector_db.search(query, limit=top_k)
                return [doc.text for doc in results]
                
            # The function call, arguments, and return value will be logged
            contexts = retrieve_documents("What is the capital of France?")
            

    Args:
        func: The function to be traced.


    Returns:
        Callable: A wrapped version of the function that logs execution details.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Format parameters in the requested format
        params = []
        # Add positional arguments
        for i, arg in enumerate(args):
            # Try to get the parameter name from the function signature
            try:
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if i < len(param_names):
                    params.append(f"{param_names[i]}={repr(arg)}")
                else:
                    params.append(f"{repr(arg)}")
            except:
                params.append(f"{repr(arg)}")
        
        # Add keyword arguments
        for k, v in kwargs.items():
            params.append(f"{k}={repr(v)}")
        
        # Create the formatted function call string
        formatted_call = f"={func.__name__}({', '.join(params)})"

        # Prepare structured input format
        function_input = [
            {
                "role": "user",
                "content": formatted_call,
                "tool_call": True
            }
        ]

        function_output = {
            "result": result
        }

        # Log the function execution
        ragmetrics_client._log_trace(
            input_messages=function_input,
            response=function_output,
            metadata_llm=None,
            contexts=None,
            duration=duration,
            tools=None,  
            callback_result={
                "input": formatted_call,
                "output": result
            }
        )
        return result

    return wrapper

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
        self.access_token = None
        self.base_url = 'https://ragmetrics.ai'
        self.logging_off = False
        self.metadata = None
        self.conversation_id = self.new_conversation()
        self.callback = None
    
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
        return self.conversation_id

    def _find_external_caller(self) -> str:
        """
        Find the first non-ragmetrics function in the call stack.
        
        Used internally to identify which user function triggered a logging event.

    
    Returns:
            str: The name of the first external function that called into ragmetrics,
                 or an empty string if none is found.
        """
        external_caller = ""
        frame = sys._getframe()
        while frame:
            module_name = frame.f_globals.get("__name__", "")
            if not module_name.startswith("ragmetrics"):
                external_caller = frame.f_code.co_name
                break
            frame = frame.f_back
        return external_caller

    def _log_trace(
            self, 
            input_messages, 
            response, 
            metadata_llm=None, 
            contexts=None,
            expected=None,
            duration=None, 
            callback_result=None,
            **kwargs
        ):
        """
        Log a trace of an LLM interaction to the RagMetrics API.
        
        This is the core logging method used by monitored LLM clients to record
        interactions with LLMs. It handles various formats and includes detailed
        metadata about the interaction.

    
    Args:
            input_messages: The input messages sent to the LLM (prompts, queries, etc.).
            response: The response received from the LLM. Follows OpenAI message list standard.            
            metadata_llm: Additional metadata about the LLM and the interaction.
            contexts: Context information or retrieved documents used in the interaction.
            expected: The expected output from the LLM.
            duration: The duration of the interaction in seconds.
            tools: Any tools or functions used during the interaction.
            callback_result: Optional processed results from a custom callback function.
            **kwargs: Additional keyword arguments to include in the trace.

    
    Raises:
            ValueError: If access token is missing.

    
    Returns:
            Response: The API response from logging the trace.
        """
        if self.logging_off:
            return

        if not self.access_token:
            raise ValueError("Missing access token. Please log in.")
        
        if isinstance(input_messages, list) and len(input_messages) == 1 \
            and not input_messages[0].get("tool_call", False) is True:
            self.new_conversation()

        # If response is a pydantic model, dump it. Supports both pydantic v2 and v1.
        if hasattr(response, "model_dump"):
            #Pydantic v2
            response_processed = response.model_dump() 
        if hasattr(response, "dict"):
            #Pydantic v1
            response_processed = response.dict()
        else:
            response_processed = response

        # Merge context and metadata dictionaries; treat non-dict values as empty.
        union_metadata = {}
        if isinstance(self.metadata, dict):
            union_metadata.update(self.metadata)
        if isinstance(metadata_llm, dict):
            union_metadata.update(metadata_llm)

        tools = kwargs.get('tools', None)
        conversation_id = kwargs.get('conversation_id', self.conversation_id)

        # Construct the payload with placeholders for callback result
        payload = {
            "raw": {
                "input": input_messages,
                "output": response_processed,
                "id": str(uuid.uuid4()),
                "duration": duration,
                "caller": self._find_external_caller()
            },
            "metadata": union_metadata,
            "contexts": contexts,
            "expected": expected,
            "tools": tools,
            "input": None,
            "output": None,
            "scores": None,
            "conversation_id": conversation_id
        }

        # Process callback_result if provided
        for key in ["input", "output", "expected"]:
            if callback_result and key in callback_result:
                payload[key] = callback_result[key]

        # Serialize
        payload_str = json.dumps(
            payload, 
            indent=4, 
            default=lambda o: (
                o.model_dump() if hasattr(o, "model_dump")
                else o.dict() if hasattr(o, "dict")
                else str(o)
            )
        )
        payload = json.loads(payload_str)

        log_resp = self._make_request(
            method='post',
            endpoint='/api/client/logtrace/',
            json=payload,
            headers={
                "Authorization": f"Token {self.access_token}",
                "Content-Type": "application/json"
            }
        )
        return log_resp

    def login(self, key, base_url=None, off=False):
        """
        Authenticate with the RagMetrics API.
        
        This method must be called before using other functionality that requires
        authentication. It can use an explicit API key or look for one in the
        RAGMETRICS_API_KEY environment variable.

    
    Args:
            key: The API key for authentication. Get this from your RagMetrics dashboard.
            base_url: Optional custom base URL for the API (default: https://ragmetrics.ai).
            off: Whether to disable logging entirely (default: False).

    
    Raises:
            ValueError: If no API key is provided or if the key is invalid.

    
    Returns:
            bool: True if login is successful.
        """
        if off:
            self.logging_off = True
        else:
            self.logging_off = False

        if not key:
            if 'RAGMETRICS_API_KEY' in os.environ:
                key = os.environ['RAGMETRICS_API_KEY']
        if not key:
            raise ValueError("Missing access token. Please get one at RagMetrics.ai.")

        if base_url:
            self.base_url = base_url
        elif 'RAGMETRICS_BASE_URL' in os.environ:
            self.base_url = os.environ['RAGMETRICS_BASE_URL']
        else:
            self.base_url = 'https://ragmetrics.ai'

        response = self._make_request(
            method='post',
            endpoint='/api/client/login/',
            json={"key": key}
        )

        if response.status_code == 200:
            self.access_token = key
            self.new_conversation()
            return True
        raise ValueError("Invalid access token. Please get a new one at RagMetrics.ai.")

    def _get_input_from_client_call(self, client_type, args, kwargs):
        """
        Extract input messages from client-specific call patterns.
        
        Args:
            client_type: Type of client ('openai', 'invoke', 'completion', 'agent', 'runner')
            args: Positional arguments passed to the client call
            kwargs: Keyword arguments passed to the client call
            
        Returns:
            The input messages in the appropriate format for the client type
        """
        if client_type in ['openai', 'completion']:
            return kwargs.get('messages')
        elif client_type == 'invoke':
            input_messages = kwargs.get('input')
            if input_messages is None and len(args) > 1:
                input_messages = args[1]
            return input_messages
        elif client_type == 'runner':
            # For Runner.run, args[1] is the input, which can be str, Message, or list[Message]
            if len(args) > 1:
                # Format the input into a consistent structure
                input_val = args[1]
                if isinstance(input_val, str):
                    return [{"role": "user", "content": input_val}]
                # Input could be a Message object or list of Messages
                # We'll assume the Runner.run method handles these formats correctly
                return input_val
            return kwargs.get('input')
        return None

    def _detect_client_type(self, client):
        """
        Detect the type of LLM client.
        
        Args:
            client: The client to detect type for
            
        Returns:
            tuple: (client_type, invoke_method)
                - client_type is one of: 'openai', 'invoke', 'lite', 'agent', 'runner'
                - invoke_method is the original method to call
                
        Raises:
            ValueError: If client type is not supported
        """

        # Detect MCP SDK- ClientSession object
        try:
            from mcp import ClientSession
            if client.__name__ == "ClientSession":
                return 'clientsession', None
        except ImportError:
            pass

        # Detect OpenAI Agents SDK- Runner object
        try:
            from agents import Runner
            if client is Runner or isinstance(client, Runner):
                # Runner instrumentation handled in _apply_client_wrapper
                return 'runner', None
        except ImportError:
            pass
            
        # Detect OpenAI Agents SDK - Agent object
        try:
            from agents import Agent
            if isinstance(client, Agent):
                return 'agent', None
        except ImportError:
            pass
            
        # OpenAI-style clients
        if hasattr(client, "chat") and hasattr(client.chat.completions, 'create'):
            return 'openai', type(client.chat.completions).create
        
        # LangChain-style clients with invoke method
        elif hasattr(client, "invoke") and callable(getattr(client, "invoke")):
            return 'invoke', getattr(client, "invoke")
        
        # LiteLLM-style clients
        elif hasattr(client, "completion"):
            return 'completion', client.completion
        
        else:
            raise ValueError("Unsupported client type")
    
    def _apply_client_wrapper(
            self,
            client,
            client_type,
            orig_invoke,
            wrapper_func
        ):
        """
        Apply the wrapper function to the appropriate method of the client.
        
        Args:
            client: The client to wrap
            client_type: Type of client ('openai', 'invoke', 'completion', 'agent', 'runner')
            orig_invoke: Original invoke method
            wrapper_func: The wrapper function to apply
            
        Returns:
            The modified client
        """
        if client_type == 'openai':
            client.chat.completions.create = \
                types.MethodType(wrapper_func, client.chat.completions)

        elif client_type == 'invoke':
            if isinstance(client, type):
                setattr(client, "invoke", wrapper_func)
            else:
                client.invoke = types.MethodType(wrapper_func, client)
                
        elif client_type == 'completion':
            if isinstance(client, type):
                setattr(client, "completion", wrapper_func)
            else:
                client.completion = types.MethodType(wrapper_func, client)

        elif client_type == 'runner':
            # Monitor OpenAI Async client as foundation for Agent SDK
            try:                
                from openai import AsyncOpenAI
                from ragmetrics.integrations.agents import monitor_agents
                async_client = AsyncOpenAI()
                monitor_agents(async_client)
            except Exception as e:
                print(f"Error setting up Runner monitoring: {e}")
        
        elif client_type == 'clientsession':
            # Monitor MCP Async client as foundation for MCP SDK
            try:                
                from mcp import ClientSession
                from ragmetrics.integrations.agents import monitor_mcp_server
                monitor_mcp_server(ClientSession)
            except Exception as e:
                print(f"Error setting up ClientSession monitoring: {e}")
            
            return client

    def _generate_wrapper(
        self,
        client_type,
        orig_invoke,
        callback,
        client=None
    ):
        """
        Create a wrapper function for various LLM client types.
        
        Args:
            client_type: Type of wrapper to generate ('openai', 'invoke', 'completion', 'runner')
            orig_invoke: Original method being wrapped
            callback: Callback function to process results
            client: Client object (required for OpenAI wrappers)
            
        Returns:
            A wrapper function for the specified client type
        """
        if client_type == 'openai':
            return lambda self_instance, *args, **kwargs: self._unified_sync_generation_wrapper(
                client_type, orig_invoke, callback, client, self_instance, *args, **kwargs
            )
        elif client_type == 'runner':
            return None
        elif client_type == 'clientsession':
            return None
        elif client_type == 'completion':
            # Special handling for LiteLLM completion
            return lambda *args, **kwargs: self._unified_sync_generation_wrapper(
                client_type, orig_invoke, callback, *args, **kwargs
            )
        else:
            return lambda *args, **kwargs: self._unified_sync_generation_wrapper(
                client_type, orig_invoke, callback, *args, **kwargs
            )

    def _unified_sync_generation_wrapper(
        self,
        client_type,
        orig_invoke,
        callback,
        *args,
        **kwargs
    ):
        """
        Unified wrapper method for all types of LLM API calls.
        
        Args:
            client_type: The type of wrapper ('openai', 'invoke', 'completion', 'runner_sync')
            orig_invoke: The original method being wrapped
            callback: The callback function to process results
            *args: Positional arguments to the original method
            **kwargs: Keyword arguments to the original method
            
        Returns:
            The response from the original method
        """
        # Extract RagMetrics-specific fields
        if client_type == 'completion':
            # litellm also accepts a metadata argument
            # so retain a copy in kwargs            
            if 'metadata' in kwargs:
                metadata_llm = kwargs['metadata'].copy()
                del kwargs['metadata']
            else:
                metadata_llm = None
        else:
            metadata_llm = kwargs.pop('metadata', None)
        
        contexts = kwargs.pop('contexts', None)
        expected = kwargs.pop('expected', None)
        
        # Execute client-specific generation
        start_time = time.time()
        if client_type == 'openai':
            client, self_instance, *invoke_args = args
            response = orig_invoke(self_instance, *invoke_args, **kwargs)
        
        elif client_type == 'completion':
            # For LiteLLM, we need to ensure we're not passing the model parameter twice
            # or passing the module itself as the model parameter
            
            # If first arg is litellm module itself, remove it
            if len(args) > 0 and hasattr(args[0], '__name__') and args[0].__name__ == 'litellm':
                args = args[1:]
                response = orig_invoke(*args, **kwargs)

            # If model is in both args and kwargs, remove it from kwargs
            elif len(args) > 0 and 'model' in kwargs:
                kwargs.pop('model', None)
                response = orig_invoke(*args, **kwargs)

            # Vanila completion
            else:
                response = orig_invoke(*args, **kwargs)
        else:
            response = orig_invoke(*args, **kwargs)

        duration = time.time() - start_time

        # Process and log the interaction
        input_messages = \
            self._get_input_from_client_call(client_type, args, kwargs)        

        cb_result = callback(input_messages, response)
        
        self._log_trace(
            input_messages=input_messages,
            response=response,
            duration=duration,
            callback_result=cb_result,
            metadata_llm=metadata_llm,
            contexts=contexts,
            expected=expected,
            **kwargs
        )
        return response

    def monitor(self, client, metadata, callback: Optional[Callable[[Any, Any], dict]] = None):
        """
        Monitor an LLM client's interactions by wrapping its API calls.
        
        This method creates a monitored version of an LLM client by wrapping its
        API methods. All interactions with the wrapped client will be automatically
        logged to RagMetrics.
        
        Supported client types:
        * OpenAI API clients (client.chat.completions.create)
        * LangChain (client.invoke)
        * LiteLLM (client.completion)

    
    Args:
            client: The LLM client to monitor.
            metadata: Additional metadata to include with each logged interaction.
            callback: Optional function to process inputs/outputs before logging.
                     Should accept (input, output) and return a dict with "input" and "output" keys.

    
    Raises:
            ValueError: If access token is missing or client type is unsupported.

    
    Returns:
            The wrapped client with monitoring enabled.
        """
        if not self.access_token:
            raise ValueError("Missing access token. Please get a new one at RagMetrics.ai.")
        if metadata is not None:
            self.metadata = metadata
        
        self.new_conversation()

        # Use default callback if none provided.
        if callback is None:
            callback = default_callback
        self.callback = callback

        # Detect client type and get original invoke method
        client_type, orig_invoke = self._detect_client_type(client)

        # Generate appropriate wrapper based on client type
        wrapper_func = self._generate_wrapper(client_type, orig_invoke, callback, client)

        # Apply the wrapper (handles runner internally)
        self._apply_client_wrapper(client, client_type, orig_invoke, wrapper_func)
        
        return client

    def _make_request(self, endpoint, method="post", **kwargs):
        """
        Make an HTTP request to the RagMetrics API.
        
        Used internally by various methods to communicate with the RagMetrics API.

    
    Args:
            endpoint: The API endpoint to call (e.g., "/api/client/login/").
            method: The HTTP method to use (default: "post").
            **kwargs: Additional arguments to pass to the requests library.

    
    Returns:
            Response: The HTTP response from the API.
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, **kwargs)
        return response

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

def login(key=None, base_url=None, off=False):
    """
    Create and authenticate a new RagMetricsClient instance.
    
    This is a convenience function that uses the global client instance.
    
    Example:
    
        .. code-block:: python
        
            # Login with explicit API key
            ragmetrics.login(key="your-api-key")
            
            # Login with environment variable (recommended)
            import os
            os.environ['RAGMETRICS_API_KEY'] = 'your-api-key' 
            ragmetrics.login()
            
            # Login with custom base URL (for testing/dev environments)
            ragmetrics.login(base_url="https://staging.ragmetrics.ai")
            
            # Disable logging (useful for testing)
            ragmetrics.login(off=True)


    Args:
        key: Optional API key for authentication. If not provided, will check
             the RAGMETRICS_API_KEY environment variable.
        base_url: Optional custom base URL for the API.
        off: Whether to disable logging (default: False).


    Returns:
        bool: True if login was successful.
        

    Raises:
        ValueError: If authentication fails.
    """
    return ragmetrics_client.login(key, base_url, off)

def monitor(client, metadata=None, callback: Optional[Callable[[Any, Any], dict]] = None):
    """
    Create a monitored version of an LLM client.
    
    This is a convenience function that uses the global client instance.
    
    Example for OpenAI:
    
        .. code-block:: python
        
            import openai
            import ragmetrics
            
            # Login to RagMetrics
            ragmetrics.login("your-api-key")
            
            # Create a monitored OpenAI client
            openai_client = ragmetrics.monitor(openai.OpenAI(), metadata={"app": "my-app"})
            
            # Use the monitored client as normal - all calls will be logged
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "user", "content": "What is the capital of France?"}],
                metadata={"request_id": "123"}, # Additional request-specific metadata
                contexts=["Paris is the capital of France"] # Context provided to the LLM
            )
    
    Example for LiteLLM:
    
        .. code-block:: python
        
            import litellm
            import ragmetrics
            
            # Login to RagMetrics
            ragmetrics.login("your-api-key")
            
            # Monitor LiteLLM
            ragmetrics.monitor(litellm, metadata={"client": "litellm"})
            
            # All completions will be logged
            response = litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "What is the capital of France?"}],
                metadata={"task": "geography_test"}
            )
        
    Example with custom callback:
    
        .. code-block:: python
        
            import openai
            import ragmetrics
            
            # Define a custom callback to process inputs/outputs before logging
            def my_callback(raw_input, raw_output):
                return {
                    "input": raw_input,
                    "output": raw_output,
                    "expected": "Paris" # Optional expected answer for automatic evaluation
                }
                
            # Monitor with custom callback
            openai_client = ragmetrics.monitor(
                openai.OpenAI(), 
                metadata={"app": "qa-system"},
                callback=my_callback
            )


    Args:
        client: The LLM client to monitor.
        metadata: Optional metadata for the monitoring session.
        callback: Optional callback function for custom processing.


    Returns:
        The wrapped client with monitoring enabled.
        

    Raises:
        ValueError: If not logged in or client type is unsupported.
    """
    return ragmetrics_client.monitor(client, metadata, callback)
