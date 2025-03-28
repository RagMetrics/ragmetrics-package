import types
import requests
import sys
import os
import time
import json
import uuid
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

def default_input(raw_input):
    if not raw_input:
        return None
    if isinstance(raw_input, list):
        def format_message(m):
            if isinstance(m, dict):
                role = m.get("role", "unknown")
                content = m.get("content", "")
            elif hasattr(m, "role") and hasattr(m, "content"):
                role = getattr(m, "role", "unknown")
                content = getattr(m, "content", "")
            else:
                role = "unknown"
                content = str(m)
            return f"{role}: {content}"
        return "\n".join(format_message(m) for m in raw_input)
    elif isinstance(raw_input, dict):
        role = raw_input.get("role", "unknown")
        content = raw_input.get("content", "")
        return f"{role}: {content}"
    elif hasattr(raw_input, "role") and hasattr(raw_input, "content"):
        return f"{raw_input.role}: {raw_input.content}"
    else:
        return str(raw_input)

def default_output(raw_response):
    if not raw_response:
        return None
    # OpenAI chat completion
    if hasattr(raw_response, "choices") and raw_response.choices:
        try:
            # OpenAI ChatCompletion objects expose choices as objects with a message attribute.
            return raw_response.choices[0].message.content
        except Exception as e:
            logger.error("Error extracting content from response.choices: %s", e)
    # If response has a text attribute, return it (for non-chat completions)
    if hasattr(raw_response, "text"):
        return raw_response.text
    # Fallback to checking for a content attribute (if it's a simple object)
    if hasattr(raw_response, "content"):
        return raw_response.content
    # Unable to determine response content, log and return the raw response.
    return raw_response

def default_callback(raw_input, raw_output) -> dict:
    return {
        "input": default_input(raw_input),
        "output": default_output(raw_output)
    }

def trace_function_call(func):
    """
    Decorator to trace function execution and log structured input/output.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Prepare structured input format
        function_input = [
            {
                "role": "user",
                "content": f"{func.__name__} called with args: {json.dumps(args)}, kwargs: {json.dumps(kwargs)}"
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
                "input": function_input, 
                "output": default_output(function_output)
            },
            trace_type="retrieval"
        )
        return result

    return wrapper

class RagMetricsClient:
    def __init__(self):
        self.access_token = None
        self.base_url = 'https://ragmetrics.ai'
        self.logging_off = False
        self.metadata = None
        self.conversation_id = str(uuid.uuid4())
    
    def new_conversation(self):
        """Reset the conversation_id to a new UUID."""
        self.conversation_id = str(uuid.uuid4())

    def _find_external_caller(self) -> str:
        """
        Walk the stack and return the first function name that does not belong to 'ragmetrics'.
        If none is found, returns an empty string.
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

    def _log_trace(self, input_messages, response, metadata_llm, contexts, duration, tools, callback_result=None,trace_type = "generation", **kwargs):
        if self.logging_off:
            return

        if not self.access_token:
            raise ValueError("Missing access token. Please log in.")
        
        if trace_type!= "retrieval":
            if isinstance(input_messages, list) and len(input_messages) == 1:
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
            "tools": tools,
            "input": None,
            "output": None,
            "expected": None,            
            "scores": None,
            "conversation_id": self.conversation_id,
            "trace_type":"generation"
        }

        # Process callback_result if provided
        for key in ["input", "output", "expected"]:
            if key in callback_result:
                payload[key] = callback_result[key]

        if (("output" not in payload or payload["output"] is None) and tools is not None):
            try:
                trace_type = "tools"
                if hasattr(response, "choices") and response.choices:
                    payload["output"] = response.choices[0].message.tool_calls
                elif isinstance(response, dict) and "choices" in response:
                    payload["output"] = response["choices"][0]["message"]["tool_calls"]
            except Exception as e:
                logger.error("Error extracting tool_calls from response: %s", e)
        
        payload["trace_type"] = trace_type
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

        # Use data=payload_str (which is a string) and specify the content-type header.
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

    def _original_llm_invoke(self, client):
        """
        Returns the original LLM invocation function from the client.
        Checks first for chat-style (OpenAI), then for a callable invoke (LangChain),
        and finally for a module-level 'completion' function.
        Works whether the client is a class or an instance.
        """
        if hasattr(client, "chat") and hasattr(client.chat.completions, 'create'):
            return type(client.chat.completions).create
        elif hasattr(client, "invoke") and callable(getattr(client, "invoke")):
            return getattr(client, "invoke")
        elif hasattr(client, "completion"):
            return client.completion
        else:
            raise ValueError("Unsupported client")

    def _make_request(self, endpoint, method="post", **kwargs):
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, **kwargs)
        return response

    def monitor(self, client, metadata, callback: Optional[Callable[[Any, Any], dict]] = None):
        if not self.access_token:
            raise ValueError("Missing access token. Please get a new one at RagMetrics.ai.")
        if metadata is not None:
            self.metadata = metadata
        
        self.new_conversation()

        # Use default callback if none provided.
        if callback is None:
            callback = default_callback

        orig_invoke = self._original_llm_invoke(client)

        # Chat-based clients (OpenAI)
        if hasattr(client, "chat") and hasattr(client.chat.completions, 'create'):
            def openai_wrapper(self_instance, *args, **kwargs):
                start_time = time.time()
                metadata_llm = kwargs.pop('metadata', None)
                contexts = kwargs.pop('contexts', None)
                response = orig_invoke(self_instance, *args, **kwargs)
                duration = time.time() - start_time
                input_messages = kwargs.get('messages')
                cb_result = callback(input_messages, response)
                tools= kwargs.pop('tools', None)
                self._log_trace(input_messages, response, metadata_llm, contexts, duration, tools, callback_result=cb_result, **kwargs)
                return response
            client.chat.completions.create = types.MethodType(openai_wrapper, client.chat.completions)
        
        # LangChain-style clients that support invoke (class or instance)
        elif hasattr(client, "invoke") and callable(getattr(client, "invoke")):
            def invoke_wrapper(*args, **kwargs):
                start_time = time.time()
                metadata_llm = kwargs.pop('metadata', None) 
                contexts = kwargs.pop('contexts', None)
                response = orig_invoke(*args, **kwargs)
                duration = time.time() - start_time
                tools = kwargs.pop('tools', None)
                input_messages = kwargs.pop('input', None)
                cb_result = callback(input_messages, response)
                self._log_trace(input_messages, response, metadata_llm, contexts, duration, tools, callback_result=cb_result, **kwargs)
                return response
            if isinstance(client, type):
                setattr(client, "invoke", invoke_wrapper)
            else:
                client.invoke = types.MethodType(invoke_wrapper, client)
        
        # LiteLLM-style clients (module-level function)
        elif hasattr(client, "completion"):
            def lite_wrapper(*args, **kwargs):
                start_time = time.time()
                metadata_llm = kwargs.pop('metadata', None)
                contexts = kwargs.pop('contexts', None)
                response = orig_invoke(*args, **kwargs)
                duration = time.time() - start_time
                tools = kwargs.pop('tools', None)
                input_messages = kwargs.get('messages')
                cb_result = callback(input_messages, response)
                self._log_trace(input_messages, response, metadata_llm, contexts, duration, tools, callback_result=cb_result, **kwargs)
                return response
            client.completion = lite_wrapper
        
        #Unknown client
        else:
            raise ValueError("Unsupported client")

class RagMetricsObject:
    object_type: str = None

    def to_dict(self):
        """Convert the object into a dict for API payload.
        Subclasses should implement this."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict):
        """Instantiate an object from API data.
        Subclasses may override this if needed."""
        return cls(**data)

    def save(self):
        """
        Saves the object using a API endpoint.
        The endpoint is determined by the object's type.
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
        Downloads the object from the API using an endpoint.

        Examples:
          - MyObject.download(123) uses 123 as the id.
          - MyObject.download(name="foo") uses "foo" as the name.
          - MyObject.download(123, name="foo") uses 123 as the id.
          
        Raises an error if neither parameter is provided.
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
        
# Wrapper calls for simpler calling
ragmetrics_client = RagMetricsClient()

def login(key=None, base_url=None, off=False):
    return ragmetrics_client.login(key, base_url, off)

def monitor(client, metadata=None, callback: Optional[Callable[[Any, Any], dict]] = None):
    return ragmetrics_client.monitor(client, metadata, callback)
