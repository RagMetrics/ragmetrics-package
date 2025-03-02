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
        input_str = "\n".join(f"{m['role']}: {m['content']}" for m in raw_input)
    else:
        input_str = str(raw_input)
    return input_str

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

class RagMetricsClient:
    def __init__(self):
        self.access_token = None
        self.base_url = 'https://ragmetrics.ai'
        self.logging_off = False
        self.metadata = None

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

    def _log_trace(self, input_messages, response, metadata_llm, contexts, duration, callback_result=None, **kwargs):
        if self.logging_off:
            return

        if not self.access_token:
            raise ValueError("Missing access token. Please log in.")

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
            "input": None,
            "output": None,
            "expected": None,            
            "scores": None
        }

        # Process callback_result if provided
        for key in ["input", "output", "expected"]:
            if key in callback_result:
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
                self._log_trace(input_messages, response, metadata_llm, contexts, duration, callback_result=cb_result, **kwargs)
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
                input_messages = kwargs.pop('input', None)
                cb_result = callback(input_messages, response)
                self._log_trace(input_messages, response, metadata_llm, contexts, duration, callback_result=cb_result, **kwargs)
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
                input_messages = kwargs.get('messages')
                cb_result = callback(input_messages, response)
                self._log_trace(input_messages, response, metadata_llm, contexts, duration, callback_result=cb_result, **kwargs)
                return response
            client.completion = lite_wrapper
        
        #Unknown client
        else:
            raise ValueError("Unsupported client")

# Wrapper calls for simpler calling
ragmetrics_client = RagMetricsClient()

def login(key=None, base_url=None, off=False):
    return ragmetrics_client.login(key, base_url, off)

def monitor(client, metadata=None, callback: Optional[Callable[[Any, Any], dict]] = None):
    return ragmetrics_client.monitor(client, metadata, callback)
