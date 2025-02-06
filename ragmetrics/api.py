import types
import requests
import sys
import os
import time


class RagMetricsClient:
    def __init__(self):
        self.access_token = None
        self.base_url = 'https://ragmetrics.ai'
        self.logging_off = False
        self.context = None

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

    def _log_trace(self, input_messages, response, context, metadata, duration, **kwargs):
        if self.logging_off:
            return

        if not self.access_token:
            raise ValueError("Missing access token. Please log in.")

        # If response is a pydantic model, dump it. Supports both pydantic v2 and v1.
        if hasattr(response, "model_dump"):
            dump = response.model_dump()
        elif hasattr(response, "dict"):
            dump = response.dict()
        else:
            dump = response

        trace = kwargs
        trace.update({
            "input": input_messages,
            "response": dump,
            "context": context,
            "metadata": metadata,
            "caller": self._find_external_caller(),
            "duration": kwargs.get("duration", None)
        })

        log_resp = self._make_request(
            method='post',
            endpoint='/api/client/logtrace/',
            json=trace,
            headers={"Authorization": f"Token {self.access_token}"}
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

    def _make_request(self, method, endpoint, **kwargs):
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, **kwargs)
        return response

    def monitor(self, client, context):
        if not self.access_token:
            raise ValueError("Missing access token. Please get a new one at RagMetrics.ai.")
        if context is not None:
            self.context = context

        orig_invoke = self._original_llm_invoke(client)

        # Handle chat-based clients (OpenAI)
        if hasattr(client, "chat") and hasattr(client.chat.completions, 'create'):
            def openai_wrapper(self_instance, *args, **kwargs):
                start_time = time.time()
                metadata = kwargs.pop('metadata', None)
                response = orig_invoke(self_instance, *args, **kwargs)
                duration = time.time() - start_time
                input_messages = kwargs.get('messages')
                self._log_trace(input_messages, response, context, metadata, duration, **kwargs)
                return response
            client.chat.completions.create = types.MethodType(openai_wrapper, client.chat.completions)
        # Handle LangChain-style clients that support invoke (class or instance)
        elif hasattr(client, "invoke") and callable(getattr(client, "invoke")):
            def invoke_wrapper(*args, **kwargs):
                start_time = time.time()
                metadata = kwargs.pop('metadata', None)
                # If a messages list is provided, convert it to the input format expected by BaseChatModel.
                messages = kwargs.pop('messages', None)
                if messages is not None:
                    input_str = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                    kwargs["input"] = input_str
                response = orig_invoke(*args, **kwargs)
                duration = time.time() - start_time
                self._log_trace(messages, response, context, metadata, duration, **kwargs)
                return response
            if isinstance(client, type):
                setattr(client, "invoke", invoke_wrapper)
            else:
                client.invoke = types.MethodType(invoke_wrapper, client)
        # Handle lite-style clients (module-level function)
        elif hasattr(client, "completion"):
            def lite_wrapper(*args, **kwargs):
                start_time = time.time()
                metadata = kwargs.pop('metadata', None)
                response = orig_invoke(*args, **kwargs)
                duration = time.time() - start_time
                input_messages = kwargs.get('messages')
                self._log_trace(input_messages, response, context, metadata, duration, **kwargs)
                return response
            client.completion = lite_wrapper
        else:
            raise ValueError("Unsupported client")


# Wrapper calls for simpler calling
ragmetrics_client = RagMetricsClient()


def login(key=None, base_url=None, off=False):
    return ragmetrics_client.login(key, base_url, off)


def monitor(client, context=None):
    return ragmetrics_client.monitor(client, context)
