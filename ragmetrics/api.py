import types
import requests
import litellm


class RagMetricsClient:
    def __init__(self):
        self.access_token = None
        self.site_domain = 'http://localhost:8000'
        self.logging_off = False

    def _log_response(self, context, response, **kwargs):
        if not self.logging_off:
            input_data = kwargs.copy()
            input_data.update({
                "context": context,
                "response": response.model_dump() if hasattr(response, 'model_dump') else response
            })
            self.log_trace(trace_json=input_data)

    def login(self, key, off):
        if off:
            self.logging_off = True

        response = self._make_request(
            method='post',
            endpoint='/api/client_login/',
            json={"key": key}
        )

        if response.status_code == 200:
            self.access_token = key
            return True
        raise ValueError("Invalid access token.")

    def fetch_api_key(self):
        if not self.access_token:
            raise ValueError("You must log in first.")

        response = requests.get(
            f"{self.site_domain}/api/fetch_api_key/",
            headers={"Authorization": f"Token {self.access_token}"}
        )
        if response.status_code == 200:
            return response.json().get("api_key")
        raise Exception("Failed to fetch API key.")

    def _get_original_method(self, client):
        if client == litellm:
            return litellm.completion
        elif hasattr(client.chat.completions, 'create'):
            # Get the unbound method from the class
            return type(client.chat.completions).create
        else:
            raise ValueError("Unsupported client")

    def _make_request(self, method, endpoint, **kwargs):
        url = f"{self.site_domain}{endpoint}"
        response = requests.request(method, url, **kwargs)
        return response

    def monitor(self, client, context):
        original_method = self._get_original_method(client)

        if client == litellm:
            # Handle LiteLLM (module-level function)
            def litellm_wrapper(*args, **kwargs):
                response = original_method(*args, **kwargs)
                self._log_response(context, response, **kwargs)
                return response
            client.completion = litellm_wrapper
        else:
            # Handle OpenAI (instance method)
            def openai_wrapper(self_instance, *args, **kwargs):
                response = original_method(self_instance, *args, **kwargs)
                self._log_response(context, response, **kwargs)
                return response

            # Bind the wrapper to the Completions instance
            client.chat.completions.create = types.MethodType(
                openai_wrapper, client.chat.completions
            )

    def log_trace(self, trace_json):
        if not self.access_token:
            raise ValueError("You must log in first.")

        response = self._make_request(
            method='post',
            endpoint='/api/monitor/',
            json=trace_json,
            headers={"Authorization": f"Token {self.access_token}"}
        )
        return response

ragmetrics_client = RagMetricsClient()

def login(key=None, off=False):
    return ragmetrics_client.login(key, off)

def monitor(client, context=None):
    return ragmetrics_client.monitor(client, context)