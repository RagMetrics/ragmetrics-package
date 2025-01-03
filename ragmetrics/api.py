import os
import types
import requests
from litellm import completion


class RagMetricsClient:
    def __init__(self):
        self.access_token = None
        self.site_domain = 'http://20.83.187.1:5000'
        self.original_create_method = None
        self.logging_off = False

    def login(self, key, off):
        """
        Log into RagMetrics using the provided access token.
        This function checks if the token exists in the Token table.
        """
        if off:
            self.logging_off = True

        response = self._make_request(
            method='post',
            endpoint='/api/client_login/',
            json={"key": key}
        )

        if response.status_code == 200:
            self.access_token = key
            os.environ["OPENAI_API_KEY"] = self.fetch_api_key()  # Fetch API key after successful login
            return True
        raise ValueError("Invalid access token.")

    def fetch_api_key(self):
        """
        Fetch the OpenAI API key from the Django backend.
        """
        if not self.access_token:
            raise ValueError("You must log in first.")

        response = requests.get(f"{self.site_domain}/api/fetch_api_key/",
                                headers={"Authorization": f"Token {self.access_token}"})
        if response.status_code == 200:
            return response.json().get("api_key")
        raise Exception("Failed to fetch API key.")

    def monitor(self, client, context):
        """
        Patch the OpenAI-compatible client to automatically log chat completions.
        This method assumes the user may provide their own client with an API key.
        """
        # Check if the client has an API key set
        if not client.api_key:
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key is not set. Please log in first.")
            client.api_key = os.environ["OPENAI_API_KEY"]  # Set the API key for the client

        # Define a new method to handle completion requests
        def new_create_method(*args, **kwargs):
            # Prepare the data to send to the Litellm proxy
            input_data = {
                "off": self.logging_off,
                "model": kwargs.get('model', []),
                "messages": kwargs.get('messages', []),
                "metadata": kwargs.get('metadata', {}),
                "context": context,
                "output": '',
            }

            # Log the call to RagMetrics
            trace_response = self.log_trace(api_client=client, input_data=input_data)

            # Call the Litellm proxy to get the response
            response = completion(model=kwargs.get('model'), messages=kwargs.get('messages'), api_key=os.environ["OPENAI_API_KEY"])
            return response  # Return the Litellm response as the actual response

        # Patch the create method for OpenAI-compatible clients
        if hasattr(client.chat.completions, 'create'):
            client.chat.completions.create = types.MethodType(new_create_method, client.chat.completions)
        elif hasattr(client.chat, 'complete'):
            client.chat.complete = types.MethodType(new_create_method, client.chat)

    def log_trace(self, api_client, input_data):
        """
        Monitor LLM calls from the provided client and log them in the traces table.
        The actual response is fetched from the traces API.
        """
        if not self.access_token:
            raise ValueError("You must log in first.")

        response = self._make_request(
            method='post',
            endpoint='/api/monitor/',
            json=input_data,
            headers={"Authorization": f"Token {self.access_token}"}
        )

        return response

    def _make_request(self, method, endpoint, **kwargs):
        """
        Helper method to make API requests to the RagMetrics server.
        """
        url = f"{self.site_domain}{endpoint}"
        response = requests.request(method, url, **kwargs)
        return response


# Module-level variable to hold the client instance
ragmetrics_client = RagMetricsClient()


def login(key, off):
    """
    Log into RagMetrics using the provided access token.
    """
    return ragmetrics_client.login(key, off)


def monitor(client, context):
    """
    Start monitoring LLM calls from the provided client.
    """
    return ragmetrics_client.monitor(client, context)
