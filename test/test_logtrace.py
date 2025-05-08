#!pip install ragmetrics-client
#!pip install openai litellm langchain_groq

import ragmetrics
from openai import OpenAI
import litellm
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import pytest
import requests
from unittest.mock import patch, MagicMock
from ragmetrics.api import RagMetricsClient, RagMetricsAPIError
load_dotenv(".env")

# os.environ['RAGMETRICS_API_KEY'] = 'your_ragmetrics_key'
# os.environ['GROQ_API_KEY'] = 'your_groq_key'
# os.environ['OPENAI_API_KEY'] = 'your_openai_key'

# Login with the API key from environment
ragmetrics.login(off=True)

@pytest.fixture
def client():
    """Provides a RagMetricsClient instance for tests."""
    # Return the global client instance configured by ragmetrics.login()
    return ragmetrics.ragmetrics_client

def create_messages(client_name, country):
    return [
        {"role": "system", "content": f"You are a helpful assistant based on {client_name}."},
        {"role": "user", "content": f"What is the capital of {country}?"}
    ]

# Define a callback that takes raw input and output and returns processed fields.
def my_callback(raw_input, raw_output):
    # Your custom post-processing logic here. For example:
    processed = {
         "input": raw_input,
         "output": raw_output
    }
    return processed

# Test OpenAI client (chat-based)
openai_client = OpenAI()
# Pass RagMetrics metadata via monitor, not directly to create()
ragmetrics.monitor(openai_client, metadata={"client": "openai", "test_case": "openai"})
messages = create_messages("OpenAI", "France")
resp = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
    # metadata={"client": "OpenAI Native", "step": "1"} # Removed - OpenAI API rejects this directly
)
print(resp)

# Test LiteLLM client (module-level function)
# Pass RagMetrics metadata via monitor
ragmetrics.monitor(litellm, metadata={"client": "litellm", "test_case": "litellm"})
messages = create_messages("LiteLLM", "Germany")
resp = litellm.completion(
    model="gpt-3.5-turbo",
    messages=messages
    # metadata={"task": "test", "step": "litellm"} # Removed - LiteLLM passes its own metadata internally if needed
)
print(resp)

# Test LangChain-style client
# Pass RagMetrics metadata via monitor
ragmetrics.monitor(ChatGroq, metadata={"client": "langchain", "test_case": "langchain"}, callback=my_callback)
langchain_model = ChatGroq(model="llama3-8b-8192")
messages = create_messages("LangChain", "Italy")
resp = langchain_model.invoke(
    input=messages
    # metadata={"task": "test", "step": "langchain"} # Removed - Pass via monitor() or wrapper's metadata kwarg
)
print(resp)
