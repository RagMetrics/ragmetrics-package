import sys
import os
import json
import requests
from dotenv import load_dotenv
import pytest
from unittest.mock import patch, MagicMock

import ragmetrics
from ragmetrics import trace_function_call
from openai import OpenAI
from ragmetrics.api import RagMetricsClient

# Load environment variables from .env file
load_dotenv(".env")

# Attempt to login using environment variables
ragmetrics.login(off=True)

@pytest.fixture
def logged_in_client():
    """Fixture to provide a logged-in RagMetrics client instance."""
    # If login(off=True) was called, this might not be truly logged in.
    # Tests using this fixture might need to mock network calls or ensure login state.
    client = ragmetrics.ragmetrics_client 
    # Ensure logging is not off for tests that need it (override module-level setting)
    client.logging_off = False 
    # Mock _log_trace if tests focus on wrapper logic, not backend communication
    # client._log_trace = MagicMock()
    # Provide a dummy token if needed for internal checks, even if login was off
    if not client.access_token:
        client.access_token = "dummy-test-token" 
    return client

# Example 1: Weather API function
@trace_function_call
def get_weather(latitude, longitude):
    """
    Get the current temperature for a location.
    This function will be automatically traced by RagMetrics.
    """
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    )
    data = response.json()
    return data['current']['temperature_2m']

# Run the test functions
# OpenAI function calling (tool use)
client = OpenAI()
ragmetrics.monitor(client)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current temperature for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"}
                },
                "required": ["latitude", "longitude"],
            },
        },
    }
]

messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

tool_call = completion.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
result = get_weather(**args)

messages.append(completion.choices[0].message)  # append model's function call message
messages.append({                               # append result message
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": str(result)
})

second_completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)    