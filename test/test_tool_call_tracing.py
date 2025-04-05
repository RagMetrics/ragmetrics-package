import sys
import os
import json
import requests
from dotenv import load_dotenv

import ragmetrics
from ragmetrics import trace_function_call
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(".env")

# Login to RagMetrics - either with explicit key or from environment
ragmetrics.login()

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