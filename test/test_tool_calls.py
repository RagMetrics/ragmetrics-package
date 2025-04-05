import sys
import os
import json


# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ragmetrics.api import _extract_content

# Test with OpenAI API response as a dictionary
openai_dict_response = {
  "id": "chatcmpl-BIIPNjll8CCnfVReFy7THofZgAcHJ",
  "model": "gpt-4o-mini-2024-07-18",
  "usage": {
    "total_tokens": 84,
    "prompt_tokens": 59,
    "completion_tokens": 25,
    "prompt_tokens_details": {
      "audio_tokens": 0,
      "cached_tokens": 0
    },
    "completion_tokens_details": {
      "audio_tokens": 0,
      "reasoning_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  },
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "audio": None,
        "content": None,
        "refusal": None,
        "tool_calls": [
          {
            "id": "call_XJgxp98050xUo7Hlsfto6U8J",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": '{"latitude":48.8566,"longitude":2.3522}'
            }
          }
        ],
        "annotations": [],
        "function_call": None
      },
      "logprobs": None,
      "finish_reason": "tool_calls"
    }
  ],
  "created": 1743700365,
  "service_tier": "default",
  "system_fingerprint": "fp_86d0290411"
}

# Test with OpenAI Python client response (object format)
class Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class ToolCall:
    def __init__(self, id, type, function):
        self.id = id
        self.type = type
        self.function = function

class Message:
    def __init__(self, role, content, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls

class Choice:
    def __init__(self, message):
        self.message = message

class OpenAIResponse:
    def __init__(self, choices):
        self.choices = choices

# Create object format with tool_calls
tool_call = ToolCall(
    id="call_XJgxp98050xUo7Hlsfto6U8J",
    type="function",
    function=Function(
        name="get_weather", 
        arguments='{"latitude":48.8566,"longitude":2.3522}'
    )
)

openai_obj_response = OpenAIResponse([
    Choice(
        Message(
            role="assistant",
            content=None,
            tool_calls=[tool_call]
        )
    )
])

# Create an object-style response with normal content
normal_obj_response = OpenAIResponse([
    Choice(
        Message(
            role="assistant",
            content="This is a normal response",
            tool_calls=None
        )
    )
])

# Test the function with different types of responses
print("== Testing dict-style response with tool_calls ==")
result1 = _extract_content(openai_dict_response, "output")
print(f"Result: {result1}")
print()

print("== Testing object-style response with tool_calls ==")
result2 = _extract_content(openai_obj_response, "output")
print(f"Result: {result2}")
print()

print("== Testing object-style response with normal content ==")
result3 = _extract_content(normal_obj_response, "output")
print(f"Result: {result3}")
print()

# Test with a more complex function call
complex_args = '{"query":"temperature in Paris","units":"celsius","details":true}'
complex_tool_call = ToolCall(
    id="call_complex",
    type="function",
    function=Function(
        name="search_weather", 
        arguments=complex_args
    )
)

complex_response = OpenAIResponse([
    Choice(
        Message(
            role="assistant",
            content=None,
            tool_calls=[complex_tool_call]
        )
    )
])

print("== Testing complex function arguments ==")
result4 = _extract_content(complex_response, "output")
print(f"Result: {result4}") 