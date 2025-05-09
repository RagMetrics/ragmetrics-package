import json
import os # Import os to check environment variables
import requests

import ragmetrics # ragmetrics module itself
from ragmetrics import trace_function_call # Specific decorator
from ragmetrics.trace import Trace # Import Trace from ragmetrics.trace
from openai import OpenAI

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

def test_tool_call_flow(ragmetrics_test_client): # Use the fixture from conftest.py
    """Tests the full OpenAI tool calling flow with RagMetrics tracing."""

    openai_api_client = OpenAI()
    ragmetrics_test_client.monitor(openai_api_client)

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

    messages = [{"role": "user", "content": "What's the weather like in LA?"}]

    # First call to OpenAI
    completion = openai_api_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assert completion.choices[0].message.tool_calls is not None

    tool_call = completion.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    
    # Call to the traced function
    result = get_weather(**args)

    assert isinstance(result, (float, int))

    messages.append(completion.choices[0].message)  # append model's function call message
    messages.append({                               # append result message
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    })

    # Second call to OpenAI
    second_completion = openai_api_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )    
    
    assert second_completion.choices[0].message.content is not None

    test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"

    if test_mock:
        assert len(ragmetrics_test_client.test_logged_trace_ids) == 0, \
            "When TEST_MOCK is true, no trace IDs should be logged"
        ragmetrics_logger = ragmetrics.api.logger # Get logger for info
        ragmetrics_logger.info("TEST_MOCK is true, skipping trace download and conversation ID assertions.")
    else:
        logged_ids = ragmetrics_test_client.test_logged_trace_ids
        assert len(logged_ids) == 3, f"Expected 3 trace IDs, got {len(logged_ids)}"

        fetched_traces = []
        for trace_id in logged_ids:
            trace_obj = Trace.download(id=trace_id)
            assert trace_obj is not None, f"Failed to download trace for ID: {trace_id}"
            assert trace_obj.id == int(trace_id) # Verify ID on the downloaded object
            fetched_traces.append(trace_obj)
        
        assert len(fetched_traces) == 3, "Should have downloaded all 3 traces"

        conversation_ids = [t.conversation_id for t in fetched_traces]
        
        # Check that all traces have the same conversation ID
        first_conv_id = conversation_ids[0]
        assert first_conv_id is not None, "First trace conversation ID should not be None"
        assert all(cid == first_conv_id for cid in conversation_ids), \
            f"Not all traces share the same conversation ID. Got: {conversation_ids}"
        
        ragmetrics_logger = ragmetrics.api.logger # Get logger for info
        ragmetrics_logger.info(f"All {len(logged_ids)} traces successfully share conversation ID: {first_conv_id}")
