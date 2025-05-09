# test/integrations/openai_chat_wrapper/test_extract_details_with_tool_choice_tools.py
import pytest
from ragmetrics.client_integrations.openai_chat_wrapper import _extract_openai_call_details

def test_extract_openai_call_details_with_tool_choice_and_tools():
    kwargs = {
        "model": "gpt-4-tools",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "tool_choice": "auto",
        "metadata": {"test_id": "tool-call-test"}
    }
    contexts, expected, llm_call_kwargs, additional_llm_metadata = _extract_openai_call_details(kwargs.copy())
    
    assert contexts is None
    assert expected is None
    assert llm_call_kwargs == {
        "model": "gpt-4-tools",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "tool_choice": "auto"
    }
    # Ensure that 'tools' and 'tool_choice' are also added to additional_llm_metadata 
    # if they are present in the original llm_call_kwargs, as per _extract_openai_call_details logic.
    expected_meta = {
        "test_id": "tool-call-test",
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "tool_choice": "auto"
    }
    assert additional_llm_metadata == expected_meta 