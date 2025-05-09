# test/integrations/openai_chat_wrapper/test_extract_details_simple_call.py
import pytest
from ragmetrics.client_integrations.openai_chat_wrapper import _extract_openai_call_details

# Test _extract_openai_call_details function
def test_extract_openai_call_details_simple_call():
    kwargs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "max_tokens": 50,
        "metadata": {"user_id": 123}, # RagMetrics specific
        "contexts": ["doc1"],         # RagMetrics specific
        "expected": "Hi there"        # RagMetrics specific
    }
    contexts, expected, llm_call_kwargs, additional_llm_metadata = _extract_openai_call_details(kwargs)
    
    assert contexts == ["doc1"]
    assert expected == "Hi there"
    assert llm_call_kwargs == {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "max_tokens": 50
    }
    assert additional_llm_metadata == {"user_id": 123} # from metadata kwarg 