# test/integrations/openai_chat_wrapper/test_extract_details_empty_kwargs.py
import pytest
from ragmetrics.client_integrations.openai_chat_wrapper import _extract_openai_call_details

def test_extract_openai_call_details_empty_kwargs():
    kwargs = {}
    contexts, expected, llm_call_kwargs, additional_llm_metadata = _extract_openai_call_details(kwargs.copy())
    
    assert contexts is None
    assert expected is None
    assert llm_call_kwargs == {}
    assert additional_llm_metadata == {} 