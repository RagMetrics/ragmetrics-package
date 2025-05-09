# test/integrations/openai_chat_wrapper/test_extract_details_no_ragmetrics_args.py
import pytest
from ragmetrics.client_integrations.openai_chat_wrapper import _extract_openai_call_details

def test_extract_openai_call_details_no_ragmetrics_args():
    kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Another test"}],
        "n": 2
    }
    contexts, expected, llm_call_kwargs, additional_llm_metadata = _extract_openai_call_details(kwargs.copy()) # Pass a copy
    
    assert contexts is None
    assert expected is None
    assert llm_call_kwargs == {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Another test"}],
        "n": 2
    }
    assert additional_llm_metadata == {} # No metadata kwarg means empty dict here 