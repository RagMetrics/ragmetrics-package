# test/integrations/openai_chat_wrapper/test_extract_details_only_ragmetrics_args.py
import pytest
from ragmetrics.client_integrations.openai_chat_wrapper import _extract_openai_call_details

def test_extract_openai_call_details_only_ragmetrics_args():
    kwargs = {
        "metadata": {"source": "test"},
        "contexts": ["c1"],
        "expected": "e1"
    }
    contexts, expected, llm_call_kwargs, additional_llm_metadata = _extract_openai_call_details(kwargs.copy())
    
    assert contexts == ["c1"]
    assert expected == "e1"
    assert llm_call_kwargs == {}
    assert additional_llm_metadata == {"source": "test"} 