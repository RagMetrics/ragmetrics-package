# test/integrations/test_openai_chat_wrapper.py
import pytest
from unittest.mock import patch, MagicMock, ANY
import time

# Modules to test
from ragmetrics.api import RagMetricsClient 
# Updated import for the specific sync wrapper
from ragmetrics.client_integrations.openai_chat_wrapper import wrap_openai_chat_completions_create 
from ragmetrics.utils import default_callback

# --- Fixtures --- 

@pytest.fixture
def logged_in_rm_client():
    """Provides a logged-in RagMetricsClient instance with a mock _log_trace."""
    client = RagMetricsClient()
    client.access_token = "test-token"
    client.logging_off = False
    # Mock the internal logging method that wrappers will call
    client._log_trace = MagicMock() 
    # Also mock _alog_trace for completeness, though not used by sync tests here
    client._alog_trace = MagicMock() 
    return client

@pytest.fixture
def mock_openai_completions_object():
    """Creates a mock OpenAI completions object with a create method."""
    # This is the object that our wrapper will operate on (e.g., client.chat.completions)
    mock_original_create_method = MagicMock(return_value="original_response")
    completions_object = MagicMock()
    completions_object.create = mock_original_create_method
    # To allow type(completions_object).create later if needed, though our wrapper doesn't need it directly
    # type(completions_object).create = mock_original_create_method 
    return completions_object, mock_original_create_method

# --- Tests --- 

def test_openai_wrapper_applies_and_calls_original(logged_in_rm_client, mock_openai_completions_object):
    rm_client = logged_in_rm_client
    completions_obj, original_create_mock = mock_openai_completions_object
    
    # Apply the wrapper directly to the completions object
    wrapped = wrap_openai_chat_completions_create(
        rm_client, 
        completions_obj, # Pass the completions object that has the .create method
        default_callback
    )
    
    assert wrapped is True # Check wrapper reported success
    assert hasattr(completions_obj, 'create')
    # Verify the original mock is NOT the current attribute (it should be wrapped)
    assert completions_obj.create.__func__ is not original_create_mock # create_sync_wrapper binds it
    
    # --- Call the wrapped method --- 
    input_messages = [{"role": "user", "content": "Hi OpenAI"}]
    response = completions_obj.create(
        model="gpt-test", 
        messages=input_messages,
        metadata={"call_meta": "data"}, 
        contexts=["context1"],        
        temperature=0.5                
    )
    
    assert response == "original_response" 
    
    original_create_mock.assert_called_once()
    call_args, call_kwargs = original_create_mock.call_args
    assert call_kwargs['model'] == "gpt-test"
    assert call_kwargs['messages'] == input_messages
    assert call_kwargs['temperature'] == 0.5
    assert 'metadata' not in call_kwargs 
    assert 'contexts' not in call_kwargs 
    
    rm_client._log_trace.assert_called_once()
    log_args_tuple, log_kwargs_dict = rm_client._log_trace.call_args
    
    assert log_kwargs_dict['input_messages'] == input_messages
    # The output_extractor for openai chat now returns a dict or the message object
    # For this mock, original_response is a string. Extractor will return it as is.
    assert log_kwargs_dict['response'] == "original_response"
    assert log_kwargs_dict['metadata_llm']["call_meta"] == "data"
    # Check for OpenAI specific params in additional_llm_metadata
    assert log_kwargs_dict['metadata_llm']["temperature"] == 0.5
    assert log_kwargs_dict['model_name'] == "gpt-test" # Extracted by dynamic_llm_details_extractor
    assert log_kwargs_dict['contexts'] == ["context1"]
    assert log_kwargs_dict['expected'] is None
    assert isinstance(log_kwargs_dict['duration'], float) and log_kwargs_dict['duration'] >= 0
    assert log_kwargs_dict['tools'] is None 
    # The callback result structure depends on default_callback and its inputs.
    # default_callback(input_messages, "original_response")
    # Since the output_extractor now might return a dict, let's adjust.
    # For this test, the output_extractor will return the raw "original_response" string because it doesn't match the OpenAI choice structure.
    expected_callback_res = default_callback(input_messages, "original_response")
    assert log_kwargs_dict['callback_result'] == expected_callback_res

def test_openai_wrapper_handles_client_metadata(logged_in_rm_client, mock_openai_completions_object):
    rm_client = logged_in_rm_client
    rm_client.metadata = {"client_meta": "global"} 
    completions_obj, original_create_mock = mock_openai_completions_object

    wrap_openai_chat_completions_create(rm_client, completions_obj, default_callback)
    
    completions_obj.create(
        model="gpt-test", 
        messages=[{"role": "user", "content": "Test"}],
        metadata={"call_meta": "local"} 
    )

    rm_client._log_trace.assert_called_once()
    log_args, log_kwargs = rm_client._log_trace.call_args
    
    # Correct expected metadata_llm: merge client, dynamic(additional), user
    # dynamic_extractor adds {"model": "gpt-test"} to additional_llm_metadata
    expected_metadata = {
        "client_meta": "global",  # From rm_client.metadata
        "model": "gpt-test",     # From dynamic_extractor -> additional_llm_metadata
        "call_meta": "local"       # From user_call_metadata
    }
    assert log_kwargs['metadata_llm'] == expected_metadata
    assert log_kwargs['model_name'] == "gpt-test" # Check model_name separately

def test_openai_wrapper_handles_tools(logged_in_rm_client, mock_openai_completions_object):
    rm_client = logged_in_rm_client
    completions_obj, original_create_mock = mock_openai_completions_object

    wrap_openai_chat_completions_create(rm_client, completions_obj, default_callback)
    
    tools_def = [{"type": "function", "function": {"name": "test_func"}}]
    input_messages = [{"role": "user", "content": "Use tool"}]
    completions_obj.create(
        model="gpt-test", 
        messages=input_messages,
        tools=tools_def
    )

    rm_client._log_trace.assert_called_once()
    log_args, log_kwargs = rm_client._log_trace.call_args
    assert log_kwargs['tools'] == tools_def
    assert log_kwargs['model_name'] == "gpt-test"
    
    # Correct expected metadata_llm:
    # dynamic_extractor adds {"model": "gpt-test", "tools": tools_def} to additional_llm_metadata
    expected_metadata = {
        "model": "gpt-test",
        "tools": tools_def
    }
    assert log_kwargs['metadata_llm'] == expected_metadata
    
    # Correct callback assertion
    expected_callback_res = default_callback(input_messages, "original_response")
    assert log_kwargs['callback_result'] == expected_callback_res 