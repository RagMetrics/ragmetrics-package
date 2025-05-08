# test/test_api.py
import pytest
import os
import time # For duration testing
import uuid # For conversation IDs
from unittest.mock import patch, MagicMock, ANY # ANY for flexible matching
import requests # Import requests library for mocking

from ragmetrics.api import RagMetricsClient, login, RagMetricsConfigError, RagMetricsAuthError, RagMetricsAPIError, ragmetrics_client, monitor # Added monitor import
from ragmetrics.utils import default_callback # Import default_callback for testing

# --- Fixtures --- 

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to temporarily set environment variables."""
    monkeypatch.setenv("RAGMETRICS_API_KEY", "test_env_key")
    monkeypatch.setenv("RAGMETRICS_BASE_URL", "https://test-env.ragmetrics.ai")
    return monkeypatch # Return it for potential further use in tests

@pytest.fixture
def mock_successful_login_response():
    """Fixture for a successful mocked API login response."""
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None # Mock successful check
    mock_resp.json.return_value = {"message": "Login successful"} # Example success payload
    return mock_resp

@pytest.fixture
def mock_failed_login_response_401():
    """Fixture for a 401 Unauthorized mocked API login response."""
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 401
    mock_resp.text = "Invalid API Key"
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_resp)
    return mock_resp

@pytest.fixture
def mock_failed_login_response_500(monkeypatch):
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error Text"
    
    # Mock the request object that would have led to this response
    mock_request_obj = MagicMock(spec=requests.Request)
    mock_request_obj.url = "http://mockurl/api/client/login/"
    mock_response.request = mock_request_obj # Attach mock_request_obj to mock_response

    # Configure raise_for_status to raise an HTTPError that includes the mock_response
    http_error = requests.exceptions.HTTPError("500 Server Error for url: http://mockurl/api/client/login/", response=mock_response)
    mock_response.raise_for_status.side_effect = http_error
    return mock_response

@pytest.fixture
def logged_in_client(mock_successful_login_response):
    """Provides a RagMetricsClient instance that is already logged in."""
    with patch('requests.request') as mock_request:
        mock_request.return_value = mock_successful_login_response
        client = RagMetricsClient()
        client.login(key="test_key_for_logged_in")
        # Reset mock call count after login for subsequent tests
        mock_request.reset_mock()
    return client

# --- Tests for RagMetricsClient Initialization --- 

def test_client_init_defaults():
    client = RagMetricsClient()
    assert client.access_token is None
    assert client.base_url == 'https://ragmetrics.ai'
    assert not client.logging_off
    assert client.metadata is None
    assert isinstance(client.conversation_id, str)

def test_client_init_env_override(mock_env_vars): # Uses the mock env fixture
    client = RagMetricsClient() # Init reads env vars
    assert client.base_url == "https://test-env.ragmetrics.ai"

# --- Tests for RagMetricsClient.login --- 

@patch('requests.request') # Mock the actual request call
def test_client_login_success_with_key(mock_request, mock_successful_login_response):
    mock_request.return_value = mock_successful_login_response
    client = RagMetricsClient()
    assert client.login(key="test_key") is True
    assert client.access_token == "test_key"
    assert client.base_url == 'https://ragmetrics.ai' # Default was used
    assert not client.logging_off
    mock_request.assert_called_once_with('post', 'https://ragmetrics.ai/api/client/login/', json={"key": "test_key"}, timeout=15)

@patch('requests.request')
def test_client_login_success_with_env_key(mock_request, mock_successful_login_response, mock_env_vars):
    mock_request.return_value = mock_successful_login_response
    client = RagMetricsClient() # Reads env var for base_url on init
    assert client.login() is True # No key passed, uses env var
    assert client.access_token == "test_env_key"
    assert client.base_url == "https://test-env.ragmetrics.ai" # Env var was used
    assert not client.logging_off
    mock_request.assert_called_once_with('post', 'https://test-env.ragmetrics.ai/api/client/login/', json={"key": "test_env_key"}, timeout=15)

@patch('requests.request')
def test_client_login_arg_overrides_env(mock_request, mock_successful_login_response, mock_env_vars):
    mock_request.return_value = mock_successful_login_response
    client = RagMetricsClient() 
    assert client.login(key="arg_key", base_url="https://arg.ragmetrics.ai") is True
    assert client.access_token == "arg_key"
    assert client.base_url == "https://arg.ragmetrics.ai"
    assert not client.logging_off
    mock_request.assert_called_once_with('post', 'https://arg.ragmetrics.ai/api/client/login/', json={"key": "arg_key"}, timeout=15)

def test_client_login_missing_key(monkeypatch):
    client = RagMetricsClient()
    # Ensure RAGMETRICS_API_KEY is not set for this test
    monkeypatch.delenv("RAGMETRICS_API_KEY", raising=False)
    with pytest.raises(RagMetricsConfigError, match="Missing API key"):
        client.login()
    assert client.access_token is None
    assert client.logging_off is True # Should disable logging

@patch('requests.request')
def test_client_login_auth_error(mock_request, mock_failed_login_response_401):
    mock_request.return_value = mock_failed_login_response_401
    client = RagMetricsClient()
    with pytest.raises(RagMetricsAuthError):
        client.login(key="wrong_key")
    assert client.access_token is None
    assert client.logging_off is True

@patch('requests.request')
def test_client_login_api_error(mock_request, mock_failed_login_response_500):
    mock_request.return_value = mock_failed_login_response_500
    client = RagMetricsClient()
    with pytest.raises(RagMetricsAPIError):
        client.login(key="test_key")
    assert client.access_token is None
    assert client.logging_off is True

@patch('requests.request', side_effect=requests.exceptions.ConnectionError("Network Error"))
def test_client_login_connection_error(mock_request):
    client = RagMetricsClient()
    with pytest.raises(RagMetricsAPIError): # Should be caught and re-raised as APIError
        client.login(key="test_key")
    assert client.access_token is None
    assert client.logging_off is True

def test_client_login_off():
    client = RagMetricsClient()
    assert client.login(off=True) is True
    assert client.logging_off is True
    assert client.access_token is None

# --- Tests for global login convenience function --- 
# These implicitly test the global ragmetrics_client instance

@patch('ragmetrics.api.ragmetrics_client.login') # Patch the method on the global instance
def test_global_login_calls_instance_login(mock_instance_login):
    mock_instance_login.return_value = True
    result = login(key="global_key")
    assert result is True
    mock_instance_login.assert_called_once_with(key="global_key", base_url=None, off=False)

@patch('ragmetrics.api.ragmetrics_client.login')
def test_global_login_raises(mock_instance_login):
    mock_instance_login.side_effect = RagMetricsAuthError("Global Auth Failed")
    with pytest.raises(RagMetricsAuthError):
        login(key="bad_global_key")
    mock_instance_login.assert_called_once_with(key="bad_global_key", base_url=None, off=False)

# --- Tests for RagMetricsClient._log_trace --- 

@patch('ragmetrics.api.RagMetricsClient._make_request')
def test_log_trace_basic_payload(mock_make_request, logged_in_client):
    mock_make_request.return_value = MagicMock(status_code=200) # Mock successful API call
    
    client = logged_in_client
    initial_conv_id = client.conversation_id
    
    input_msg = [{"role": "user", "content": "log this"}]
    response_msg = {"output": "logged"}
    metadata = {"key": "value"}
    contexts = ["context doc 1"]
    expected = "expected answer"
    duration = 1.23
    tools_def = [{"type": "function", "function": {"name": "dummy"}}]
    cb_result = default_callback(input_msg, response_msg) # Use default callback
    
    client._log_trace(
        input_messages=input_msg,
        response=response_msg,
        metadata_llm=metadata,
        contexts=contexts,
        expected=expected,
        duration=duration,
        tools=tools_def,
        callback_result=cb_result,
        conversation_id=initial_conv_id # Explicitly pass conversation_id
    )
    
    mock_make_request.assert_called_once()
    call_args = mock_make_request.call_args
    assert call_args.kwargs['method'] == 'post'
    assert call_args.kwargs['endpoint'] == '/api/client/logtrace/'
    
    payload = call_args.kwargs['json']
    assert payload['raw']['input'] == input_msg
    assert payload['raw']['output'] == response_msg
    assert isinstance(payload['raw']['id'], str)
    assert payload['raw']['duration'] == duration
    # assert payload['raw']['caller'] # Caller is hard to test deterministically
    assert payload['metadata'] == metadata # Assumes no client.metadata was set
    assert payload['contexts'] == contexts
    assert payload['expected'] == expected
    assert payload['tools'] == tools_def
    assert payload['input'] == cb_result['input']
    assert payload['output'] == cb_result['output']
    assert payload['conversation_id'] == initial_conv_id # Heuristic didn't reset
    assert 'force_new_conversation' not in payload # Should not be in payload

@patch('ragmetrics.api.RagMetricsClient._make_request')
def test_log_trace_force_new_conversation(mock_make_request, logged_in_client):
    mock_make_request.return_value = MagicMock(status_code=200)
    client = logged_in_client
    original_conv_id = client.conversation_id

    client._log_trace(
        input_messages=[{"role": "user", "content": "second interaction"}],
        response="response",
        metadata_llm=None, contexts=None, expected=None, duration=0.5, tools=None,
        callback_result=default_callback([{"role": "user", "content": "second interaction"}], "response"),
        force_new_conversation=True # Explicitly force new conversation
    )

    mock_make_request.assert_called_once()
    payload = mock_make_request.call_args.kwargs['json']
    assert payload['conversation_id'] != original_conv_id
    assert uuid.UUID(payload['conversation_id']) # Check it's a valid UUID format

@patch('ragmetrics.api.RagMetricsClient._make_request')
def test_log_trace_heuristic_new_conversation(mock_make_request, logged_in_client):
    mock_make_request.return_value = MagicMock(status_code=200)
    client = logged_in_client
    original_conv_id = client.conversation_id
    
    # Log first message - should reset conversation ID via heuristic
    client._log_trace(
        input_messages=[{"role": "user", "content": "first interaction"}],
        response="first response",
        metadata_llm=None, contexts=None, expected=None, duration=0.5, tools=None,
        callback_result=default_callback([{"role": "user", "content": "first interaction"}], "first response")
    )
    first_call_payload = mock_make_request.call_args_list[0].kwargs['json']
    new_conv_id = first_call_payload['conversation_id']
    assert new_conv_id != original_conv_id
    assert client.conversation_id == new_conv_id # Client's current ID should be updated

    # Log second message (e.g., assistant response) - should continue conversation
    client._log_trace(
        input_messages=[{"role": "assistant", "content": "follow up"}], # Assistant role should not reset
        response="another response",
        metadata_llm=None, contexts=None, expected=None, duration=0.3, tools=None,
        callback_result=default_callback([{"role": "assistant", "content": "follow up"}], "another response")
    )
    second_call_payload = mock_make_request.call_args_list[1].kwargs['json']
    assert second_call_payload['conversation_id'] == new_conv_id # Should be same as the new ID
    assert client.conversation_id == new_conv_id # Client ID remains

@patch('ragmetrics.api.RagMetricsClient._make_request')
def test_log_trace_explicit_conversation_id_overrides_heuristic(mock_make_request, logged_in_client):
    mock_make_request.return_value = MagicMock(status_code=200)
    client = logged_in_client
    original_conv_id = client.conversation_id
    fixed_conv_id = "my_fixed_conversation_123"

    # This input would normally trigger the heuristic
    input_msg = [{"role": "user", "content": "User starts interaction"}]
    
    client._log_trace(
        input_messages=input_msg,
        response="response",
        metadata_llm=None, contexts=None, expected=None, duration=0.5, tools=None,
        callback_result=default_callback(input_msg, "response"),
        conversation_id=fixed_conv_id # Explicitly provide ID
    )

    mock_make_request.assert_called_once()
    payload = mock_make_request.call_args.kwargs['json']
    assert payload['conversation_id'] == fixed_conv_id
    assert client.conversation_id == original_conv_id # Client's default ID should NOT change

def test_log_trace_when_logged_off(logged_in_client):
    client = logged_in_client
    client.logging_off = True
    with patch('ragmetrics.api.RagMetricsClient._make_request') as mock_make_request:
        result = client._log_trace(input_messages="in", response="out", metadata_llm={}, contexts=[], expected=None, duration=0.1, tools=[], callback_result={})
        assert result is None
        mock_make_request.assert_not_called()

def test_log_trace_when_not_logged_in():
    client = RagMetricsClient() # Not logged in
    assert client.access_token is None
    with patch('ragmetrics.api.RagMetricsClient._make_request') as mock_make_request:
        result = client._log_trace(input_messages="in", response="out", metadata_llm={}, contexts=[], expected=None, duration=0.1, tools=[], callback_result={})
        assert result is None
        mock_make_request.assert_not_called()

# --- Tests for RagMetricsClient.monitor dispatch --- 
# These tests check if monitor calls the *correct* wrapper finder/function
# They do NOT test the wrappers themselves, just the dispatch logic.

@patch('ragmetrics.api.find_integration') # Mock the registry finder
def test_monitor_dispatches_correctly(mock_find, logged_in_client):
    client_to_monitor = MagicMock()
    # Add the method that the integration expects to wrap
    client_to_monitor.some_method = MagicMock() # The original method on the client

    # Define the mock wrapper function separately for easier assertion
    mock_wrapper_for_some_method = MagicMock(return_value=True)

    mock_integration = {
        "name": "Mock Integration",
        "client_type_check": lambda c: True, 
        "target_object_path": None, # Target is the client itself
        "methods_to_wrap": {
            "some_method": mock_wrapper_for_some_method
        },
        "async_methods_to_wrap": {}
    }
    mock_find.return_value = mock_integration
    
    # Use a real callback for testing the argument passing
    test_callback = default_callback 

    monitored_client = logged_in_client.monitor(client_to_monitor, callback=test_callback)

    mock_find.assert_called_once_with(client_to_monitor)
    # Check that the wrap_func from the found integration was called correctly
    mock_wrapper_for_some_method.assert_called_once_with(logged_in_client, client_to_monitor, test_callback)
    # Assert client is returned
    assert monitored_client is client_to_monitor

@patch('ragmetrics.api.find_integration', return_value=None) # Mock finder to return None
def test_monitor_no_integration_found(mock_find, logged_in_client):
    client_to_monitor = "some_unsupported_client_type"
    
    # Use pytest's LogCapture fixture (or caplog) if you want to assert the warning
    # For simplicity, we just check the return value here
    monitored_client = logged_in_client.monitor(client_to_monitor)
    
    mock_find.assert_called_once_with(client_to_monitor)
    assert monitored_client is client_to_monitor # Should return client unmodified

# --- Tests for global convenience functions (Assume existing login tests are here) --- 

@patch('ragmetrics.api.ragmetrics_client.monitor') # Patch the method on the global instance
def test_global_monitor_calls_instance_monitor(mock_instance_monitor):
    mock_client = MagicMock()
    mock_instance_monitor.return_value = mock_client # Monitor returns the client
    # Call the global monitor function (imported now)
    result = monitor(mock_client, metadata={"global": "meta"}) 
    mock_instance_monitor.assert_called_once_with(mock_client, metadata={"global": "meta"}, callback=None)
    assert result is mock_client

# Next: Tests for decorators, models, and individual client wrappers 