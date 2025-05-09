# test/test_api.py
import pytest
import os
import time # For duration testing
import uuid # For conversation IDs
import json # Import json for dumps
from unittest.mock import patch, MagicMock, ANY # ANY for flexible matching
import requests # Import requests library for mocking

from ragmetrics import RagMetricsClient # Corrected import
from ragmetrics.api import RagMetricsAuthError, RagMetricsAPIError, RagMetricsConfigError, RagMetricsError # Corrected import and added RagMetricsConfigError and RagMetricsError
from ragmetrics.utils import default_callback # Already present, ensure it stays
from ragmetrics.api import login, monitor # This is used for global login and monitor tests

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

def test_client_init_defaults(monkeypatch):
    # Isolate from environment variables to test internal defaults
    monkeypatch.delenv("RAGMETRICS_API_KEY", raising=False)
    monkeypatch.delenv("RAGMETRICS_BASE_URL", raising=False)

    client = RagMetricsClient()
    assert client.access_token is None
    assert client.base_url == 'https://ragmetrics.ai' # Internal default
    assert not client.logging_off
    assert client.metadata is None
    assert isinstance(client.conversation_id, str)

def test_client_init_env_override(mock_env_vars): # Uses the mock env fixture
    client = RagMetricsClient() # Init reads env vars
    assert client.base_url == "https://test-env.ragmetrics.ai"

# --- Tests for RagMetricsClient.login --- 

@patch('ragmetrics.api.requests.post') # Patched requests.post directly
def test_client_login_success_with_key(mock_api_post_request, mock_successful_login_response, monkeypatch):
    mock_api_post_request.return_value = mock_successful_login_response
    
    monkeypatch.delenv("RAGMETRICS_API_KEY", raising=False)
    monkeypatch.delenv("RAGMETRICS_BASE_URL", raising=False)

    client = RagMetricsClient()
    assert client.base_url == 'https://ragmetrics.ai'

    assert client.login(key="test_key") is True
    assert client.access_token == "test_key"
    assert client.base_url == 'https://ragmetrics.ai'
    assert not client.logging_off
    mock_api_post_request.assert_called_once_with(
        'https://ragmetrics.ai/api/client/login/', 
        data=json.dumps({"key": "test_key"}),
        headers=ANY,
        params=None
    )

@patch('ragmetrics.api.requests.post') # Patched requests.post directly
def test_client_login_success_with_env_key(mock_api_post_request, mock_successful_login_response, mock_env_vars):
    mock_api_post_request.return_value = mock_successful_login_response
    
    client = RagMetricsClient()
    assert client.base_url == "https://test-env.ragmetrics.ai"

    assert client.login() is True
    assert client.access_token == "test_env_key"
    assert not client.logging_off
    mock_api_post_request.assert_called_once_with(
        'https://test-env.ragmetrics.ai/api/client/login/', 
        data=json.dumps({"key": "test_env_key"}),
        headers=ANY,
        params=None
    )

@patch('ragmetrics.api.requests.post') # Patched requests.post directly
def test_client_login_arg_overrides_env(mock_api_post_request, mock_successful_login_response, mock_env_vars):
    mock_api_post_request.return_value = mock_successful_login_response
    
    client = RagMetricsClient() 
    assert client.base_url == "https://test-env.ragmetrics.ai" 

    assert client.login(key="arg_key", base_url="https://arg.ragmetrics.ai") is True
    assert client.access_token == "arg_key"
    assert client.base_url == "https://arg.ragmetrics.ai"
    assert not client.logging_off
    mock_api_post_request.assert_called_once_with(
        'https://arg.ragmetrics.ai/api/client/login/', 
        data=json.dumps({"key": "arg_key"}),
        headers=ANY,
        params=None
    )

def test_client_login_missing_key(monkeypatch):
    client = RagMetricsClient()
    # Ensure RAGMETRICS_API_KEY is not set for this test
    monkeypatch.delenv("RAGMETRICS_API_KEY", raising=False)
    with pytest.raises(RagMetricsConfigError, match="Missing API key"):
        client.login()
    assert client.access_token is None
    assert client.logging_off is True # Should disable logging

@patch('ragmetrics.api.requests.post') # Patched requests.post directly
def test_client_login_auth_error(mock_api_post_request, mock_failed_login_response_401, monkeypatch):
    mock_api_post_request.return_value = mock_failed_login_response_401
    
    monkeypatch.delenv("RAGMETRICS_API_KEY", raising=False)
    monkeypatch.delenv("RAGMETRICS_BASE_URL", raising=False)
    client = RagMetricsClient()
    assert client.base_url == 'https://ragmetrics.ai'

    with pytest.raises(RagMetricsAuthError, match=r"Authentication error for POST https://ragmetrics.ai/api/client/login/: 401 - Invalid API Key..."):
        client.login(key="wrong_key")
    assert client.access_token is None
    assert client.logging_off is True
    mock_api_post_request.assert_called_once_with(
        'https://ragmetrics.ai/api/client/login/',
        data=json.dumps({"key": "wrong_key"}),
        headers=ANY,
        params=None
    )

@patch('ragmetrics.api.requests.post') # Patched requests.post directly
def test_client_login_api_error(mock_api_post_request, mock_failed_login_response_500, monkeypatch):
    mock_api_post_request.return_value = mock_failed_login_response_500
    
    monkeypatch.delenv("RAGMETRICS_API_KEY", raising=False)
    monkeypatch.delenv("RAGMETRICS_BASE_URL", raising=False)
    client = RagMetricsClient()
    assert client.base_url == 'https://ragmetrics.ai'

    expected_error_msg = "API error for POST https://ragmetrics.ai/api/client/login/: 500 - Internal Server Error Text..."
    with pytest.raises(RagMetricsAPIError, match=expected_error_msg):
        client.login(key="test_key")
    assert client.access_token is None
    assert client.logging_off is True
    mock_api_post_request.assert_called_once_with(
        'https://ragmetrics.ai/api/client/login/',
        data=json.dumps({"key": "test_key"}),
        headers=ANY,
        params=None
    )

@patch('ragmetrics.api.requests.post', side_effect=requests.exceptions.ConnectionError("Network Error")) # Patched post directly
def test_client_login_connection_error(mock_api_post_request, monkeypatch):
    monkeypatch.delenv("RAGMETRICS_API_KEY", raising=False)
    monkeypatch.delenv("RAGMETRICS_BASE_URL", raising=False)
    client = RagMetricsClient()
    assert client.base_url == 'https://ragmetrics.ai'

    with pytest.raises(RagMetricsError, match=r"An unexpected error occurred during API key validation: Request failed: Network Error"):
        client.login(key="test_key")
    assert client.access_token is None
    assert client.logging_off is True
    mock_api_post_request.assert_called_once_with(
        'https://ragmetrics.ai/api/client/login/',
        data=json.dumps({"key": "test_key"}),
        headers=ANY,
        params=None
    )

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

def test_log_trace_basic_payload(ragmetrics_test_client): # Removed mock_make_request from args
    client = ragmetrics_test_client 
    initial_conv_id = client.conversation_id
    
    # Isolate client.metadata for this test to ensure exact match for metadata_llm
    original_client_metadata = client.metadata
    client.metadata = None # Or set to a specific dict if the test implies it

    input_msg = [{"role": "user", "content": "log this"}]
    response_msg = {"output": "logged"}
    metadata = {"key": "value"}
    contexts = ["context doc 1"]
    expected = "expected answer"
    duration = 1.23
    tools_def = [{"type": "function", "function": {"name": "dummy"}}]
    cb_result = default_callback(input_msg, response_msg)
    
    if hasattr(client, 'test_logged_trace_ids'):
        client.test_logged_trace_ids = []

    with patch.object(client, '_make_request', return_value={"id": "mock-trace-id-basic"}) as mock_make_request:
        client._log_trace(
            input_messages=input_msg,
            response=response_msg,
            metadata_llm=metadata,
            contexts=contexts,
            expected=expected,
            duration=duration,
            tools=tools_def,
            callback_result=cb_result,
            conversation_id=initial_conv_id 
        )
    
        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if client.logging_off:
                mock_make_request.assert_not_called()
                assert len(client.test_logged_trace_ids) == 0
            else:
                mock_make_request.assert_called_once()
                _pos_args, called_kwargs = mock_make_request.call_args # call_args is (args_tuple, kwargs_dict)
                assert called_kwargs['method'] == 'post'
                assert called_kwargs['endpoint'] == '/api/client/logtrace/'
                payload = called_kwargs['json_payload'] 
                assert payload['raw']['input'] == input_msg
                assert payload['raw']['output'] == response_msg
                assert isinstance(payload['raw']['id'], str)
                assert payload['raw']['duration'] == duration
                assert payload['metadata'] == metadata # This is metadata_llm passed to _log_trace
                assert payload['contexts'] == contexts
                assert payload['expected'] == expected
                assert payload['tools'] == tools_def
                assert payload['input'] == cb_result['input']
                assert payload['output'] == cb_result['output']
                assert payload['conversation_id'] == initial_conv_id 
                assert 'force_new_conversation' not in payload

                assert len(client.test_logged_trace_ids) == 1
                assert client.test_logged_trace_ids[0] == "mock-trace-id-basic"
        else: # test_mock is True
            mock_make_request.assert_not_called()
            assert len(client.test_logged_trace_ids) == 0

    # Restore original client metadata
    client.metadata = original_client_metadata

def test_log_trace_force_new_conversation(ragmetrics_test_client): # Removed mock
    client = ragmetrics_test_client
    original_conv_id = client.conversation_id
    if hasattr(client, 'test_logged_trace_ids'):
        client.test_logged_trace_ids = []

    with patch.object(client, '_make_request', return_value={"id": "mock-trace-id-force"}) as mock_make_request:
        client._log_trace(
            input_messages=[{"role": "user", "content": "second interaction"}],
            response="response",
            metadata_llm=None, contexts=None, expected=None, duration=0.5, tools=None,
            callback_result=default_callback([{"role": "user", "content": "second interaction"}], "response"),
            force_new_conversation=True
        )
    
        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if client.logging_off:
                mock_make_request.assert_not_called()
                assert len(client.test_logged_trace_ids) == 0
            else:
                mock_make_request.assert_called_once()
                _pos_args, called_kwargs = mock_make_request.call_args 
                payload = called_kwargs['json_payload'] 
                assert payload['conversation_id'] != original_conv_id
                assert uuid.UUID(payload['conversation_id'])
                assert len(client.test_logged_trace_ids) == 1
                assert client.test_logged_trace_ids[0] == "mock-trace-id-force"
        else:
            mock_make_request.assert_not_called()
            assert len(client.test_logged_trace_ids) == 0

def test_log_trace_heuristic_new_conversation(ragmetrics_test_client): 
    client = ragmetrics_test_client 
    original_conv_id = client.conversation_id # ID before any calls
    if hasattr(client, 'test_logged_trace_ids'):
        client.test_logged_trace_ids = []
        
    with patch.object(client, '_make_request') as mock_make_request:
        mock_make_request.return_value = {"id": "mock-trace-id-heuristic-1"} # Corrected: No escaped quotes needed here
        # First call - should trigger heuristic to start a new conversation
        client._log_trace(
            input_messages=[{"role": "user", "content": "first interaction"}],
            response="first response",
            metadata_llm=None, contexts=None, expected=None, duration=0.5, tools=None,
            callback_result=default_callback([{"role": "user", "content": "first interaction"}], "first response")
        )
        # Capture the NEW conversation ID set by the heuristic
        heuristic_conv_id = client.conversation_id
        
        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if client.logging_off:
                mock_make_request.assert_not_called()
                assert len(client.test_logged_trace_ids) == 0
            else:
                assert mock_make_request.call_count == 1
                _pos_args, called_kwargs_1 = mock_make_request.call_args_list[0]
                payload1 = called_kwargs_1['json_payload'] 
                assert payload1['conversation_id'] == heuristic_conv_id
                assert heuristic_conv_id != original_conv_id 
                assert len(client.test_logged_trace_ids) == 1
                assert client.test_logged_trace_ids[0] == "mock-trace-id-heuristic-1"

                # --- Second call --- 
                mock_make_request.return_value = {"id": "mock-trace-id-heuristic-2"} # Corrected
                client.test_logged_trace_ids = [] 
                second_call_conv_id_to_pass = heuristic_conv_id 
                client._log_trace(
                    input_messages=[{"role": "user", "content": "second interaction in same conversation"}],
                    response="second response",
                    metadata_llm=None, contexts=None, expected=None, duration=0.5, tools=None,
                    callback_result=default_callback([{"role": "user", "content": "second interaction in same conversation"}], "second response"),
                    conversation_id=second_call_conv_id_to_pass 
                )
                assert mock_make_request.call_count == 2
                _pos_args, called_kwargs_2 = mock_make_request.call_args_list[1] 
                payload2 = called_kwargs_2['json_payload'] 
                assert payload2['conversation_id'] == second_call_conv_id_to_pass
                assert len(client.test_logged_trace_ids) == 1
                assert client.test_logged_trace_ids[0] == "mock-trace-id-heuristic-2"

def test_log_trace_explicit_conversation_id_overrides_heuristic(ragmetrics_test_client): # Removed mock
    client = ragmetrics_test_client
    if hasattr(client, 'test_logged_trace_ids'):
        client.test_logged_trace_ids = []
    explicit_conv_id = str(uuid.uuid4())

    with patch.object(client, '_make_request', return_value={"id": "mock-trace-id-explicit"}) as mock_make_request:
        client._log_trace(
            input_messages=[{"role": "user", "content": "explicit id test"}],
            response="response to explicit id test",
            metadata_llm=None, contexts=None, expected=None, duration=0.1, tools=None,
            callback_result=default_callback([{"role": "user", "content": "explicit id test"}], "response to explicit id test"),
            conversation_id=explicit_conv_id
        )
    
        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if client.logging_off:
                mock_make_request.assert_not_called()
                assert len(client.test_logged_trace_ids) == 0
            else:
                mock_make_request.assert_called_once()
                _pos_args, called_kwargs = mock_make_request.call_args 
                payload = called_kwargs['json_payload'] 
                assert payload['conversation_id'] == explicit_conv_id
                assert client.conversation_id != explicit_conv_id
                assert len(client.test_logged_trace_ids) == 1
                assert client.test_logged_trace_ids[0] == "mock-trace-id-explicit"
        else: # test_mock is True
            mock_make_request.assert_not_called()
            assert len(client.test_logged_trace_ids) == 0

def test_log_trace_when_logged_off(ragmetrics_test_client): # Use ragmetrics_test_client
    client = ragmetrics_test_client
    test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"

    # Ensure test_logged_trace_ids is clean
    if hasattr(client, 'test_logged_trace_ids'):
        client.test_logged_trace_ids = []

    original_logging_off_state = client.logging_off

    with patch.object(client, '_make_request') as mock_make_request:
        if not test_mock:
            # If not mocking, we need to explicitly turn logging off for this test scenario
            client.logging_off = True
        
        # Regardless of test_mock, if logging_off is True (either by fixture or by manual set), no call should happen
        assert client.logging_off is True, "Logging should be off for this test scenario"

        client._log_trace(
            input_messages=[{"role": "user", "content": "no log"}],
            response="this won\'t be logged",
            metadata_llm=None, contexts=None, expected=None, duration=0.1, tools=None,
            callback_result=default_callback([{"role": "user", "content": "no log"}], "this won\'t be logged")
        )
        mock_make_request.assert_not_called()
        assert len(client.test_logged_trace_ids) == 0

    # Restore original logging_off state if we changed it, to not affect other tests if test_mock was false
    if not test_mock:
        client.logging_off = original_logging_off_state

# For test_log_trace_when_not_logged_in, it creates its own client, so it doesn't need ragmetrics_test_client
# and tests a state before the fixture would normally run.
# It should remain as is, ensuring it cleans up env vars if it modifies them (which it does via monkeypatch).

# @patch('requests.request') # This is for the original version
# def test_log_trace_when_not_logged_in(mock_request, monkeypatch): # Original signature
def test_log_trace_when_not_logged_in(monkeypatch): # Updated: no mock_request needed as _make_request isn't called
    # Ensure no RAGMETRICS_API_KEY is set for this specific test, to simulate not being logged in.
    monkeypatch.delenv("RAGMETRICS_API_KEY", raising=False)
    # Create a fresh client that will not have an access token and will not attempt to login via env var
    client = RagMetricsClient() # Instantiate without api_key argument
    client.access_token = None # Ensure it's None for the test's purpose
    # client.logged_in = False # This attribute doesn't exist on RagMetricsClient
    client.logging_off = False # Critical: logging must be ON for the internal check of access_token to matter

    # Ensure test_logged_trace_ids is clean
    if hasattr(client, 'test_logged_trace_ids'):
        client.test_logged_trace_ids = []

    # We expect _make_request not to be called because logged_in is False
    with patch.object(client, '_make_request') as mock_make_request:
        client._log_trace(
            input_messages=[{"role": "user", "content": "no log due to no auth"}],
            response="this won\'t be logged due to no auth",
            metadata_llm=None, contexts=None, expected=None, duration=0.1, tools=None,
            callback_result=default_callback([{"role": "user", "content": "no log due to no auth"}], "this won\'t be logged due to no auth")
        )
        mock_make_request.assert_not_called()
        assert len(client.test_logged_trace_ids) == 0

# --- Tests for RagMetricsClient.monitor and global monitor --- 

@patch('ragmetrics.api.find_integration') 
def test_monitor_dispatches_correctly(mock_find, ragmetrics_test_client): # Use ragmetrics_test_client
    client_to_monitor = MagicMock()
    integration_mock = MagicMock() # This is the mock for the sync_wrapper function
    mock_find.return_value = {
        'name': 'MockIntegration',
        'condition': lambda c: True,
        'methods_to_wrap': {'some_method': integration_mock}, # Simulate methods_to_wrap structure
        'async_methods_to_wrap': {}
    }
    
    client = ragmetrics_test_client
    result = client.monitor(client_to_monitor, metadata={"key": "value"}, callback=default_callback)
    
    # Changed to equality check, also check if integration_mock was called at all
    assert result == client_to_monitor 
    mock_find.assert_called_once_with(client_to_monitor)
    integration_mock.assert_called_once() # Simpler check: was the wrapper func called?

@patch('ragmetrics.api.find_integration', return_value=None) # Mock finder to return None
def test_monitor_no_integration_found(mock_find, ragmetrics_test_client): # Use ragmetrics_test_client
    client_to_monitor = MagicMock()
    client = ragmetrics_test_client
    result = client.monitor(client_to_monitor)
    # Changed to equality check
    assert result == client_to_monitor 
    mock_find.assert_called_once_with(client_to_monitor)

@patch('ragmetrics.api.ragmetrics_client.monitor') # Patch the method on the global instance
def test_global_monitor_calls_instance_monitor(mock_instance_monitor):
    client_to_monitor = MagicMock()
    metadata_arg = {"global": "monitor"}
    callback_arg = default_callback
    
    monitor(client_to_monitor, metadata=metadata_arg, callback=callback_arg) # Call the global monitor
    mock_instance_monitor.assert_called_once_with(client_to_monitor, metadata=metadata_arg, callback=callback_arg)

# Next: Tests for decorators, models, and individual client wrappers 