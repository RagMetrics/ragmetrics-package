# test/test_decorators.py
import pytest
import time
import os # Import os
from unittest.mock import patch, MagicMock
import uuid # Import uuid for the side effect

from ragmetrics.decorators import trace_function_call
# No longer need to import ragmetrics_client directly for patching if using patch.object

# --- Test Function Definitions --- 

def simple_func(x, y):
    """A simple function to trace."""
    time.sleep(0.01) # Simulate work
    return x + y

def func_with_kwargs(a, b=10):
    """A function with keyword arguments."""
    return a * b

class MyClass:
    def __init__(self, factor):
        self.factor = factor
    
    def instance_method(self, value):
        """An instance method to trace."""
        return value * self.factor

# --- Tests --- 

def test_trace_function_call_basic(ragmetrics_test_client): # Add fixture
    client = ragmetrics_test_client
    if hasattr(client, 'test_logged_trace_ids'):
        client.test_logged_trace_ids = []

    # Mock side effect to simulate ID append
    def mock_log_trace_side_effect(*args, **kwargs):
        if hasattr(client, 'test_logged_trace_ids'):
            client.test_logged_trace_ids.append(str(uuid.uuid4())) # Append a unique ID
        return None

    with patch.object(client, '_log_trace', side_effect=mock_log_trace_side_effect) as mock_log_trace:
        decorated_func = trace_function_call(simple_func)
        result = decorated_func(5, 3)
        assert result == 8 
        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if client.logging_off:
                mock_log_trace.assert_not_called()
                assert len(client.test_logged_trace_ids) == 0
            else:
                mock_log_trace.assert_called_once() 
                args, kwargs = mock_log_trace.call_args
                assert kwargs['input_messages'] == [{'role': 'user', 'content': '=simple_func(x=5, y=3)', 'tool_call': True}]
                assert kwargs['response'] == 8 
                assert kwargs['metadata_llm'] == {"function_name": "simple_func", "decorated_call": True}
                assert kwargs['contexts'] is None
                assert kwargs['expected'] is None
                assert isinstance(kwargs['duration'], float) and kwargs['duration'] > 0
                assert kwargs['tools'] is None
                assert kwargs['callback_result'] == {"input": "=simple_func(x=5, y=3)", "output": "8"}
                assert len(client.test_logged_trace_ids) == 1 # Check that side_effect appended one ID
        else: # test_mock is True
            mock_log_trace.assert_not_called()
            assert len(client.test_logged_trace_ids) == 0

@patch('ragmetrics.api.ragmetrics_client._log_trace') # Keep global patch for structure, but use local patch for assertion
def test_trace_function_call_with_kwargs(mock_global_log_trace, ragmetrics_test_client): # Add fixture
    client = ragmetrics_test_client
    if hasattr(client, 'test_logged_trace_ids'): client.test_logged_trace_ids = []

    @trace_function_call 
    def func_with_kwargs_decorated(a, b=10):
        return a * b

    # Mock side effect
    def mock_log_trace_side_effect(*args, **kwargs):
        if hasattr(client, 'test_logged_trace_ids'):
            client.test_logged_trace_ids.append(str(uuid.uuid4()))
        return None

    with patch.object(client, '_log_trace', side_effect=mock_log_trace_side_effect) as mock_log_trace_local:
        result1 = func_with_kwargs_decorated(5)
        assert result1 == 50
        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if client.logging_off:
                mock_log_trace_local.assert_not_called()
                assert len(client.test_logged_trace_ids) == 0
            else:
                mock_log_trace_local.assert_called_once()
                assert len(client.test_logged_trace_ids) == 1
                client.test_logged_trace_ids = [] # Clear for next call
        else: # test_mock is True
            mock_log_trace_local.assert_not_called()
            assert len(client.test_logged_trace_ids) == 0
        
        mock_log_trace_local.reset_mock() # Reset for the next call

        result2 = func_with_kwargs_decorated(a=3, b=20)
        assert result2 == 60
        if not test_mock:
            if client.logging_off:
                mock_log_trace_local.assert_not_called()
                assert len(client.test_logged_trace_ids) == 0
            else:
                mock_log_trace_local.assert_called_once()
                assert len(client.test_logged_trace_ids) == 1
        else: # test_mock is True
            mock_log_trace_local.assert_not_called()
            assert len(client.test_logged_trace_ids) == 0

@patch('ragmetrics.api.ragmetrics_client._log_trace') # Keep global patch
def test_trace_function_call_instance_method(mock_global_log_trace, ragmetrics_test_client): # Add fixture
    client = ragmetrics_test_client
    if hasattr(client, 'test_logged_trace_ids'): client.test_logged_trace_ids = []

    instance = MyClass(factor=3)
    
    # Mock side effect
    def mock_log_trace_side_effect(*args, **kwargs):
        if hasattr(client, 'test_logged_trace_ids'):
            client.test_logged_trace_ids.append(str(uuid.uuid4()))
        return None

    with patch.object(client, '_log_trace', side_effect=mock_log_trace_side_effect) as mock_log_trace_local:
        decorated_method = trace_function_call(instance.instance_method)
        result = decorated_method(7)
        assert result == 21
        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if client.logging_off:
                mock_log_trace_local.assert_not_called()
                assert len(client.test_logged_trace_ids) == 0
            else:
                mock_log_trace_local.assert_called_once()
                assert len(client.test_logged_trace_ids) == 1
        else: # test_mock is True
            mock_log_trace_local.assert_not_called()
            assert len(client.test_logged_trace_ids) == 0

# This test becomes more about the fixture's behavior when TEST_MOCK=True
def test_trace_function_call_when_logged_off(ragmetrics_test_client):
    client = ragmetrics_test_client
    test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"

    if hasattr(client, 'test_logged_trace_ids'): client.test_logged_trace_ids = []

    with patch.object(client, '_log_trace') as mock_log_trace_local:
        original_logging_off = client.logging_off
        if not test_mock:
            # If not mocking, fixture sets logging_off=False. So, turn it on for this test.
            client.logging_off = True
        
        assert client.logging_off is True # Key condition for this test

        decorated_func = trace_function_call(simple_func)
        result = decorated_func(1, 1)

        assert result == 2
        mock_log_trace_local.assert_not_called()
        assert len(client.test_logged_trace_ids) == 0

        if not test_mock:
            client.logging_off = original_logging_off # Restore state for non-mocked runs

# This test is tricky because it tests the state of "not logged in"
# The fixture will try to log in or set a dummy token if TEST_MOCK=False
# If TEST_MOCK=True, logging is off anyway.
def test_trace_function_call_when_not_logged_in(ragmetrics_test_client, monkeypatch): # Add fixture
    client = ragmetrics_test_client
    test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"

    if hasattr(client, 'test_logged_trace_ids'): client.test_logged_trace_ids = []

    with patch.object(client, '_log_trace') as mock_log_trace_local:
        # For this test to be meaningful, we need: 
        # 1. client.access_token to be None (or client.logged_in = False)
        # 2. client.logging_off to be False (so that the auth check is actually hit)

        original_token = client.access_token
        # original_logged_in_state = client.logged_in # This attribute does not exist
        original_logging_off_state = client.logging_off

        # Simulate not logged in, but logging is ON
        client.access_token = None
        # client.logged_in = False # This attribute does not exist
        client.logging_off = False # Crucial for this test scenario

        decorated_func = trace_function_call(simple_func)
        result = decorated_func(2, 2)

        assert result == 4
        # Regardless of test_mock, if access_token is None and logging_off is False, _log_trace should not be called by decorator
        mock_log_trace_local.assert_not_called()
        assert len(client.test_logged_trace_ids) == 0

        # Restore client state to what fixture might have set, to avoid side effects
        client.access_token = original_token
        # client.logged_in = original_logged_in_state # This attribute does not exist
        client.logging_off = original_logging_off_state 