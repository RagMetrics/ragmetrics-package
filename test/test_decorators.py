# test/test_decorators.py
import pytest
import time
from unittest.mock import patch, MagicMock

from ragmetrics.decorators import trace_function_call
# Import the global client instance to patch its method
from ragmetrics.api import ragmetrics_client

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

@patch('ragmetrics.api.ragmetrics_client._log_trace') # Patch the method on the imported global instance
def test_trace_function_call_basic(mock_log_trace):
    # Assume client is logged in for decorator to work
    ragmetrics_client.access_token = "fake_token_for_test"
    ragmetrics_client.logging_off = False

    decorated_func = trace_function_call(simple_func)
    result = decorated_func(5, 3)

    assert result == 8 # Check original function logic
    mock_log_trace.assert_called_once()
    
    # Check arguments passed to _log_trace
    args, kwargs = mock_log_trace.call_args
    
    assert kwargs['input_messages'] == [{'role': 'user', 'content': '=simple_func(x=5, y=3)', 'tool_call': True}]
    assert kwargs['response'] == 8 # Raw result is passed as response
    assert kwargs['metadata_llm'] == {"function_name": "simple_func", "decorated_call": True}
    assert kwargs['contexts'] is None
    assert kwargs['expected'] is None
    assert isinstance(kwargs['duration'], float) and kwargs['duration'] > 0
    assert kwargs['tools'] is None
    assert kwargs['callback_result'] == {"input": "=simple_func(x=5, y=3)", "output": "8"}

    # Reset token for other tests
    ragmetrics_client.access_token = None 

@patch('ragmetrics.api.ragmetrics_client._log_trace')
def test_trace_function_call_with_kwargs(mock_log_trace):
    ragmetrics_client.access_token = "fake_token_for_test"
    ragmetrics_client.logging_off = False

    # Apply decorator directly
    @trace_function_call 
    def func_with_kwargs_decorated(a, b=10):
        return a * b

    result1 = func_with_kwargs_decorated(5) # Use default kwarg
    assert result1 == 50
    mock_log_trace.assert_called_once()
    args1, kwargs1 = mock_log_trace.call_args
    assert kwargs1['callback_result'] == {"input": "=func_with_kwargs_decorated(a=5)", "output": "50"}
    
    mock_log_trace.reset_mock()
    result2 = func_with_kwargs_decorated(a=3, b=20) # Override kwarg
    assert result2 == 60
    mock_log_trace.assert_called_once()
    args2, kwargs2 = mock_log_trace.call_args
    assert kwargs2['callback_result'] == {"input": "=func_with_kwargs_decorated(a=3, b=20)", "output": "60"}

    ragmetrics_client.access_token = None

@patch('ragmetrics.api.ragmetrics_client._log_trace')
def test_trace_function_call_instance_method(mock_log_trace):
    ragmetrics_client.access_token = "fake_token_for_test"
    ragmetrics_client.logging_off = False

    instance = MyClass(factor=3)
    # Decorate the bound method
    decorated_method = trace_function_call(instance.instance_method)
    
    result = decorated_method(7)
    assert result == 21
    mock_log_trace.assert_called_once()
    args, kwargs = mock_log_trace.call_args
    # Note: Decorating bound instance methods might not capture 'self' cleanly in input string depending on inspect capabilities
    # Here, it likely shows only the non-self arguments.
    assert kwargs['callback_result'] == {"input": "=instance_method(value=7)", "output": "21"} 
    assert kwargs['metadata_llm'] == {"function_name": "instance_method", "decorated_call": True}

    ragmetrics_client.access_token = None

@patch('ragmetrics.api.ragmetrics_client._log_trace')
def test_trace_function_call_when_logged_off(mock_log_trace):
    # Ensure logging off prevents call
    ragmetrics_client.access_token = "fake_token_for_test" 
    ragmetrics_client.logging_off = True # Turn logging off

    decorated_func = trace_function_call(simple_func)
    result = decorated_func(1, 1)

    assert result == 2
    mock_log_trace.assert_not_called()

    # Reset for other tests
    ragmetrics_client.logging_off = False
    ragmetrics_client.access_token = None

@patch('ragmetrics.api.ragmetrics_client._log_trace')
def test_trace_function_call_when_not_logged_in(mock_log_trace):
    # Ensure missing token prevents call
    ragmetrics_client.access_token = None # Ensure not logged in
    ragmetrics_client.logging_off = False

    decorated_func = trace_function_call(simple_func)
    result = decorated_func(2, 2)

    assert result == 4
    mock_log_trace.assert_not_called() 