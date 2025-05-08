# test/integrations/test_langchain_wrapper.py
import pytest
from unittest.mock import patch, MagicMock, ANY
import time
import types
from typing import Tuple, Dict, Any
import logging # Import logging

# Modules to test
from ragmetrics.api import RagMetricsClient
from ragmetrics.client_integrations.langchain_wrapper import wrap_langchain_invoke
from ragmetrics.utils import default_callback

# --- Fixtures --- 

@pytest.fixture
def logged_in_rm_client():
    """Provides a logged-in RagMetricsClient instance with a mock _log_trace."""
    client = RagMetricsClient()
    client.access_token = "test-token"
    client.logging_off = False
    client._log_trace = MagicMock()
    return client

@pytest.fixture
def mock_langchain_client_instance():
    """Creates a mock LangChain client instance."""
    mock_instance = MagicMock()
    # Define the original invoke method on the instance
    mock_instance.invoke = MagicMock(return_value="original_lc_response")
    return mock_instance

@pytest.fixture
def mock_langchain_client_class():
    """Creates a mock LangChain client class."""
    class MockLangchainClass:
        # Define the original invoke method on the class
        invoke = MagicMock(return_value="original_lc_class_response")
    return MockLangchainClass

# --- Tests --- 

def test_langchain_wrapper_instance(logged_in_rm_client, mock_langchain_client_instance):
    # # Enable debug logging for wrapper_utils for this test
    # wrapper_utils_logger = logging.getLogger("ragmetrics.client_integrations.wrapper_utils")
    # original_level = wrapper_utils_logger.getEffectiveLevel()
    # wrapper_utils_logger.setLevel(logging.DEBUG)
    # # If you also want to see it in console, ensure a handler is configured for that logger
    # # For pytest -s, prints to stdout/stderr are usually captured if not handled by logging directly to console
    # # If still no output, might need to add a StreamHandler
    # # console_handler = logging.StreamHandler()
    # # console_handler.setLevel(logging.DEBUG)
    # # wrapper_utils_logger.addHandler(console_handler)

    rm_client = logged_in_rm_client
    lc_instance = mock_langchain_client_instance

    # Apply the wrapper to the instance
    wrapped = wrap_langchain_invoke(
        rm_client, 
        lc_instance, 
        default_callback
    )
    assert wrapped is True
    # Check that the instance's invoke is now a bound method (the wrapper)
    assert hasattr(lc_instance, 'invoke')
    assert isinstance(lc_instance.invoke, types.MethodType)

    # --- Call the wrapped method --- 
    input_data = {"input_key": "input_value"}
    response = lc_instance.invoke(
        input_data, 
        config={"configurable": {"session_id": "123"}}, # Example LangChain config
        metadata={"call_meta": "lc_data"},   # Ragmetrics metadata
        contexts=["lc_context"]             # Ragmetrics context
    )
    
    # --- Assertions --- 
    assert response == "original_lc_response"
    
    # Check _log_trace call
    rm_client._log_trace.assert_called_once()
    log_args, log_kwargs = rm_client._log_trace.call_args

    assert log_kwargs['input_messages'] == input_data # Wrapper extracts input
    assert log_kwargs['response'] == "original_lc_response"
    
    expected_metadata_llm = {
        "call_meta": "lc_data",
        "langchain_config": {"configurable": {"session_id": "123"}}
    }
    assert log_kwargs['metadata_llm'] == expected_metadata_llm
    
    assert log_kwargs['contexts'] == ["lc_context"]
    assert log_kwargs['expected'] is None
    assert isinstance(log_kwargs['duration'], float) and log_kwargs['duration'] >= 0
    assert log_kwargs['tools'] is None # Tools might be in config, test separately if needed
    assert log_kwargs['callback_result'] == default_callback(input_data, "original_lc_response")
    # Check that config was passed through in kwargs to _log_trace
    assert 'config' in log_kwargs
    assert log_kwargs['config'] == {"configurable": {"session_id": "123"}}

    # # Reset logger level after test
    # wrapper_utils_logger.setLevel(original_level)
    # # if 'console_handler' in locals(): # Clean up handler if added
    # #     wrapper_utils_logger.removeHandler(console_handler)

def test_langchain_wrapper_class(logged_in_rm_client, mock_langchain_client_class):
    rm_client = logged_in_rm_client
    LcClass = mock_langchain_client_class
    # Store the original mock *before* wrapping
    original_invoke_mock = LcClass.invoke 

    # Apply the wrapper to the class
    wrapped = wrap_langchain_invoke(
        rm_client, 
        LcClass, 
        default_callback
    )
    assert wrapped is True
    # Check that the class's invoke is now the wrapper function
    assert hasattr(LcClass, 'invoke')
    # Corrected assertion: Compare against the stored original mock object
    assert LcClass.invoke is not original_invoke_mock 
    assert callable(LcClass.invoke)

    # --- Call the wrapped method (as if it were static/class method for testing) --- 
    # Note: Real LangChain usage might involve instantiation first.
    # This test primarily verifies that the class attribute was replaced.
    input_data_class = "class_input"
    
    response = LcClass.invoke(input_data_class, metadata={"class_call": True})
    
    # --- Assertions --- 
    assert response == "original_lc_class_response"
    
    # Check original method call using the stored mock
    original_invoke_mock.assert_called_once_with(input_data_class) # No metadata
    
    # Check _log_trace call
    rm_client._log_trace.assert_called_once()
    log_args, log_kwargs = rm_client._log_trace.call_args
    assert log_kwargs['input_messages'] == input_data_class
    assert log_kwargs['response'] == "original_lc_class_response"
    assert log_kwargs['metadata_llm'] == {"class_call": True}

def test_langchain_wrapper_handles_client_metadata(logged_in_rm_client, mock_langchain_client_instance):
    rm_client = logged_in_rm_client
    rm_client.metadata = {"client_meta": "global_lc"} # Set client metadata
    lc_instance = mock_langchain_client_instance

    wrap_langchain_invoke(rm_client, lc_instance, default_callback)
    
    lc_instance.invoke("Test", metadata={"call_meta": "local_lc"})

    rm_client._log_trace.assert_called_once()
    log_args, log_kwargs = rm_client._log_trace.call_args
    # Metadata should be merged, call overrides client
    assert log_kwargs['metadata_llm'] == {"client_meta": "global_lc", "call_meta": "local_lc"}

def _langchain_input_extractor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """Extracts input for LangChain's invoke/ainvoke.
       Assumes input is typically the first positional argument.
       Handles instance methods where args[0] is self.
    """
    if len(args) > 1: # If called like instance.invoke(input, ...), args = (self, input, ...)
        return args[1]
    elif len(args) == 1 and not isinstance(args[0], type): # Called like class_instance.invoke(input) but wrapped on class?
        # Check if args[0] is likely the input (not self)
        # This case is ambiguous, maybe rely on kwargs.
        pass
    elif args:
        return args[0]
        
    # Fallback to kwargs if args don't provide clear input
    return kwargs.get("input", None) 