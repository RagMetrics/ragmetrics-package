# test/integrations/test_langchain_wrapper.py
import pytest
from unittest.mock import patch, MagicMock, ANY
import time
import types
import os # Import os
from typing import Tuple, Dict, Any
import logging # Import logging

# Modules to test
from ragmetrics.api import RagMetricsClient
from ragmetrics.client_integrations.langchain_wrapper import wrap_langchain_invoke
from ragmetrics.utils import default_callback

# --- Fixtures --- 

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

def test_langchain_wrapper_instance(ragmetrics_test_client, mock_langchain_client_instance):
    rm_client = ragmetrics_test_client
    # Isolate client metadata for this test
    original_client_meta = rm_client.metadata
    rm_client.metadata = None # Ensure no global metadata affects this test

    lc_instance = mock_langchain_client_instance
    original_invoke_mock = lc_instance.invoke

    if hasattr(rm_client, 'test_logged_trace_ids'): rm_client.test_logged_trace_ids = []

    def mock_log_trace_side_effect_instance(*args, **kwargs):
        if hasattr(rm_client, 'test_logged_trace_ids'):
            rm_client.test_logged_trace_ids.append("mock-lc-instance-trace-id")
        return {"id": "mock-lc-instance-trace-id"}

    with patch.object(rm_client, '_log_trace') as mock_log_trace_local:
        mock_log_trace_local.side_effect = mock_log_trace_side_effect_instance
        wrapped = wrap_langchain_invoke(
            rm_client, 
            lc_instance, 
            default_callback
        )
        assert wrapped is True
        assert hasattr(lc_instance, 'invoke')
        assert isinstance(lc_instance.invoke, types.MethodType)
        # Ensure the original method wasn't called *during* wrapping
        original_invoke_mock.assert_not_called()

        input_data = {"input_key": "input_value"}
        response = lc_instance.invoke(
            input_data, 
            config={"configurable": {"session_id": "123"}},
            metadata={"call_meta": "lc_data"},
            contexts=["lc_context"]
        )
        
        assert response == "original_lc_response"
        # Now the original method should have been called by the wrapper
        original_invoke_mock.assert_called_once_with(input_data, config={"configurable": {"session_id": "123"}})

        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            mock_log_trace_local.assert_called_once()
            log_args, log_kwargs = mock_log_trace_local.call_args

            assert log_kwargs['input_messages'] == input_data
            assert log_kwargs['response'] == "original_lc_response"
            
            expected_metadata_llm = {
                "call_meta": "lc_data",
                "langchain_config": {"configurable": {"session_id": "123"}}
            }
            assert log_kwargs['metadata_llm'] == expected_metadata_llm
            
            assert log_kwargs['contexts'] == ["lc_context"]
            assert log_kwargs['expected'] is None
            assert isinstance(log_kwargs['duration'], float) and log_kwargs['duration'] >= 0
            assert log_kwargs['tools'] is None
            assert log_kwargs['callback_result'] == default_callback(input_data, "original_lc_response")
            assert 'config' in log_kwargs
            assert log_kwargs['config'] == {"configurable": {"session_id": "123"}}
            assert len(rm_client.test_logged_trace_ids) == 1
        else:
            mock_log_trace_local.assert_not_called()
            assert len(rm_client.test_logged_trace_ids) == 0

    # Restore original client metadata
    rm_client.metadata = original_client_meta

def test_langchain_wrapper_class(ragmetrics_test_client, mock_langchain_client_class):
    rm_client = ragmetrics_test_client
    # Isolate client metadata for this test
    original_client_meta = rm_client.metadata
    rm_client.metadata = None # Ensure no global metadata affects this test

    LcClass = mock_langchain_client_class
    original_invoke_mock = LcClass.invoke 
    if hasattr(rm_client, 'test_logged_trace_ids'): rm_client.test_logged_trace_ids = []

    def mock_log_trace_side_effect_class(*args, **kwargs):
        if hasattr(rm_client, 'test_logged_trace_ids'):
            rm_client.test_logged_trace_ids.append("mock-lc-class-trace-id")
        return {"id": "mock-lc-class-trace-id"}

    with patch.object(rm_client, '_log_trace') as mock_log_trace_local:
        mock_log_trace_local.side_effect = mock_log_trace_side_effect_class
        # Apply the wrapper to the class
        wrapped = wrap_langchain_invoke(
            rm_client, 
            LcClass, 
            default_callback
        )
        assert wrapped is True
        assert hasattr(LcClass, 'invoke')
        assert LcClass.invoke is not original_invoke_mock 
        assert callable(LcClass.invoke)

        # Ensure original mock wasn't called during wrapping
        original_invoke_mock.assert_not_called()

        input_data_class = "class_input"
        response = LcClass.invoke(input_data_class, metadata={"class_call": True})
        
        assert response == "original_lc_class_response"
        original_invoke_mock.assert_called_once_with(input_data_class) # Original called by wrapper
        
        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            mock_log_trace_local.assert_called_once()
            log_args, log_kwargs = mock_log_trace_local.call_args
            assert log_kwargs['input_messages'] == input_data_class
            assert log_kwargs['response'] == "original_lc_class_response"
            assert log_kwargs['metadata_llm'] == {"class_call": True}
            assert len(rm_client.test_logged_trace_ids) == 1
        else:
            mock_log_trace_local.assert_not_called()
            assert len(rm_client.test_logged_trace_ids) == 0

    # Restore original client metadata
    rm_client.metadata = original_client_meta

def test_langchain_wrapper_handles_client_metadata(ragmetrics_test_client, mock_langchain_client_instance):
    rm_client = ragmetrics_test_client
    original_client_meta = rm_client.metadata
    rm_client.metadata = {"client_meta": "global_lc"} 
    lc_instance = mock_langchain_client_instance
    if hasattr(rm_client, 'test_logged_trace_ids'): rm_client.test_logged_trace_ids = []

    def mock_log_trace_side_effect_metadata(*args, **kwargs):
        if hasattr(rm_client, 'test_logged_trace_ids'):
            rm_client.test_logged_trace_ids.append("mock-lc-metadata-trace-id")
        return {"id": "mock-lc-metadata-trace-id"}

    with patch.object(rm_client, '_log_trace') as mock_log_trace_local:
        mock_log_trace_local.side_effect = mock_log_trace_side_effect_metadata
        wrap_langchain_invoke(rm_client, lc_instance, default_callback)
        
        lc_instance.invoke("Test", metadata={"call_meta": "local_lc"})

        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            mock_log_trace_local.assert_called_once()
            log_args, log_kwargs = mock_log_trace_local.call_args
            assert log_kwargs['metadata_llm'] == {"client_meta": "global_lc", "call_meta": "local_lc"}
            assert len(rm_client.test_logged_trace_ids) == 1
        else:
            mock_log_trace_local.assert_not_called()
            assert len(rm_client.test_logged_trace_ids) == 0
    
    rm_client.metadata = original_client_meta # Restore

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