# test/integrations/openai_chat_wrapper/test_wrapper_handles_error_in_original_call.py
import pytest
import os
from unittest.mock import patch, ANY, MagicMock 
from ragmetrics import RagMetricsClient
from ragmetrics.client_integrations.openai_chat_wrapper import wrap_openai_chat_completions_create
from ragmetrics.utils import default_callback # default_callback might be used if error handling still calls it
# Fixtures auto-discovered

# Test wrapper behavior when the original OpenAI call raises an error
def test_openai_wrapper_handles_error_in_original_call(ragmetrics_test_client: RagMetricsClient, mock_openai_completions_object):
    rm_client = ragmetrics_test_client
    if hasattr(rm_client, 'test_logged_trace_ids'): rm_client.trace_ids = []

    completions_obj, original_create_mock = mock_openai_completions_object
    
    # Configure the original create method to raise an error
    simulated_error = Exception("Simulated OpenAI API Error")
    original_create_mock.side_effect = simulated_error

    # Define a side effect function for the mock_log_trace
    def mock_log_trace_side_effect_error(*args, **kwargs):
        if hasattr(rm_client, 'test_logged_trace_ids'):
            rm_client.trace_ids.append("mock-trace-id-error-side-effect")
        # Check that the error is passed to _log_trace
        assert kwargs.get('error') is simulated_error
        return {"id": "mock-trace-id-error-side-effect"}

    with patch.object(rm_client, '_log_trace') as mock_log_trace_local:
        mock_log_trace_local.side_effect = mock_log_trace_side_effect_error # Assign side effect

        wrap_openai_chat_completions_create(rm_client, completions_obj, default_callback)
        
        input_messages=[{"role": "user", "content": "Error Test"}]
        # Expect the wrapper to re-raise the error from the original call
        with pytest.raises(Exception, match="Simulated OpenAI API Error"):
            completions_obj.create(
                model="gpt-error-test", 
                messages=input_messages
            )

        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if rm_client.logging_off:
                mock_log_trace_local.assert_not_called()
                assert len(rm_client.trace_ids) == 0
            else:
                mock_log_trace_local.assert_called_once()
                log_args, log_kwargs = mock_log_trace_local.call_args
                assert log_kwargs['input_messages'] == input_messages
                assert log_kwargs['error'] is simulated_error # Verify error was logged
                assert len(rm_client.trace_ids) == 1
        else: 
            mock_log_trace_local.assert_not_called()
            assert len(rm_client.trace_ids) == 0 