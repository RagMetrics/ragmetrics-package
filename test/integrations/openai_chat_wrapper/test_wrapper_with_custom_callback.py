# test/integrations/openai_chat_wrapper/test_wrapper_with_custom_callback.py
import pytest
import os
from unittest.mock import patch, ANY, MagicMock 
from ragmetrics import RagMetricsClient
from ragmetrics.client_integrations.openai_chat_wrapper import wrap_openai_chat_completions_create
# default_callback is not used here, custom_callback_mock is used instead
# Fixtures ragmetrics_test_client and mock_openai_completions_object are auto-discovered

# Test wrapper with a custom callback
def test_openai_wrapper_with_custom_callback(ragmetrics_test_client: RagMetricsClient, mock_openai_completions_object):
    rm_client = ragmetrics_test_client
    if hasattr(rm_client, 'test_logged_trace_ids'): rm_client.test_logged_trace_ids = []

    completions_obj, original_create_mock = mock_openai_completions_object

    # Mock for the custom callback function
    custom_callback_mock = MagicMock(return_value={"input": "custom_input", "output": "custom_output"})

    # Define a side effect function for the mock_log_trace
    def mock_log_trace_side_effect_custom_cb(*args, **kwargs):
        if hasattr(rm_client, 'test_logged_trace_ids'):
            rm_client.test_logged_trace_ids.append("mock-trace-id-custom-cb-side-effect")
        return {"id": "mock-trace-id-custom-cb-side-effect"}

    with patch.object(rm_client, '_log_trace') as mock_log_trace_local:
        mock_log_trace_local.side_effect = mock_log_trace_side_effect_custom_cb # Assign side effect

        wrap_openai_chat_completions_create(rm_client, completions_obj, custom_callback_mock)
        
        input_messages=[{"role": "user", "content": "Custom CB Test"}]
        completions_obj.create(
            model="gpt-custom-cb", 
            messages=input_messages
        )

        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if rm_client.logging_off:
                mock_log_trace_local.assert_not_called()
                custom_callback_mock.assert_not_called() # Callback shouldn't be called if logging is off
                assert len(rm_client.test_logged_trace_ids) == 0
            else:
                custom_callback_mock.assert_called_once_with(input_messages, "original_response")
                mock_log_trace_local.assert_called_once()
                log_args, log_kwargs = mock_log_trace_local.call_args
                # Check that the result from the custom_callback_mock was used in _log_trace
                assert log_kwargs['callback_result'] == {"input": "custom_input", "output": "custom_output"}
                assert len(rm_client.test_logged_trace_ids) == 1
        else: 
            mock_log_trace_local.assert_not_called()
            custom_callback_mock.assert_not_called() 
            assert len(rm_client.test_logged_trace_ids) == 0 