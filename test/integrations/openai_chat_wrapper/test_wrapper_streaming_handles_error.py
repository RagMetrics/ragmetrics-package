# test/integrations/openai_chat_wrapper/test_wrapper_streaming_handles_error.py
import pytest
import os
from unittest.mock import patch, ANY, MagicMock 
from ragmetrics import RagMetricsClient
from ragmetrics.client_integrations.openai_chat_wrapper import wrap_openai_chat_completions_create
from ragmetrics.utils import default_callback 
# Fixtures auto-discovered

# Test wrapper behavior with streaming responses when an error occurs
def test_openai_wrapper_streaming_handles_error(ragmetrics_test_client: RagMetricsClient, mock_openai_completions_object):
    rm_client = ragmetrics_test_client
    if hasattr(rm_client, 'test_logged_trace_ids'): rm_client.test_logged_trace_ids = []

    completions_obj, original_create_mock = mock_openai_completions_object

    # Configure the original create method (which mockingly handles stream) to raise an error
    simulated_error = Exception("Simulated OpenAI Streaming Error")
    original_create_mock.side_effect = simulated_error

    def mock_log_trace_side_effect_stream_error(*args, **kwargs):
        if hasattr(rm_client, 'test_logged_trace_ids'):
            rm_client.test_logged_trace_ids.append("mock-trace-id-stream-error-side-effect")
        assert kwargs.get('error') is simulated_error # Check error is passed
        return {"id": "mock-trace-id-stream-error-side-effect"}

    with patch.object(rm_client, '_log_trace') as mock_log_trace_local:
        mock_log_trace_local.side_effect = mock_log_trace_side_effect_stream_error

        wrap_openai_chat_completions_create(rm_client, completions_obj, default_callback)
        
        input_messages=[{"role": "user", "content": "Streaming Error Test"}]
        with pytest.raises(Exception, match="Simulated OpenAI Streaming Error"):
            completions_obj.create(
                model="gpt-stream-error-test", 
                messages=input_messages,
                stream=True 
            )

        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if rm_client.logging_off:
                mock_log_trace_local.assert_not_called()
                assert len(rm_client.test_logged_trace_ids) == 0
            else:
                mock_log_trace_local.assert_called_once()
                log_args, log_kwargs = mock_log_trace_local.call_args
                assert log_kwargs['input_messages'] == input_messages
                assert log_kwargs['error'] is simulated_error
                assert log_kwargs['metadata_llm']["stream"] is True # stream=True should still be in metadata
                assert len(rm_client.test_logged_trace_ids) == 1
        else: 
            mock_log_trace_local.assert_not_called()
            assert len(rm_client.test_logged_trace_ids) == 0 