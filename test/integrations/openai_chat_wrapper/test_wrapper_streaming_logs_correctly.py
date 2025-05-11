# test/integrations/openai_chat_wrapper/test_wrapper_streaming_logs_correctly.py
import pytest
import os
from unittest.mock import patch, ANY, MagicMock 
from ragmetrics import RagMetricsClient
from ragmetrics.client_integrations.openai_chat_wrapper import wrap_openai_chat_completions_create
from ragmetrics.utils import default_callback 
# Fixtures auto-discovered

# Test wrapper behavior with streaming responses
def test_openai_wrapper_streaming_logs_correctly(ragmetrics_test_client: RagMetricsClient, mock_openai_completions_object):
    rm_client = ragmetrics_test_client
    if hasattr(rm_client, 'test_logged_trace_ids'): rm_client.trace_ids = []

    completions_obj, original_create_mock = mock_openai_completions_object

    # Simulate a streaming response from the original create method
    # The wrapper should collect this into a single response object or list of chunks
    # For simplicity here, let's assume the output_extractor or callback handles stream assembly.
    # The mock_openai_completions_object already returns "original_response" which is a simple string.
    # The current wrapper passes this simple string response to _log_trace.
    # If the actual OpenAI client returns an iterator for streams, the mock_openai_completions_object
    # and the test assertions would need to be more complex to simulate chunking and assembly.
    # For now, we rely on the existing mock behavior and assume "original_response" represents the assembled stream.

    def mock_log_trace_side_effect_streaming(*args, **kwargs):
        if hasattr(rm_client, 'test_logged_trace_ids'):
            rm_client.trace_ids.append("mock-trace-id-streaming-side-effect")
        # Potentially assert something about the nature of the 'response' kwarg if it's a stream placeholder
        return {"id": "mock-trace-id-streaming-side-effect"}

    with patch.object(rm_client, '_log_trace') as mock_log_trace_local:
        mock_log_trace_local.side_effect = mock_log_trace_side_effect_streaming

        wrap_openai_chat_completions_create(rm_client, completions_obj, default_callback)
        
        input_messages=[{"role": "user", "content": "Streaming Test"}]
        # Simulate a streaming call by setting stream=True
        # The actual mock_openai_completions_object doesn't change its response based on stream=True,
        # it still returns "original_response". The wrapper should still log this.
        response = completions_obj.create(
            model="gpt-stream-test", 
            messages=input_messages,
            stream=True 
        )

        assert response == "original_response" # Mock still returns this

        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if rm_client.logging_off:
                mock_log_trace_local.assert_not_called()
                assert len(rm_client.trace_ids) == 0
            else:
                mock_log_trace_local.assert_called_once()
                log_args, log_kwargs = mock_log_trace_local.call_args
                assert log_kwargs['input_messages'] == input_messages
                assert log_kwargs['response'] == "original_response" # Assembled or final response
                assert log_kwargs['metadata_llm']["stream"] is True # stream=True should be in metadata
                assert len(rm_client.trace_ids) == 1
        else: 
            mock_log_trace_local.assert_not_called()
            assert len(rm_client.trace_ids) == 0 