# test/integrations/openai_chat_wrapper/test_wrapper_logs_trace_by_default.py
import pytest
import os
from unittest.mock import patch, ANY, MagicMock 
from ragmetrics import RagMetricsClient
from ragmetrics.client_integrations.openai_chat_wrapper import wrap_openai_chat_completions_create
from ragmetrics.utils import default_callback
# Fixtures auto-discovered

# Test default behavior: logging on, no specific client metadata
def test_openai_wrapper_logs_trace_by_default(ragmetrics_test_client: RagMetricsClient, mock_openai_completions_object):
    rm_client = ragmetrics_test_client
    # Ensure client metadata is None for this test of defaults
    original_client_meta = rm_client.metadata
    rm_client.metadata = None
    if hasattr(rm_client, 'test_logged_trace_ids'): rm_client.trace_ids = []

    completions_obj, original_create_mock = mock_openai_completions_object

    def mock_log_trace_side_effect_default(*args, **kwargs):
        if hasattr(rm_client, 'test_logged_trace_ids'):
            rm_client.trace_ids.append("mock-trace-id-default-side-effect")
        return {"id": "mock-trace-id-default-side-effect"}

    with patch.object(rm_client, '_log_trace') as mock_log_trace_local:
        mock_log_trace_local.side_effect = mock_log_trace_side_effect_default

        wrap_openai_chat_completions_create(rm_client, completions_obj, default_callback)
        
        input_messages = [{"role": "user", "content": "Default Test"}]
        completions_obj.create(
            model="gpt-default-test", 
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
                assert log_kwargs['response'] == "original_response"
                # No client metadata, no call metadata, only model is dynamic
                assert log_kwargs['metadata_llm'] == {"model": "gpt-default-test"} 
                assert log_kwargs['model_name'] == "gpt-default-test"
                assert len(rm_client.trace_ids) == 1
        else: 
            mock_log_trace_local.assert_not_called()
            assert len(rm_client.trace_ids) == 0
    
    rm_client.metadata = original_client_meta # Restore 