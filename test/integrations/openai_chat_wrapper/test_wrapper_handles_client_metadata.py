# test/integrations/openai_chat_wrapper/test_wrapper_handles_client_metadata.py
import pytest
import os
from unittest.mock import patch, ANY, MagicMock # Added MagicMock
from ragmetrics import RagMetricsClient
from ragmetrics.client_integrations.openai_chat_wrapper import wrap_openai_chat_completions_create
from ragmetrics.utils import default_callback
# Fixtures auto-discovered

def test_openai_wrapper_handles_client_metadata(ragmetrics_test_client: RagMetricsClient, mock_openai_completions_object):
    rm_client = ragmetrics_test_client
    original_client_meta = rm_client.metadata
    rm_client.metadata = {"client_meta": "global"} 
    if hasattr(rm_client, 'test_logged_trace_ids'): rm_client.trace_ids = []

    completions_obj, original_create_mock = mock_openai_completions_object

    # Define a side effect function for the mock_log_trace
    def mock_log_trace_side_effect_metadata(*args, **kwargs):
        if hasattr(rm_client, 'test_logged_trace_ids'):
            rm_client.trace_ids.append("mock-trace-id-metadata-side-effect")
        return {"id": "mock-trace-id-metadata-side-effect"}

    with patch.object(rm_client, '_log_trace') as mock_log_trace_local:
        mock_log_trace_local.side_effect = mock_log_trace_side_effect_metadata # Assign side effect

        wrap_openai_chat_completions_create(rm_client, completions_obj, default_callback)
        
        completions_obj.create(
            model="gpt-test", 
            messages=[{"role": "user", "content": "Test"}],
            metadata={"call_meta": "local"} 
        )

        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
        if not test_mock:
            if rm_client.logging_off:
                mock_log_trace_local.assert_not_called()
                assert len(rm_client.trace_ids) == 0, "Trace IDs should be empty if logging is off"
            else:
                mock_log_trace_local.assert_called_once()
                log_args, log_kwargs = mock_log_trace_local.call_args
                
                expected_metadata = {
                    "client_meta": "global", 
                    "model": "gpt-test",    
                    "call_meta": "local"      
                }
                assert log_kwargs['metadata_llm'] == expected_metadata
                assert log_kwargs['model_name'] == "gpt-test"
                assert len(rm_client.trace_ids) == 1, "One trace ID should be logged"
        else: 
            mock_log_trace_local.assert_not_called()
            assert len(rm_client.trace_ids) == 0, "Trace IDs should be empty when mocking"
    
    rm_client.metadata = original_client_meta 