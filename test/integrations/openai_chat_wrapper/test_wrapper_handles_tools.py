# test/integrations/openai_chat_wrapper/test_wrapper_handles_tools.py
import pytest
import os
from unittest.mock import patch, ANY, MagicMock # Added MagicMock
from ragmetrics import RagMetricsClient
from ragmetrics.client_integrations.openai_chat_wrapper import wrap_openai_chat_completions_create
from ragmetrics.utils import default_callback
# Fixtures auto-discovered

def test_openai_wrapper_handles_tools(ragmetrics_test_client: RagMetricsClient, mock_openai_completions_object):
    rm_client = ragmetrics_test_client
    if hasattr(rm_client, 'test_logged_trace_ids'): rm_client.test_logged_trace_ids = []
    original_client_meta = rm_client.metadata 
    rm_client.metadata = None 

    completions_obj, original_create_mock = mock_openai_completions_object

    # Define a side effect function for the mock_log_trace
    def mock_log_trace_side_effect_tools(*args, **kwargs):
        if hasattr(rm_client, 'test_logged_trace_ids'):
            rm_client.test_logged_trace_ids.append("mock-trace-id-tools-side-effect")
        return {"id": "mock-trace-id-tools-side-effect"}

    with patch.object(rm_client, '_log_trace') as mock_log_trace_local:
        mock_log_trace_local.side_effect = mock_log_trace_side_effect_tools # Assign side effect

        wrap_openai_chat_completions_create(rm_client, completions_obj, default_callback)
        
        tools_def = [{"type": "function", "function": {"name": "test_func"}}]
        input_messages = [{"role": "user", "content": "Use tool"}]
        completions_obj.create(
            model="gpt-test", 
            messages=input_messages,
            tools=tools_def
        )

        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"

        if not test_mock:
            if rm_client.logging_off:
                mock_log_trace_local.assert_not_called()
                assert len(rm_client.test_logged_trace_ids) == 0
            else:
                mock_log_trace_local.assert_called_once()
                log_args, log_kwargs = mock_log_trace_local.call_args
                assert log_kwargs['tools'] == tools_def
                assert log_kwargs['model_name'] == "gpt-test"
                
                expected_metadata = {
                    "model": "gpt-test",
                    "tools": tools_def
                }
                assert log_kwargs['metadata_llm'] == expected_metadata
                
                expected_callback_res = default_callback(input_messages, "original_response")
                assert log_kwargs['callback_result'] == expected_callback_res 
                assert len(rm_client.test_logged_trace_ids) == 1
        else:
            mock_log_trace_local.assert_not_called()
            assert len(rm_client.test_logged_trace_ids) == 0
    
    rm_client.metadata = original_client_meta 