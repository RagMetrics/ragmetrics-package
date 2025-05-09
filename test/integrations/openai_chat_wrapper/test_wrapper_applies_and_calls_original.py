# test/integrations/openai_chat_wrapper/test_wrapper_applies_and_calls_original.py
import pytest
import os
from unittest.mock import patch, ANY, MagicMock # Added MagicMock for fixture type hinting if needed by linters
from ragmetrics import RagMetricsClient 
from ragmetrics.client_integrations.openai_chat_wrapper import wrap_openai_chat_completions_create
from ragmetrics.utils import default_callback
# Fixtures ragmetrics_test_client (from test/conftest.py) and 
# mock_openai_completions_object (from test/integrations/conftest.py) are auto-discovered.

def test_openai_wrapper_applies_and_calls_original(ragmetrics_test_client: RagMetricsClient, mock_openai_completions_object):
    rm_client = ragmetrics_test_client
    completions_obj, original_create_mock = mock_openai_completions_object

    if hasattr(rm_client, 'test_logged_trace_ids'):
        rm_client.test_logged_trace_ids = []

    # Define a side effect function for the mock_log_trace
    def mock_log_trace_side_effect_basic(*args, **kwargs):
        # Simulate the real _log_trace appending an ID
        if hasattr(rm_client, 'test_logged_trace_ids'):
            rm_client.test_logged_trace_ids.append("mock-trace-id-basic-side-effect")
        # Simulate returning a response like the real _log_trace might
        return {"id": "mock-trace-id-basic-side-effect"}

    with patch.object(rm_client, '_log_trace') as mock_log_trace_local:
        mock_log_trace_local.side_effect = mock_log_trace_side_effect_basic # Assign the side effect
        
        wrapped = wrap_openai_chat_completions_create(
            rm_client,
            completions_obj,
            default_callback
        )

        assert wrapped is True
        assert hasattr(completions_obj, 'create')
        assert completions_obj.create.__func__ is not original_create_mock.__func__ if hasattr(original_create_mock, '__func__') else completions_obj.create is not original_create_mock

        input_messages = [{"role": "user", "content": "Hi OpenAI"}]
        response = completions_obj.create(
            model="gpt-test",
            messages=input_messages,
            metadata={"call_meta": "data"}, 
            contexts=["context1"],          
            temperature=0.5
        )

        assert response == "original_response"

        original_create_mock.assert_called_once()
        call_args, call_kwargs = original_create_mock.call_args
        assert call_kwargs['model'] == "gpt-test"
        assert call_kwargs['messages'] == input_messages
        assert call_kwargs['temperature'] == 0.5
        assert 'metadata' not in call_kwargs 
        assert 'contexts' not in call_kwargs 

        test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"

        if not test_mock:
            if rm_client.logging_off: 
                mock_log_trace_local.assert_not_called()
                assert len(rm_client.test_logged_trace_ids) == 0, "Trace IDs should be empty if logging is off"
            else: 
                mock_log_trace_local.assert_called_once()
                log_args_tuple, log_kwargs_dict = mock_log_trace_local.call_args

                assert log_kwargs_dict['input_messages'] == input_messages
                assert log_kwargs_dict['response'] == "original_response"
                assert log_kwargs_dict['metadata_llm']["call_meta"] == "data"
                assert log_kwargs_dict['metadata_llm']["temperature"] == 0.5 
                assert log_kwargs_dict['model_name'] == "gpt-test"
                assert log_kwargs_dict['contexts'] == ["context1"]
                assert log_kwargs_dict['expected'] is None
                assert isinstance(log_kwargs_dict['duration'], float) and log_kwargs_dict['duration'] >= 0
                assert log_kwargs_dict['tools'] is None
                expected_callback_res = default_callback(input_messages, "original_response")
                assert log_kwargs_dict['callback_result'] == expected_callback_res
                assert len(rm_client.test_logged_trace_ids) == 1, "One trace ID should be logged"
        else: 
            mock_log_trace_local.assert_not_called()
            assert len(rm_client.test_logged_trace_ids) == 0, "No trace IDs logged when TEST_MOCK=true" 