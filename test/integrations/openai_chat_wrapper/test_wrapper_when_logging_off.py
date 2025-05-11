# test/integrations/openai_chat_wrapper/test_wrapper_when_logging_off.py
import pytest
import os
from unittest.mock import patch, ANY, MagicMock 
from ragmetrics import RagMetricsClient
from ragmetrics.client_integrations.openai_chat_wrapper import wrap_openai_chat_completions_create
from ragmetrics.utils import default_callback
# Fixtures auto-discovered

# Test wrapper behavior when client.logging_off is True
def test_openai_wrapper_when_logging_off(ragmetrics_test_client: RagMetricsClient, mock_openai_completions_object):
    rm_client = ragmetrics_test_client
    
    # Ensure logging is explicitly OFF for this test, overriding fixture's potential state
    original_logging_off_state = rm_client.logging_off
    rm_client.logging_off = True
    if hasattr(rm_client, 'test_logged_trace_ids'): rm_client.trace_ids = []

    completions_obj, original_create_mock = mock_openai_completions_object

    # Patch _make_request on this client to ensure the real _log_trace is hit
    # and its internal guard for logging_off prevents calling _make_request.
    with patch.object(rm_client, '_make_request') as mock_make_request_on_logged_off_client: 
        wrap_openai_chat_completions_create(rm_client, completions_obj, default_callback)
        
        completions_obj.create(
            model="gpt-test-log-off", 
            messages=[{"role": "user", "content": "Logging off test"}]
        )

        # The real _log_trace should be called, but it should not call _make_request
        mock_make_request_on_logged_off_client.assert_not_called()
        assert len(rm_client.trace_ids) == 0 # No IDs should be logged

    # Restore original logging_off state from fixture to avoid affecting other tests
    rm_client.logging_off = original_logging_off_state 