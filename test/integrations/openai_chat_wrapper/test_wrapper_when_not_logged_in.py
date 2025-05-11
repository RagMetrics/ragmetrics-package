# test/integrations/openai_chat_wrapper/test_wrapper_when_not_logged_in.py
import pytest
import os
from unittest.mock import patch, ANY, MagicMock 
from ragmetrics import RagMetricsClient # Import client directly for instantiation
from ragmetrics.client_integrations.openai_chat_wrapper import wrap_openai_chat_completions_create
from ragmetrics.utils import default_callback
# mock_openai_completions_object fixture is auto-discovered

# Test wrapper behavior when client is not logged in (no access token)
def test_openai_wrapper_when_not_logged_in(mock_openai_completions_object, monkeypatch):
    # Create a fresh client instance that is explicitly not logged in
    # Isolate from RAGMETRICS_API_KEY env var for this test
    monkeypatch.delenv("RAGMETRICS_API_KEY", raising=False)
    rm_client_not_logged_in = RagMetricsClient() 
    rm_client_not_logged_in.access_token = None # Explicitly ensure no token
    rm_client_not_logged_in.logging_off = False # Ensure logging is ON for the check to matter

    if hasattr(rm_client_not_logged_in, 'test_logged_trace_ids'): 
        rm_client_not_logged_in.trace_ids = []

    completions_obj, original_create_mock = mock_openai_completions_object

    # Patch _make_request on this client to ensure the real _log_trace is hit
    # and its internal guard for no access_token prevents calling _make_request.
    with patch.object(rm_client_not_logged_in, '_make_request') as mock_make_request_on_unauth_client: 
        wrap_openai_chat_completions_create(rm_client_not_logged_in, completions_obj, default_callback)
        
        completions_obj.create(
            model="gpt-test-not-logged-in", 
            messages=[{"role": "user", "content": "Not logged in test"}]
        )

        # The real _log_trace should be called by the wrapper, but it should not call _make_request
        mock_make_request_on_unauth_client.assert_not_called()
        # test_logged_trace_ids would also be empty as _make_request (which returns the ID) isn't called.
        assert len(rm_client_not_logged_in.trace_ids) == 0 