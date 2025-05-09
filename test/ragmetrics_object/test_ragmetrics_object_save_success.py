# test/ragmetrics_object/test_ragmetrics_object_save_success.py
import pytest
import os
from unittest.mock import patch
# Import necessary items from ragmetrics.api (client, exceptions)
from ragmetrics.api import ragmetrics_client, RagMetricsAPIError, RagMetricsAuthError 
# Explicitly import from parent conftest.py
from ..conftest import MockDataObject 
# Fixtures (mock_api_post_success) are automatically available from conftest.py

@pytest.mark.skipif(os.getenv("TEST_MOCK", "False").lower() == "true", reason="save requires API calls, skipped when TEST_MOCK is true")
@patch('ragmetrics.api.RagMetricsClient._make_request') # Patching the class method used internally
def test_ragmetricsobject_save_success(mock_make_request, mock_api_post_success, ragmetrics_test_client): # Add fixture
    # This test runs only if TEST_MOCK is false. 
    # The fixture ensures the client is logged in (or tries to).
    client = ragmetrics_test_client 
    # The mock_make_request here is the one patched on RagMetricsClient class for this test method's scope.
    # We need to ensure it behaves like the real _make_request, returning parsed JSON.
    # The mock_api_post_success fixture is a mock Response object.
    # The real _make_request would call .json() on this.
    mock_make_request.return_value = mock_api_post_success.json() # Corrected: return the JSON dict
    
    # MockDataObject is available from conftest.py
    obj = MockDataObject(name="test_obj", value=10) 
    assert obj.id is None
    
    # The save method uses ragmetrics_client (the global instance) internally.
    # The ragmetrics_test_client fixture configures this global instance (e.g. for login).
    # We are patching _make_request on the RagMetricsClient class, so it affects the global instance.
    
    # No need to manually manage access_token here if the fixture handles login
    # and the test is correctly skipped if TEST_MOCK=true (which it is).
    # The skipif decorator handles the TEST_MOCK=true case.

    response = obj.save() # This should now work if _make_request returns the dict

    assert obj.id == "new-id-123"
    assert response is obj # save() method returns self on success

    # Verify the call to the patched _make_request on the client instance
    # The _make_request on the class was patched, so it applies to the global ragmetrics_client instance.
    # We expect the global ragmetrics_client's _make_request to have been called.
    # However, the mock_make_request is passed directly to the test function by @patch decorator.
    mock_make_request.assert_called_once_with(
        method="post",
        endpoint="/api/client/mockdata/save/",
        json_payload={'name': 'test_obj', 'value': 10, 'id': None} # Expect id to be None before saving
        # headers are handled internally by the real _make_request, so not asserted here on the direct mock
    ) 