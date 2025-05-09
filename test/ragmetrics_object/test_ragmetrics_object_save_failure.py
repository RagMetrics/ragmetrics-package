# test/test_ragmetrics_object_save_failure.py
import pytest
import os
from unittest.mock import patch, ANY # Import ANY
import json # for json.dumps in assertion
import requests # Import requests for exceptions
from ragmetrics.api import ragmetrics_client, RagMetricsAPIError, RagMetricsAuthError, RagMetricsError 
from ..conftest import MockDataObject
# MockDataObject and mock_api_failure from conftest.py

@pytest.mark.skipif(os.getenv("TEST_MOCK", "False").lower() == "true", reason="save requires API calls, skipped when TEST_MOCK is true")
# Patch the underlying requests.post call that _make_request uses
@patch('ragmetrics.api.requests.post') 
def test_ragmetricsobject_save_failure(mock_requests_post, mock_api_failure, ragmetrics_test_client): # mock_requests_post is the new mock
    client = ragmetrics_test_client
    assert client.access_token is not None

    obj_to_save = MockDataObject(name="fail_obj", value=500)

    # Configure the mock for requests.post to return our mock_api_failure response
    mock_requests_post.return_value = mock_api_failure
    
    # Configure mock_api_failure (the Response object) so its raise_for_status() raises RagMetricsAPIError
    # This is what _make_request will call and catch
    error_to_be_raised_by_make_request = RagMetricsAPIError(
        f"API error for POST {client.base_url}/api/client/mockdata/save/: 400 - Bad Request Data...",
        status_code=400,
        response_text="Bad Request Data"
    ) 
    # The actual HTTPError raised by requests.post().raise_for_status() would be a requests.exceptions.HTTPError
    # _make_request catches this and re-raises as RagMetricsAPIError.
    # So, we set mock_api_failure.raise_for_status to raise the underlying HTTPError.
    http_error_sim = requests.exceptions.HTTPError(response=mock_api_failure) # Simulate HTTPError
    mock_api_failure.raise_for_status.side_effect = http_error_sim

    # Now, when obj_to_save.save() calls _make_request, and _make_request calls 
    # response.raise_for_status() (on mock_api_failure), it will raise http_error_sim.
    # _make_request should catch this and raise a RagMetricsAPIError.
    with pytest.raises(RagMetricsAPIError) as excinfo: 
        obj_to_save.save()
    
    # Check the caught RagMetricsAPIError details (this comes from _make_request's except block)
    assert excinfo.value.status_code == 400
    assert "API error for POST" in str(excinfo.value)
    assert "/api/client/mockdata/save/" in str(excinfo.value)
    assert "400 - Bad Request Data..." in str(excinfo.value)
    
    # Verify requests.post was called correctly by _make_request
    expected_url = f"{client.base_url.rstrip('/')}/api/client/mockdata/save/"
    expected_headers_for_post = {
        "Accept": "application/json", 
        "Authorization": f"Token {client.access_token}",
        "Content-Type": "application/json" # Added by _make_request
    }
    mock_requests_post.assert_called_once_with(
        expected_url,
        headers=expected_headers_for_post,
        data=json.dumps(obj_to_save.to_dict()), 
        params=None 
    )

    assert obj_to_save.id is None 