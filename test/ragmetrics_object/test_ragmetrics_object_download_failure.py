# test/test_ragmetrics_object_download_failure.py
import pytest
import os
from unittest.mock import patch, ANY
import requests # For requests.exceptions.HTTPError
from ragmetrics.api import ragmetrics_client, RagMetricsAPIError, RagMetricsAuthError, RagMetricsError
from ..conftest import MockDataObject
# MockDataObject and mock_api_failure from conftest.py

@pytest.mark.skipif(os.getenv("TEST_MOCK", "False").lower() == "true", reason="download failure requires API calls, skipped when TEST_MOCK is true")
@patch('ragmetrics.api.requests.get') # Patch the underlying requests.get call that _make_request uses
def test_ragmetricsobject_download_failure(mock_requests_get, mock_api_failure, ragmetrics_test_client): # mock_requests_get is new mock
    client = ragmetrics_test_client
    assert client.access_token is not None

    # Configure requests.get to return our mock_api_failure response
    mock_requests_get.return_value = mock_api_failure

    # Configure mock_api_failure (the Response object) so its raise_for_status() raises HTTPError
    # _make_request will call this, catch HTTPError, and re-raise as RagMetricsAPIError.
    http_error_sim = requests.exceptions.HTTPError(response=mock_api_failure)
    mock_api_failure.raise_for_status.side_effect = http_error_sim

    with pytest.raises(RagMetricsAPIError) as excinfo: 
        MockDataObject.download(id="nonexistent-id")
    
    # Check the caught RagMetricsAPIError (this comes from _make_request's except block)
    assert excinfo.value.status_code == 400
    assert "API error for GET" in str(excinfo.value)
    assert "/api/client/mockdata/download/" in str(excinfo.value)
    assert "400 - Bad Request Data..." in str(excinfo.value)
    
    # Verify requests.get was called correctly by _make_request
    expected_url = f"{client.base_url.rstrip('/')}/api/client/mockdata/download/"
    expected_headers_for_get = {
        "Accept": "application/json",
        "Authorization": f"Token {client.access_token}" # Added by _make_request
    }
    mock_requests_get.assert_called_once_with(
        expected_url,
        headers=expected_headers_for_get,
        params={"id": "nonexistent-id"}
    ) 