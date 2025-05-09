# test/test_ragmetrics_object_download_by_name_success.py
import pytest
import os
from unittest.mock import patch, ANY
from ragmetrics.api import ragmetrics_client, RagMetricsAPIError, RagMetricsAuthError
from ..conftest import MockDataObject
# MockDataObject and mock_api_get_success from conftest.py

@pytest.mark.skipif(os.getenv("TEST_MOCK", "False").lower() == "true", reason="download requires API calls, skipped when TEST_MOCK is true")
@patch('ragmetrics.api.ragmetrics_client._make_request') # Patch global client's method
def test_ragmetricsobject_download_by_name_success(mock_make_request, mock_api_get_success, ragmetrics_test_client): # Use fixture
    client = ragmetrics_test_client # Assign for clarity and access to token
    assert client.access_token is not None 

    mock_make_request.return_value = mock_api_get_success.json() 

    downloaded_obj = MockDataObject.download(name="downloaded_obj")

    assert downloaded_obj.id == "dl-id-456"
    
    mock_make_request.assert_called_once_with(
        method="get", 
        endpoint="/api/client/mockdata/download/", 
        params={"name": "downloaded_obj"}
    ) 