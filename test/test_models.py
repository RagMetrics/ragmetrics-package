# test/test_models.py
import pytest
from unittest.mock import patch, MagicMock
import requests

from ragmetrics.models import RagMetricsObject
from ragmetrics.api import ragmetrics_client, RagMetricsAPIError, RagMetricsAuthError # Import global client and error

# --- Test Setup --- 

# Create a concrete subclass for testing
class MockDataObject(RagMetricsObject):
    object_type = "mockdata" # Define the API object type

    def __init__(self, name: str, value: int, id: str = None):
        self.name = name
        self.value = value
        self.id = id # ID is usually assigned after saving

    def to_dict(self):
        # Simple serialization for testing
        return {"name": self.name, "value": self.value, "id": self.id}
    
    # from_dict uses the default implementation which works with __init__

@pytest.fixture
def mock_api_post_success():
    """Mocks a successful POST request (e.g., for save)."""
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    # API often returns the saved object nested under its type
    mock_resp.json.return_value = {"mockdata": {"id": "new-id-123", "name": "saved_obj", "value": 100}}
    return mock_resp

@pytest.fixture
def mock_api_get_success():
    """Mocks a successful GET request (e.g., for download)."""
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"mockdata": {"id": "dl-id-456", "name": "downloaded_obj", "value": 200}}
    return mock_resp

@pytest.fixture
def mock_api_failure():
    """Mocks a failed API request."""
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 400
    mock_resp.text = "Bad Request Data"
    return mock_resp

# --- Tests --- 

@patch('ragmetrics.api.ragmetrics_client._make_request')
def test_ragmetricsobject_save_success(mock_make_request, mock_api_post_success):
    mock_make_request.return_value = mock_api_post_success
    # Assume client is logged in for save/download
    ragmetrics_client.access_token = "fake_token"
    
    obj = MockDataObject(name="test_obj", value=10)
    assert obj.id is None
    
    response = obj.save()
    
    assert response.status_code == 200
    assert obj.id == "new-id-123" # ID should be set from response
    mock_make_request.assert_called_once_with(
        method="post", 
        endpoint="/api/client/mockdata/save/", 
        json={"name": "test_obj", "value": 10, "id": None}, # Initial ID is None
        headers={"Authorization": "Token fake_token"}
    )
    ragmetrics_client.access_token = None # Clean up

@patch('ragmetrics.api.ragmetrics_client._make_request')
def test_ragmetricsobject_save_failure(mock_make_request, mock_api_failure):
    mock_make_request.return_value = mock_api_failure
    ragmetrics_client.access_token = "fake_token"

    obj = MockDataObject(name="bad_obj", value=-1)
    with pytest.raises(Exception) as excinfo:
        obj.save()
    assert "Failed to save mockdata" in str(excinfo.value)
    assert "400" in str(excinfo.value)
    assert "Bad Request Data" in str(excinfo.value)
    assert obj.id is None # ID should not be set on failure

    ragmetrics_client.access_token = None

def test_ragmetricsobject_save_not_logged_in():
    ragmetrics_client.access_token = None # Ensure not logged in
    obj = MockDataObject(name="no_login", value=1)
    with pytest.raises(RagMetricsAuthError) as excinfo: 
        obj.save()
    assert "not authenticated" in str(excinfo.value)

@patch('ragmetrics.api.ragmetrics_client._make_request')
def test_ragmetricsobject_download_by_id_success(mock_make_request, mock_api_get_success):
    mock_make_request.return_value = mock_api_get_success
    ragmetrics_client.access_token = "fake_token"

    downloaded_obj = MockDataObject.download(id="dl-id-456")

    assert isinstance(downloaded_obj, MockDataObject)
    assert downloaded_obj.id == "dl-id-456"
    assert downloaded_obj.name == "downloaded_obj"
    assert downloaded_obj.value == 200
    mock_make_request.assert_called_once_with(
        method="get", 
        endpoint="/api/client/mockdata/download/", 
        params={"id": "dl-id-456"},
        headers={"Authorization": "Token fake_token"}
    )
    ragmetrics_client.access_token = None

@patch('ragmetrics.api.ragmetrics_client._make_request')
def test_ragmetricsobject_download_by_name_success(mock_make_request, mock_api_get_success):
    mock_make_request.return_value = mock_api_get_success
    ragmetrics_client.access_token = "fake_token"

    downloaded_obj = MockDataObject.download(name="downloaded_obj")

    assert downloaded_obj.id == "dl-id-456"
    mock_make_request.assert_called_once_with(
        method="get", 
        endpoint="/api/client/mockdata/download/", 
        params={"name": "downloaded_obj"},
        headers={"Authorization": "Token fake_token"}
    )
    ragmetrics_client.access_token = None

@patch('ragmetrics.api.ragmetrics_client._make_request')
def test_ragmetricsobject_download_failure(mock_make_request, mock_api_failure):
    mock_make_request.return_value = mock_api_failure
    ragmetrics_client.access_token = "fake_token"

    with pytest.raises(Exception) as excinfo:
        MockDataObject.download(id="nonexistent-id")
    assert "Failed to download mockdata" in str(excinfo.value)
    assert "400" in str(excinfo.value)
    assert "Bad Request Data" in str(excinfo.value)

    ragmetrics_client.access_token = None

def test_ragmetricsobject_download_not_logged_in():
    ragmetrics_client.access_token = None
    with pytest.raises(RagMetricsAuthError) as excinfo:
        MockDataObject.download(id="some-id")
    assert "not authenticated" in str(excinfo.value)

def test_ragmetricsobject_download_no_id_or_name():
    ragmetrics_client.access_token = "fake_token"
    with pytest.raises(ValueError) as excinfo:
        MockDataObject.download()
    assert "Either id or name must be provided" in str(excinfo.value)
    ragmetrics_client.access_token = None 