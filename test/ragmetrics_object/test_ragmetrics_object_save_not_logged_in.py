# test/test_ragmetrics_object_save_not_logged_in.py
import pytest
from ragmetrics.api import ragmetrics_client, RagMetricsAPIError, RagMetricsAuthError 
from ..conftest import MockDataObject
# MockDataObject from conftest.py

# This test checks the state *before* login, so it doesn't use the fixture.
# Ensure the global client is in the desired state (no token, logging on).
def test_ragmetricsobject_save_not_logged_in():
    original_token = ragmetrics_client.access_token
    original_logging_off = ragmetrics_client.logging_off
    ragmetrics_client.access_token = None 
    ragmetrics_client.logging_off = False # Ensure logging is attempted
    # ragmetrics_client.logged_in = False # Attribute doesn't exist
    
    obj = MockDataObject(name="no_login", value=1)
    # Expect an API error (like 404) or potentially Auth error if server handles it differently
    with pytest.raises(RagMetricsAPIError) as excinfo: 
        obj.save()
    # Check for API/HTTP related error messages rather than specific auth message
    assert "API error" in str(excinfo.value) or "HTTPError" in str(excinfo.value) 
    # Check that the status code (if available on the exception) matches the expected network/API failure
    if hasattr(excinfo.value, 'status_code'):
        assert excinfo.value.status_code >= 400 # 404, 401, 403 etc.
    assert "/api/client/mockdata/save/" in str(excinfo.value) # Ensure the correct endpoint was involved

    # Restore global client state
    ragmetrics_client.access_token = original_token
    ragmetrics_client.logging_off = original_logging_off
    # Re-evaluate logged_in based on restored state if needed, assume login() handles this typically 