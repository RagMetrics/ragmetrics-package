# test/test_ragmetrics_object_download_not_logged_in.py
import pytest
from ragmetrics.api import ragmetrics_client, RagMetricsAPIError, RagMetricsAuthError
from ..conftest import MockDataObject
# MockDataObject from conftest.py

# This test checks the state *before* login, so it doesn't use the fixture.
# Ensure the global client is in the desired state (no token, logging on).
def test_ragmetricsobject_download_not_logged_in():
    original_token = ragmetrics_client.access_token
    original_logging_off = ragmetrics_client.logging_off
    ragmetrics_client.access_token = None
    ragmetrics_client.logging_off = False # Ensure logging is attempted
    # ragmetrics_client.logged_in = False # Attribute doesn't exist

    with pytest.raises(RagMetricsAuthError) as excinfo:
        MockDataObject.download(id="some-id")
    assert "not authenticated" in str(excinfo.value)

    # Restore global client state
    ragmetrics_client.access_token = original_token
    ragmetrics_client.logging_off = original_logging_off 