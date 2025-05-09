# test/test_ragmetrics_object_download_no_id_or_name.py
import pytest
from ragmetrics.api import ragmetrics_client, RagMetricsAPIError, RagMetricsAuthError
from ..conftest import MockDataObject
# MockDataObject from conftest.py

# This tests client-side validation, doesn't need fixture or mocking of API
def test_ragmetricsobject_download_no_id_or_name():
    # Need to ensure client *thinks* it's logged in to get past the first auth check
    original_token = ragmetrics_client.access_token
    original_logging_off = ragmetrics_client.logging_off
    ragmetrics_client.access_token = "fake_token_for_value_error_test"
    ragmetrics_client.logging_off = False
    # ragmetrics_client.logged_in = True # Simulate logged in - attr doesn't exist

    with pytest.raises(ValueError) as excinfo:
        MockDataObject.download()
    assert "Either id or name must be provided" in str(excinfo.value)

    # Restore global client state
    ragmetrics_client.access_token = original_token
    ragmetrics_client.logging_off = original_logging_off 