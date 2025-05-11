# conftest.py - pytest configuration and fixtures
# Add fixtures here if needed across multiple test files 
import os
import pytest
import ragmetrics # Import the main package
from ragmetrics.api import logger as ragmetrics_logger # Assuming logger is in ragmetrics.api
from dotenv import load_dotenv

# Imports needed for the moved fixtures/class
from unittest.mock import MagicMock
import requests
from ragmetrics.base_object import RagMetricsObject

# --- Moved from test_ragmetrics_object.py ---

# Create a concrete subclass for testing RagMetricsObject
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

# --- End Moved Code ---

@pytest.fixture(scope="session")
def openai_api_key():
    """
    Load the OpenAI API key from the environment
    
    This fixture will:
    1. Load the .env file from the project root
    2. Check for and return the OPENAI_API_KEY environment variable
    3. Skip tests if the API key is not available
    
    Returns:
        str: The OpenAI API key
    """
    # Load .env file from the root of the ragmetrics-package 
    api_key = os.environ.get('OPENAI_API_KEY', None)
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in .env file or environment variables")
        
    return api_key

@pytest.fixture(scope="session")
def ragmetrics_test_client():
    """
    Initializes and configures the RagMetrics client for the test session.
    Handles .env loading, TEST_MOCK logic, login, and applies test-specific settings.
    Returns the configured ragmetrics.ragmetrics_client instance.
    Runs once per test session when first requested.
    """
    # Load .env file from the root of the ragmetrics-package,
    # assuming conftest.py is in ragmetrics-package/test/
    # and .env is in ragmetrics-package/
    dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path)
    
    test_mock = os.getenv("TEST_MOCK", "False").lower() == "true"
    
    # Get the global client instance to configure it
    client = ragmetrics.ragmetrics_client
    
    # Ensure test_logged_trace_ids is clean for this session/test usage
    if hasattr(client, 'test_logged_trace_ids'):
        client.trace_ids = [] 

    if test_mock:
        ragmetrics.login(off=True) # This modifies the global client
        ragmetrics_logger.info("RagMetrics: TEST_MOCK is True. Global client login is OFF for the test session.")
    else:
        ragmetrics_logger.info("RagMetrics: TEST_MOCK is False. Attempting real login for the test session...")
        try:
            ragmetrics.login() # Attempts login using env vars, modifies global client
            ragmetrics_logger.info("RagMetrics: Real login successful for the test session.")
        except Exception as e:
            ragmetrics_logger.error(f"RagMetrics: Real login failed during session setup: {e}. Falling back to login(off=True).")
    
    # Apply further test-specific configurations to the global client
    if not test_mock: # Only set logging_off to False if not mocking
        client.logging_off = False 
    
    if not client.access_token: # This can still apply if login(off=True) was used or real login failed
        client.access_token = "dummy-test-token"
        ragmetrics_logger.info("RagMetrics: Applied dummy access token as no token was present after login attempt.")
        
    return client # Return the configured global client

# The ragmetrics_client_for_tests fixture is now removed as its functionality is merged above. 