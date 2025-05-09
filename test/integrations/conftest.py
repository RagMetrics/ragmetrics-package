# conftest.py for test/integrations
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_openai_completions_object():
    """Mocks an OpenAI client's chat.completions object and its create method."""
    completions_obj = MagicMock()
    # Mock the original create method BEFORE it gets wrapped
    original_create_method = MagicMock(return_value="original_response") 
    completions_obj.create = original_create_method
    return completions_obj, original_create_method 