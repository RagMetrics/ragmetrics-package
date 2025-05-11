"""
This test file shows how to handle both versions of the OpenAI API with RagMetrics.
It defines tests that can run with either the legacy (v0.x) or modern (v1.x) OpenAI SDK.
"""

import pytest
import ragmetrics
from ragmetrics.trace import Trace  # Import Trace class
import os
import logging
import time
logger = logging.getLogger(__name__)

def openai_version():
    """
    Return 'v1' if using OpenAI API version 1.x, otherwise 'v0'.
    Raises ImportError if OpenAI is not installed.
    """
    import openai
    version = getattr(openai, '__version__', '0.0.0')
    if version.startswith('1.'):
        return 'v1'
    elif version.startswith('0.'):
        return 'v0'
    else:
        raise ValueError(f"Unexpected OpenAI version: {version}")

# Skip all tests if OpenAI not installed
try:
    OPENAI_VERSION = openai_version()
except ImportError:
    pytest.skip("OpenAI package not installed", allow_module_level=True)

@pytest.mark.parametrize("api_version", [
    pytest.param("v0", marks=pytest.mark.skipif(OPENAI_VERSION == "v1", reason="Requires OpenAI v0.x")), 
    pytest.param("v1", marks=pytest.mark.skipif(OPENAI_VERSION == "v0", reason="Requires OpenAI v1.x"))
])
def test_openai_integration(ragmetrics_test_client, openai_api_key, api_version):
    """
    Test RagMetrics integration with either OpenAI v0.x or v1.x API
    
    This parameterized test handles both API versions, focusing only on the 
    differences in how the API is called.
    """
    logger.info(f"Testing OpenAI {api_version}, detected in environment: {OPENAI_VERSION}")
    
    # Common message payload and parameters
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep answers brief."},
        {"role": "user", "content": "What is the capital of NJ?"}
    ]    
    params = {
        "model": "gpt-3.5-turbo",
        "max_tokens": 20,
        "messages": messages
    }
    monitor_metadata = {"test": f"{api_version}_api", "key2": "from_monitor"}
    create_metadata = {"key2": "from_create", "key3": "value3"}
    
    # Get client, monitor it, and make API call
    if api_version == "v1":
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        ragmetrics.monitor(
            client, 
            metadata=monitor_metadata
        )
        
        response = client.chat.completions.create(
            **params, 
            metadata=create_metadata
        )
        content = response.choices[0].message.content
    elif api_version == "v0":
        import openai
        openai.api_key = openai_api_key
        ragmetrics.monitor(
            openai, 
            metadata=monitor_metadata
        )        
        response = openai.ChatCompletion.create(
            **params, 
            metadata=create_metadata
        )
        content = response["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Invalid API version: {api_version}")
    
    # Common validations
    logger.info(f"OpenAI {api_version} response: {content}")
    assert content, f"No content received in {api_version} response"    
    
    # Verify RagMetrics captured the trace
    trace_len = len(ragmetrics_test_client.test_logged_trace_ids)
    assert trace_len == 1, \
        f"{trace_len} traces were logged, expected 1"
        
    # Download trace directly and verify metadata independently
    # This provides a robust check using the actual API
    trace_id = ragmetrics_test_client.test_logged_trace_ids[0]
    # Wait a moment for the trace to be available via API
    time.sleep(0.5)
    
    # Download the trace by ID
    downloaded_trace = Trace.download(id=trace_id)
    
    # Verify the downloaded trace has the expected metadata
    assert downloaded_trace is not None, f"Failed to download trace {trace_id}"
    assert hasattr(downloaded_trace, 'metadata'), "Downloaded trace has no metadata attribute"
    
    # Verify metadata has the expected value
    # The metadata should include everything from both monitor() and create() calls,
    # with create() keys taking precedence if there are duplicates
    trace_metadata = downloaded_trace.metadata
    logger.info(f"Downloaded trace metadata: {trace_metadata}")
    expected_metadata = {
        "test": f"{api_version}_api",  # From create(), overrides monitor()'s value
        "key2": "from_create",  # From monitor() only
        "key3": "value3"   # From create() only
    }    

    
    # Check that all expected metadata keys are present with correct values
    for key, value in expected_metadata.items():
        assert key in trace_metadata, f"Missing expected key '{key}' in downloaded trace metadata: {trace_metadata}"
        assert trace_metadata[key] == value, f"Incorrect value for '{key}' key. Expected: {value}, Got: {trace_metadata[key]}"
    # Verify no OpenAI parameters were included
    for param in ["api", "version", "model", "max_tokens", "temperature"]:
        assert param not in trace_metadata, f"Unexpected parameter '{param}' found in downloaded trace metadata: {trace_metadata}"
    
    logger.info(f"✅ Downloaded trace metadata verified: {trace_metadata}")
