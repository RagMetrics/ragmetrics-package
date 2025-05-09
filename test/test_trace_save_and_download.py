# test/test_ragmetrics_object_download_by_id_success.py
import pytest
import logging
from ragmetrics.api import ragmetrics_client
from ragmetrics.trace import Trace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trace_save_and_download(ragmetrics_test_client):
    """
    Tests downloading a Trace object by ID using the live API.
    Tests creating a trace and then downloading it by ID.
    
    A Trace should require only input and output, with all other fields being optional.
    """
    # Create the test data up front to be used throughout the test
    input_text = "What is the capital of France?"
    output_text = "The capital of France is Paris."
    
    # Client setup
    client = ragmetrics_test_client
    
    # Verify we have a valid token for live API
    assert client.access_token is not None, "Live API test requires login and access token"
    
    # Create a minimal Trace object with just input and output
    original_trace = Trace(
        input=input_text,
        output=output_text
    )
    
    # Save the trace
    logger.info("Saving trace to server...")
    saved_trace = original_trace.save()
    assert saved_trace is not None, "Trace save should return a trace object"
    assert saved_trace.id is not None, "Saved trace should have an ID"
    logger.info(f"Trace saved with ID: {saved_trace.id}")
    
    # Download the trace
    logger.info(f"Downloading trace with ID: {saved_trace.id}")
    downloaded_trace = Trace.download(id=saved_trace.id)
    
    # Assertions - focus on the minimal required fields
    assert downloaded_trace is not None, "Download should return a Trace"
    assert isinstance(downloaded_trace, Trace), "Expected Trace instance"
    assert downloaded_trace.id == saved_trace.id, "Downloaded trace should have the same ID"
    
    # Print trace details for debugging
    logger.info(f"Downloaded trace ID: {downloaded_trace.id}")
    logger.info(f"Downloaded trace input: {downloaded_trace.input}")
    logger.info(f"Downloaded trace output: {downloaded_trace.output}")
    
    # The server might be returning fields differently - check raw_input/raw_output if input/output are None
    if downloaded_trace.input is None and downloaded_trace.raw_input is not None:
        logger.info("Using raw_input as input is None")
        if isinstance(downloaded_trace.raw_input, dict) and "content" in downloaded_trace.raw_input:
            assert downloaded_trace.raw_input["content"] == input_text, "Downloaded trace raw_input should contain the original input"
        else:
            assert str(downloaded_trace.raw_input) == input_text, "Downloaded trace raw_input should match the original input"
    else:
        assert downloaded_trace.input == input_text, "Downloaded trace should have the same input"
    
    if downloaded_trace.output is None and downloaded_trace.raw_output is not None:
        logger.info("Using raw_output as output is None")
        if isinstance(downloaded_trace.raw_output, dict) and "content" in downloaded_trace.raw_output:
            assert downloaded_trace.raw_output["content"] == output_text, "Downloaded trace raw_output should contain the original output"
        else:
            assert str(downloaded_trace.raw_output) == output_text, "Downloaded trace raw_output should match the original output"
    else:
        assert downloaded_trace.output == output_text, "Downloaded trace should have the same output"