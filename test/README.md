# RagMetrics Testing

This directory contains tests for the RagMetrics package, including support for multiple versions of the OpenAI API.

## Testing OpenAI API Versions

RagMetrics supports both the legacy OpenAI API (v0.x) and the modern OpenAI API (v1.x). The test suite is designed to work with both versions, using version detection and conditional code paths.

### Testing Approaches

#### Option 1: Using the Version Detection (Automatic)

The `test_openai_versions.py` file contains tests that automatically adapt to the installed OpenAI version:

```python
def is_openai_version_1():
    """Check if using OpenAI API version 1.x"""
    try:
        openai_version = getattr(importlib.import_module('openai'), '__version__', '0.0.0')
        return openai_version.startswith('1.')
    except ImportError:
        return False
```

Tests are then conditionally skipped based on version:

```python
@pytest.mark.skipif(not is_openai_version_1(), reason="Requires OpenAI v1.x")
def test_openai_v1_integration(...):
    # Tests for v1.x API
    
@pytest.mark.skipif(is_openai_version_1(), reason="Requires OpenAI v0.x")
def test_openai_v0_integration(...):
    # Tests for v0.x API
```

#### Option 2: Using the Multi-Version Test Script

To test against multiple OpenAI versions in sequence, use the provided script:

```bash
# Make the script executable
chmod +x test/run_multi_version_tests.sh

# Run tests against all specified OpenAI versions (default)
./test/run_multi_version_tests.sh

# Run all tests with both OpenAI versions
./test/run_multi_version_tests.sh --all

# Run a specific test file with both versions
./test/run_multi_version_tests.sh -t test_api.py

# Test with specific OpenAI versions
./test/run_multi_version_tests.sh -v 0.28.1,1.26.0,1.27.0

# Keep environments after testing (for debugging)
./test/run_multi_version_tests.sh -k
```

This script creates isolated virtual environments for each OpenAI version, runs the tests, and provides a summary.

##### Full Script Options

```
Usage: run_multi_version_tests.sh [options]

Options:
  -h, --help           Show this help message
  -a, --all            Run all tests (default runs only test_openai_versions.py)
  -t, --tests PATTERN  Run specific tests matching the pattern
  -v, --versions VERS  Comma-separated list of OpenAI versions to test (default: 0.28.1,1.27.0)
  -k, --keep           Keep temporary environments after testing
```

If you use the `-k` option, the script will show you how to activate each environment for manual debugging.

#### Option 3: Using GitHub Actions Matrix Testing

For CI/CD, we use a matrix approach in GitHub Actions that tests multiple Python versions and OpenAI versions in parallel. See `.github/workflows/test-openai-versions.yml`.

## Test Organization

- `test_openai_versions.py` - Version-agnostic tests that work with any OpenAI version
- `integrations/openai_chat_wrapper/` - Tests for the OpenAI chat wrapper functionality
- `test_logtrace.py` - Example script showing how to use the library with different LLM providers

## Best Practices

1. **Don't mix OpenAI versions in a single environment**
   - Always test with specific, pinned versions
   - Use version detection for conditional code paths

2. **Use mocks for external API calls**
   - The test suite includes mock responses for both v0.x and v1.x API formats
   - This prevents tests from making real API calls

3. **Adding new tests**
   - When adding new OpenAI-related tests, follow the version detection pattern
   - Use the `@pytest.mark.skipif` decorator to skip tests for incompatible versions

4. **Using test markers**
   - Tag tests with `v0` or `v1` to run specific tests for each version:
   ```bash
   # Run only v1.x tests
   pytest -m "v1" test/
   
   # Run only v0.x tests
   pytest -m "v0" test/
   ``` 