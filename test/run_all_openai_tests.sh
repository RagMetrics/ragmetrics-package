#!/bin/bash
# Automated testing script for multiple OpenAI versions
# This script runs tests in both v0.x and v1.x environments

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}RagMetrics Multi-Version OpenAI Test Runner${NC}"
echo "============================================="

# Test specific files or all tests
if [ $# -gt 0 ]; then
    TEST_ARGS="$@"
    echo -e "Running tests: ${YELLOW}$TEST_ARGS${NC}"
else
    TEST_ARGS="test/test_openai_versions.py"
    echo -e "Running default tests: ${YELLOW}$TEST_ARGS${NC}"
fi

# --------------------------------------------------------
# PART 1: Setup and run tests in OpenAI v0.x environment
# --------------------------------------------------------
echo -e "\n${BLUE}Setting up OpenAI v0.x environment...${NC}"

# Activate the v0.x environment
source env_openai_v0/bin/activate || { echo -e "${RED}Failed to activate v0.x environment${NC}"; exit 1; }

# Install dependencies if needed
if ! pip freeze | grep -q "openai==0.28.1"; then
    echo -e "${YELLOW}Installing OpenAI v0.28.1 and dependencies...${NC}"
    pip install openai==0.28.1 pytest python-dotenv requests
    pip install -e .
fi

# Run the tests
echo -e "\n${BLUE}Running tests with OpenAI v0.28.1...${NC}"
python -m pytest $TEST_ARGS -v || V0_FAILED=1

# Deactivate the environment
deactivate

# --------------------------------------------------------
# PART 2: Setup and run tests in current environment with OpenAI v1.x
# --------------------------------------------------------
echo -e "\n${BLUE}Running tests in current environment (OpenAI v1.x)...${NC}"

# Activate the main environment
source env/bin/activate || { echo -e "${RED}Failed to activate main environment${NC}"; exit 1; }

# Verify OpenAI version
OPENAI_VERSION=$(python -c "import openai; print(openai.__version__)")
echo -e "Detected OpenAI version: ${YELLOW}$OPENAI_VERSION${NC}"

if [[ "$OPENAI_VERSION" == 1* ]]; then
    # Run the tests
    python -m pytest $TEST_ARGS -v || V1_FAILED=1
else
    echo -e "${RED}Current environment does not have OpenAI v1.x. Skipping these tests.${NC}"
    V1_FAILED=1
fi

# --------------------------------------------------------
# PART 3: Report results
# --------------------------------------------------------
echo -e "\n${BLUE}Test Summary${NC}"
echo "=============="

if [ -z "$V0_FAILED" ] && [ -z "$V1_FAILED" ]; then
    echo -e "${GREEN}All tests passed in both OpenAI v0.x and v1.x environments${NC}"
    exit 0
else
    if [ -n "$V0_FAILED" ]; then
        echo -e "${RED}Tests failed in OpenAI v0.x environment${NC}"
    else
        echo -e "${GREEN}Tests passed in OpenAI v0.x environment${NC}"
    fi
    
    if [ -n "$V1_FAILED" ]; then
        echo -e "${RED}Tests failed in OpenAI v1.x environment${NC}"
    else
        echo -e "${GREEN}Tests passed in OpenAI v1.x environment${NC}"
    fi
    
    exit 1
fi 