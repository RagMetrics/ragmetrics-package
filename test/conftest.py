import pytest
import os
import glob

# Tell pytest to collect from the 'test' directory but ignore
# the actual test files during collection phase
collect_ignore = []

def pytest_configure(config):
    """Configure test collection - ignore files when collecting only"""
    if config.getoption('--collect-only', False):
        # During test collection phase, ignore all Python files in the test directory
        # This prevents pytest from executing the module-level code in those files
        test_dir = os.path.dirname(__file__)
        
        # Find all Python files in the test directory
        python_files = glob.glob(os.path.join(test_dir, '*.py'))
        
        # Add each Python file to collect_ignore (except conftest.py itself)
        for file_path in python_files:
            filename = os.path.basename(file_path)
            if filename != 'conftest.py':
                collect_ignore.append(filename)
