"""
Pytest configuration for the test suite.
"""
import sys
import os

# Add src directory to path so imports work correctly
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
