"""
Pytest configuration and deterministic environment setup.
"""

import os

# Set dummy API key to allow module imports without a live key
os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_for_deterministic_testing_0000000000")
