#!/usr/bin/env python3
"""Minimal test runner — runs pytest and prints summary."""
import sys
import pytest

sys.exit(pytest.main([
    "tests/",
    "-v",
    "--tb=short",
    "-q",
]))
