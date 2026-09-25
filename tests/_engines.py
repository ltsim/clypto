"""Shared helpers to run collection tests on both engines (``legacy`` and ``vectorize``)."""
import json
from pathlib import Path

import pytest

ENGINES = ("legacy", "vectorize")
EXACTNESS_PATH = Path(__file__).parent / "golden" / "vectorize_exactness.json"


def statistical_only():
    """Class names whose vectorize version is only statistically equivalent to legacy."""
    return set(json.loads(EXACTNESS_PATH.read_text())["statistical"])


def skip_if_not_bit_identical(engine, name):
    if engine == "vectorize" and name in statistical_only():
        pytest.skip("vectorize version is statistically equivalent only (see tests/test_native_statistical.py)")
