"""Shared test fixtures."""

import pytest

from soluai import MODELS_DIR


@pytest.fixture(scope="session")
def predictor():
    """The trained model (tests that need it are skipped if it has not been trained yet)."""
    if not (MODELS_DIR / "LATEST").exists():
        pytest.skip("no trained model - run `python -m soluai.train` first")
    from soluai.predict import Predictor

    return Predictor()
