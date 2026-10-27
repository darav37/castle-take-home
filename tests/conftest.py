"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _v1_engine_by_default(monkeypatch):
    """The agent tests below were written against the v1 scenario engine; Plan B tests opt in (DECISIONS #179)."""
    from worsey import config
    monkeypatch.setattr(config, "SCENARIO_ENGINE", "v1")
