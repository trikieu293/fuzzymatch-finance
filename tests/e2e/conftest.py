"""Global configuration and shared fixtures for the E2E Test Suite."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest


def require_feature(
    module_name: str,
    symbol: str | None = None,
    milestone: str = "M1",
    expected_param: str | None = None,
) -> Any:
    """Safely import a module or symbol with progressive milestone gating.

    If the requested module or symbol is not yet implemented, or does not yet
    support the expected parameter from the target milestone, raises a pytest.skip
    identifying the corresponding milestone rather than failing test collection.
    """
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(f"Milestone {milestone} module '{module_name}' not yet available: {exc}")
    if symbol is not None:
        if not hasattr(mod, symbol):
            pytest.skip(
                f"Milestone {milestone} symbol '{symbol}' not yet available in '{module_name}'"
            )
        sym = getattr(mod, symbol)
        if expected_param is not None and callable(sym):
            try:
                sig = inspect.signature(sym)
                if expected_param not in sig.parameters:
                    pytest.skip(
                        f"Milestone {milestone} symbol '{symbol}' does not yet support parameter '{expected_param}'"
                    )
            except (ValueError, TypeError):
                pass
        return sym
    return mod


@pytest.fixture(scope="session")
def require_milestone() -> Any:
    """Fixture providing milestone requirement checker."""
    return require_feature
