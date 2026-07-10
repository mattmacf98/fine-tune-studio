"""Smoke tests: core package imports and local MPS availability."""

import pytest
import torch

import core


def test_core_imports() -> None:
    """Core package is on the path and importable."""
    assert core is not None


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available (expected on Linux CI runners)",
)
def test_mps_available() -> None:
    """PyTorch sees Apple Metal (MPS) on local Mac dev machines."""
    assert torch.backends.mps.is_available()
    assert torch.backends.mps.is_built()
