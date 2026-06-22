import numpy as np
import pytest

@pytest.fixture
def simple_2d_mask():
    """A 10x10 mask with one object (label=1) in the center"""
    mask = np.zeros((10,10), dtype=int)
    mask[3:7, 3:7] = 1
    return mask

@pytest.fixture
def simple_3d_mask():
    """A 5x10x10 mask with one object across all z-planes."""
    mask = np.zeros((5,10,10), dtype=int)
    mask[:, 3:7, 3:7] = 1
    return mask