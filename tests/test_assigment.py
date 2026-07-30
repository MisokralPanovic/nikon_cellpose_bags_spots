import numpy as np
import pytest

from spot_detector.exceptions import DimensionMismatchError
from spot_detector.segmentation_detection import assign_spots_to_mask

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def simple_2d_mask():
    """A 10x10 mask with one object (label=1) in the center"""
    mask = np.zeros((10, 10), dtype=int)
    mask[3:7, 3:7] = 1
    return mask


@pytest.fixture
def simple_3d_mask():
    """A 5x10x10 mask with one object across all z-planes."""
    mask = np.zeros((5, 10, 10), dtype=int)
    mask[:, 3:7, 3:7] = 1
    return mask


# =====================================================================
# assign_spots_to_mask
# =====================================================================


def test_spots_inside_object(simple_2d_mask):
    coords = np.array([[5, 5]])
    labels = assign_spots_to_mask(coordinates=coords, masks=simple_2d_mask)
    assert labels[0] == 1


def test_spots_outside_object(simple_2d_mask):
    coords = np.array([[0, 0]])
    labels = assign_spots_to_mask(coordinates=coords, masks=simple_2d_mask)
    assert labels[0] == 0


def test_empty_coordinates(simple_2d_mask):
    coords = np.array([])
    labels = assign_spots_to_mask(coordinates=coords, masks=simple_2d_mask)
    assert len(labels) == 0


def test_3d_assigment(simple_3d_mask):
    coords = np.array([[2, 5, 5]])
    labels = assign_spots_to_mask(coordinates=coords, masks=simple_3d_mask)
    assert labels[0] == 1


def test_dimension_mismatch_raises(simple_2d_mask):
    coords = np.array([[1, 5, 5]])
    with pytest.raises(DimensionMismatchError):
        assign_spots_to_mask(coordinates=coords, masks=simple_2d_mask)
