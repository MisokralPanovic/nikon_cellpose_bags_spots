import numpy as np
import pytest
from spot_detector.segmentation_detection import assign_spots_to_mask

def test_spots_inside_object(simple_2d_mask):
    coords = np.array([[5,5]])
    labels = assign_spots_to_mask(coordinates=coords, masks=simple_2d_mask)
    assert labels[0] == 1
    
def test_spots_outside_object(simple_2d_mask):
    coords = np.array([[0,0]])
    labels = assign_spots_to_mask(coordinates=coords, masks=simple_2d_mask)
    assert labels[0] == 0
    
def test_empty_coordinates(simple_2d_mask):
    coords = np.array([])
    labels = assign_spots_to_mask(coordinates=coords, masks=simple_2d_mask)
    assert len(labels) == 0
    
def test_3d_assigment(simple_3d_mask):
    coords = np.array([[2,5,5]])
    labels = assign_spots_to_mask(coordinates=coords, masks=simple_3d_mask)
    assert labels[0] == 1
    
def test_dimension_mismatch_raises(simple_2d_mask):
    coords = np.array([[1,5,5]])
    with pytest.raises(ValueError):
        assign_spots_to_mask(coordinates=coords, masks=simple_2d_mask)