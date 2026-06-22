import numpy as np
import pandas as pd
import pytest
from skimage.measure import regionprops_table
from spot_detector.obejct_measurement import measure_objects

# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def base_params():
    return {
        "dx": 0.5,
        "dz": 2.0,
        "mode": "2d",
        "condition": "Control",
        "source_file": "test.nd2",
        "experiment": "exp1",
        "scene": 0        
    }

@pytest.fixture
def one_object_2d():
    """Generates a 2D mask with a single 10x10 square object."""
    mask = np.zeros((20,20), dtype=int)
    mask[5:15, 5:15] = 1 # 100 pixels
    return mask

@pytest.fixture
def two_objects_2d():
    """Generates a 2D mask with two separate objects of different sizes."""
    mask = np.zeros((20, 20), dtype=int)
    mask[2:5, 2:5] = 1 # 9 pixels
    mask[10:15, 10:15] = 2 # 25 pixels
    return mask

@pytest.fixture
def one_object_3d():
    """Generates a 3D mask spanning 2 slices, 10x10 in XY (200 pixels total)."""
    mask = np.zeros((5, 20, 20), dtype=int)
    mask[1:3, 5:15, 5:15] = 1
    return mask

# =====================================================================
# 2D Mode Tests
# =====================================================================

class TestMeasureObjects2D:
    
    def test_returns_one_row(self, one_object_2d, base_params):
        df = measure_objects(
            masks = one_object_2d,
            spot_labels = np.array([1, 1, 0]),
            **base_params
        )
        assert len(df) == 1
    
    def test_spot_count_correct(self, one_object_2d, base_params):
        df = measure_objects(
        masks = one_object_2d,
        spot_labels = np.array([1, 1, 0]),
        **base_params
        )
        assert df["Spot_Count"].iloc[0] == 2


    def test_area_calculated(self, one_object_2d, base_params):
        # 100 pixels * 0.5 dx * 0.5 dx = 25.0 um2
        params = {**base_params, "dx": 0.5}
        df = measure_objects(
        masks=one_object_2d,
        spot_labels=np.array([]),
        **params
        )    
        assert df["Area_um2"].iloc[0] == pytest.approx(25.0)

    def test_3d_columns_are_nan_in_2d(self, one_object_2d, base_params):
        df = measure_objects(
        masks=one_object_2d,
        spot_labels=np.array([]),
        **base_params
        )
        assert pd.isna(df["Volume_um3"].iloc[0])
        assert pd.isna(df["Z_Span_um"].iloc[0])
        assert pd.isna(df["Centroid_Z_um"].iloc[0])

    def test_metadata_columns(self, one_object_2d, base_params):
        params = {**base_params, 
                "condition": "Control", "source_file": "test.nd2", 
                "experiment": "exp1", "scene": 0}
        df = measure_objects(
        masks=one_object_2d,
        spot_labels=np.array([]),
        **params
        )
        assert df["Condition"].iloc[0] == "Control"
        assert df["Source File"].iloc[0] == "test.nd2"
        assert df["Experiment"].iloc[0] == "exp1"
        assert df["Scene"].iloc[0] == 0

    def test_multiple_objects_and_spot_isolation(self, two_objects_2d, base_params):
        df = measure_objects(
            masks = two_objects_2d,
            spot_labels = np.array([1, 1, 2]),
            **base_params
        )
        assert len(df) == 2
        
        obj1 = df[df["Object_Label"] == 1].iloc[0]
        # 9 pixels * 0.5 dx * 0.5 dx = 25.0 um2
        assert obj1["Area_um2"] == pytest.approx(2.25)
        assert obj1["Spot_Count"] == 2
    
        obj2 = df[df["Object_Label"] == 2].iloc[0]
        # 25 pixels * 0.5 dx * 0.5 dx = 25.0 um2
        assert obj2["Area_um2"] == pytest.approx(6.25)
        assert obj2["Spot_Count"] == 1
    
# =====================================================================
# 3D Mode Tests
# =====================================================================

class TestMeasureObjects3D:
    
    def test_3d_measurements_and_centroids(self, one_object_3d, base_params):
        params = {**base_params, "mode": "3d", "dx": 0.5, "dz": 2.0}
        df = measure_objects(
        masks = one_object_3d,
        spot_labels = np.array([1, 1]),
        **params
        )          
        assert len(df) == 1
        # Volume = 200 pixels * 2.0 dz * 0.5 dx * 0.5 dx = 100.0 um3
        assert df["Volume_um3"].iloc[0] == pytest.approx(100.0)
        # Z Span = (bbox-3 minus bbox-0) * dz -> (3 - 1) * 2.0 = 4.0 um
        assert df["Z_Span_um"].iloc[0] == pytest.approx(4.0)
        # Centroid Z = index 1.5 * dz (2.0) = 3.0 um
        assert df["Centroid_Z_um"].iloc[0] == pytest.approx(3.0)
        
        assert pd.isna(df["Area_um2"].iloc[0])
        assert pd.isna(df["Spot_Density_per_um2"].iloc[0])

# =====================================================================
# Edge Cases & Validation Tests
# =====================================================================

class TestMeasureObjectsEdgeCases:
    
    def test_empty_mask_return_empty_dataframe(self, base_params):
        df = measure_objects(
        masks = np.zeros((20, 20), dtype=int),
        spot_labels = np.array([]),
        **base_params
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "Spot_Count" in df.columns
        
    def test_spot_labels_exceed_max_mask_id(self, one_object_2d, base_params):
        df = measure_objects(
        masks = one_object_2d,
        spot_labels = np.array([1, 99]),
        **base_params
        )
        assert df["Spot_Count"].iloc[0] == 1
    
    def test_invalid_mode_raises_error(self, one_object_2d, base_params):
        params = {**base_params, "mode": "invalid_mode"}
        with pytest.raises(AssertionError, match="mode must be '2d' or '3d'"):
            measure_objects(
                masks = one_object_2d, spot_labels = np.array([]),
                **params
            )
        
    def test_mismatched_dimensions_raises_error(self, one_object_2d, base_params):
        params = {**base_params, "mode": "3d"}
        with pytest.raises(AssertionError, match="Expected 3D mask for mode='3d'"):
            measure_objects(
                masks = one_object_2d, spot_labels = np.array([]),
                **params
            )