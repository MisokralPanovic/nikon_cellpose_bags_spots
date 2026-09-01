import numpy as np
import pytest

from spot_detector.qc_figures import (
    ImageData,
    SpotData,
)

# =====================================================================
# Fixtures
# =====================================================================


# =====================================================================
# SpotData / ImageData dataclasses
# =====================================================================


class TestSpotData:
    def test_2d_has_spots(self):
        coords = np.array([[2.0, 4.0], [6.0, 8.0]])  # (y, x)
        spots = SpotData(coordinates=coords, dz=0.5, dx=0.2, is_3d=False)

        assert spots.has_spots is True

        assert spots.x is not None
        assert spots.x.tolist() == [4, 8]
        assert spots.y is not None
        assert spots.y.tolist() == [2, 6]
        assert spots.z is not None
        assert spots.z.tolist() == [0.0, 0.0]

        assert spots.x_um is not None
        assert spots.x_um.tolist() == pytest.approx([0.8, 1.6])
        assert spots.y_um is not None
        assert spots.y_um.tolist() == pytest.approx([0.4, 1.2])
        assert spots.z_um is None

    def test_3d_has_spots(self):
        coords = np.array([[1.0, 2.0, 3.0]])  # (z, y, x)
        spots = SpotData(coordinates=coords, dz=0.5, dx=0.2, is_3d=True)

        assert spots.has_spots is True

        assert spots.x is not None
        assert spots.x.tolist() == [3]
        assert spots.y is not None
        assert spots.y.tolist() == [2]
        assert spots.z is not None
        assert spots.z.tolist() == [1]

        assert spots.x_um is not None
        assert spots.x_um.tolist() == pytest.approx([0.6])
        assert spots.y_um is not None
        assert spots.y_um.tolist() == pytest.approx([0.4])
        assert spots.z_um is not None
        assert spots.z_um.tolist() == pytest.approx([0.5])

    def test_no_spots(self):
        coords = np.empty((0, 2))
        spots = SpotData(coordinates=coords, dz=0.5, dx=0.2, is_3d=False)

        assert spots.has_spots is False
        assert spots.x is None
        assert spots.y is None
        assert spots.z is None
        assert spots.x_um is None
        assert spots.y_um is None
        assert spots.z_um is None

    def test_rounding_uses_round_half_to_even(self):
        # np.round uses banker's rounding: 2.5 -> 2, 3.5 -> 4 (not always "round up")
        coords = np.array([[2.5, 3.5]])
        spots = SpotData(coordinates=coords, dz=0.5, dx=0.2, is_3d=False)

        assert spots.x is not None
        assert spots.x.tolist() == [4]
        assert spots.y is not None
        assert spots.y.tolist() == [2]


class TestImageData:
    def test_seg_squeeze_path_for_2d_input(self):
        seg = np.array([[0.0, 10.0], [20.0, 30.0]])
        img = ImageData(segmentation_image=seg, spot_image=np.zeros((2, 2)), masks=None)

        # brightest raw pixel (30) -> darkest output (0); darkest raw (0) -> brightest (1)
        assert img.seg_inv_norm[0, 0] == pytest.approx(1.0)
        assert img.seg_inv_norm[1, 1] == pytest.approx(0.0)

    def test_seg_stdev_projection_for_multiplane_input(self):
        seg = np.array(
            [[[0.0, 0.0], [0.0, 0.0]], [[2.0, 4.0], [6.0, 8.0]]]
        )  # shape (2, 2, 2); per-pixel std across the 2 planes -> [[1, 2], [3, 4]]
        img = ImageData(segmentation_image=seg, spot_image=np.zeros((2, 2)), masks=None)

        assert img.seg_inv_norm.shape == (2, 2)  # projected away the leading Z axis
        assert img.seg_inv_norm[0, 0] == pytest.approx(
            1.0
        )  # lowest std (1) -> brightest
        assert img.seg_inv_norm[1, 1] == pytest.approx(
            0.0
        )  # highest std (4) -> darkest

    def test_seg_squeeze_path_for_single_plane_3d_input(self):
        """
        shape[0] == 1 takes the squeeze branch, not the std branch - a real single-frame
        z-stack, not just a "2D image". If std were wrongly applied here, std of one
        sample per pixel is always 0 regardless of the data, which would NOT match img_2d.
        """
        seg_3d = np.array([[[0.0, 10.0], [20.0, 30.0]]])  # shape (1, 2, 2)
        seg_2d = seg_3d[0]
        img_3d = ImageData(
            segmentation_image=seg_3d, spot_image=np.zeros((2, 2)), masks=None
        )
        img_2d = ImageData(
            segmentation_image=seg_2d, spot_image=np.zeros((2, 2)), masks=None
        )

        assert img_3d.seg_inv_norm == pytest.approx(img_2d.seg_inv_norm)

    def test_spot_percentile_clip_uses_percentile_not_raw_max(self):
        # 199 background pixels at 5.0, one outlier at 500.0 (10x20 = 200 elements total)
        spots = np.full((10, 20), 5.0)
        spots[0, 0] = 500
        img = ImageData(
            segmentation_image=np.zeros((2, 2)), spot_image=spots, masks=None
        )

        # hand-computed: p_lo=5.0, p_hi=7.475 (linear-interpolated 99.5th percentile of
        # 199x5.0 + one 500.0) - the clip ceiling is nowhere near the raw outlier value
        assert img.spots_stdev_norm[1, 1] == pytest.approx(
            0.0
        )  # background, untouched by clip
        assert img.spots_stdev_norm[0, 0] == pytest.approx(
            1.0
        )  # outlier, clipped to the ceiling

    def test_masks_2d_projects_3d_masks_with_elementwise_max(self):
        masks_3d = np.array([[[1, 0], [0, 2]], [[0, 3], [4, 0]]])  # shape (2, 2, 2)
        img = ImageData(
            segmentation_image=np.zeros((2, 2)),
            spot_image=np.zeros((2, 2)),
            masks=masks_3d,
        )

        assert img.masks_2d is not None
        assert img.masks_2d.tolist() == [[1, 3], [4, 2]]

    def test_masks_2d_passes_through_2d_masks_unchanged(self):
        masks_2d = np.array([[1, 0], [0, 2]])
        img = ImageData(
            segmentation_image=np.zeros((2, 2)),
            spot_image=np.zeros((2, 2)),
            masks=masks_2d,
        )

        assert img.masks_2d is not None
        assert img.masks_2d.tolist() == masks_2d.tolist()

    def test_masks_2d_is_none_when_masks_is_none(self):
        img = ImageData(
            segmentation_image=np.zeros((2, 2)), spot_image=np.zeros((2, 2)), masks=None
        )

        assert img.masks_2d is None


# =====================================================================
# Panels
# =====================================================================


class TestPanelSegemntation:
    pass


class TestPanelSpotDetection:
    pass


class TestPanelFlow:
    pass


class TestPanelZDistribution:
    pass


class TestPanelECFD:
    pass


class TestPanelSpotmap:
    pass


# =====================================================================
# make_qc_figure
# =====================================================================


class TestMakeQCFigure:
    pass


# =====================================================================
# make_run_summary_figure
# =====================================================================


class TestMakeRunSummaryFigure:
    pass


# =====================================================================
# make_scene_summary_figure
# =====================================================================


class TestMakeSceneSummaryFigure:
    pass
