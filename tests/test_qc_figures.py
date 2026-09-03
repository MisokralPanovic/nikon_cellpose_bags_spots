from types import SimpleNamespace

import numpy as np
import pytest
from pytest_mock import MockerFixture

from spot_detector.exceptions import FatalPipelineError
from spot_detector.qc_figures import (
    ImageData,
    SpotData,
    _flow_to_rgb,
    _panel_flow,
    _panel_segemntation,
    _panel_spot_detection,
    _panel_z_distribution,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def ax(mocker: MockerFixture):
    return mocker.MagicMock()


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
    def test_draws_seg_image_only_when_no_masks(self, ax):
        images = SimpleNamespace(seg_inv_norm=np.zeros((2, 2)), masks_2d=None)

        _panel_segemntation(ax=ax, images=images, dx=0.2)  # type: ignore[arg-type]

        assert ax.imshow.call_count == 1
        _, kwargs = ax.imshow.call_args
        assert kwargs["cmap"] == "gray"

    def test_draw_mask_overlay_when_masks_present(self, ax):
        masks_2d = np.array([[0, 1], [2, 0]])
        images = SimpleNamespace(seg_inv_norm=np.zeros((2, 2)), masks_2d=masks_2d)

        _panel_segemntation(ax=ax, images=images, dx=0.2)  # type: ignore[arg-type]

        assert ax.imshow.call_count == 2
        _, overlay_kwargs = ax.imshow.call_args_list[1]
        assert overlay_kwargs["alpha"] == 0.3
        assert overlay_kwargs["cmap"] == "tab10"
        assert overlay_kwargs["vmax"] == 2

    def test_skips_mask_overlay_when_masks_present_but_all_zero(self, ax):
        # masks_2d is not None, but no object survived (all background) - the code
        # checks masks_2d.max() > 0, not just "is not None"
        images = SimpleNamespace(
            seg_inv_norm=np.zeros((2, 2)), masks_2d=np.zeros((2, 2))
        )

        _panel_segemntation(ax=ax, images=images, dx=0.2)  # type: ignore[arg-type]

        assert ax.imshow.call_count == 1

    def test_always_add_scalebar_title_and_axis_off(self, ax):
        images = SimpleNamespace(seg_inv_norm=np.zeros((2, 2)), masks_2d=None)

        _panel_segemntation(ax=ax, images=images, dx=0.2)  # type: ignore[arg-type]

        ax.add_artist.assert_called_once()
        ax.set_title.assert_called_once_with("StDev Projection + Masks")
        ax.axis.assert_called_once_with("off")

    def test_render_failure_falls_back_to_placeholder(self, ax):
        ax.imshow.side_effect = RuntimeError("boom")
        images = SimpleNamespace(seg_inv_norm=np.zeros((2, 2)), masks_2d=None)

        _panel_segemntation(ax=ax, images=images, dx=0.2)  # type: ignore[arg-type]

        ax.text.assert_called_once()
        assert ax.text.call_args[0][2] == "Failed to render"
        # code after the try/except is unconditional - still runs after a caught failure
        ax.add_artist.assert_called_once()

    def test_fatal_pipeline_error_propagates_uncaught(self, ax):
        ax.imshow.side_effect = FatalPipelineError("unrecoverable")
        images = SimpleNamespace(seg_inv_norm=np.zeros((2, 2)), masks_2d=None)

        with pytest.raises(FatalPipelineError):
            _panel_segemntation(ax=ax, images=images, dx=0.2)  # type: ignore[arg-type]

        # early exit: the scalebar/title/axis code after the try/except never runs
        ax.add_artist.assert_not_called()


class TestPanelSpotDetection:
    def test_no_spots_only_draws_background_image(self, ax):
        images = SimpleNamespace(spots_stdev_norm=np.zeros((2, 2)))
        spots = SimpleNamespace(has_spots=False)
        spot_labels = np.zeros((2, 2))

        _panel_spot_detection(
            ax=ax,
            images=images,  # type: ignore[arg-type]
            spots=spots,  # type: ignore[arg-type]
            spot_labels=spot_labels,
        )

        assert ax.imshow.call_count == 1
        ax.scatter.assert_not_called()

    def test_defensive_guard_skips_scatter_when_x_or_y_is_none(self, ax):
        images = SimpleNamespace(spots_stdev_norm=np.zeros((2, 2)))
        spots = SimpleNamespace(has_spots=True, x=None, y=None)
        spot_labels = np.zeros((2, 2))

        _panel_spot_detection(
            ax=ax,
            images=images,  # type: ignore[arg-type]
            spots=spots,  # type: ignore[arg-type]
            spot_labels=spot_labels,
        )

        ax.scatter.assert_not_called()

    def test_splits_spots_by_inside_vs_background(self, ax):
        images = SimpleNamespace(spots_stdev_norm=np.zeros((2, 2)))
        spots = SimpleNamespace(
            has_spots=True, x=np.array([1, 2, 3, 4]), y=np.array([10, 20, 30, 40])
        )
        spot_labels = np.array([0, 1, 0, 2])  # spots 0,2 background; 1,3 assigned

        _panel_spot_detection(
            ax=ax,
            images=images,  # type: ignore[arg-type]
            spots=spots,  # type: ignore[arg-type]
            spot_labels=spot_labels,
        )

        assert ax.scatter.call_count == 2

        bg_args, bg_kwargs = ax.scatter.call_args_list[0]
        assert bg_args[0].tolist() == [1, 3]
        assert bg_args[1].tolist() == [10, 30]
        assert bg_kwargs["color"] == "gray"
        assert bg_kwargs["alpha"] == 0.5

        in_args, in_kwargs = ax.scatter.call_args_list[1]
        assert in_args[0].tolist() == [2, 4]
        assert in_args[1].tolist() == [20, 40]
        assert in_kwargs["c"].tolist() == [1, 2]
        assert in_kwargs["cmap"] == "tab10"

    def test_always_add_title_and_axis_off(self, ax):
        images = SimpleNamespace(spots_stdev_norm=np.zeros((2, 2)))
        spots = SimpleNamespace(has_spots=False)
        spot_labels = np.zeros((2, 2))

        _panel_spot_detection(
            ax=ax,
            images=images,  # type: ignore[arg-type]
            spots=spots,  # type: ignore[arg-type]
            spot_labels=spot_labels,
        )

        ax.set_title.assert_called_once_with("Spot Detections (StDev Proj)")
        ax.axis.assert_called_once_with("off")

    def test_render_failure_falls_back_to_placeholder(self, ax):
        ax.imshow.side_effect = RuntimeError("boom")
        images = SimpleNamespace(spots_stdev_norm=np.zeros((2, 2)))
        spots = SimpleNamespace(has_spots=True, x=np.zeros((2, 2)), y=np.zeros((2, 2)))
        spot_labels = np.zeros((2, 2))

        _panel_spot_detection(
            ax=ax,
            images=images,  # type: ignore[arg-type]
            spots=spots,  # type: ignore[arg-type]
            spot_labels=spot_labels,
        )

        ax.text.assert_called_once()
        assert ax.text.call_args[0][2] == "Failed to render"
        # code after the try/except is unconditional - still runs after a caught failure
        ax.set_title.assert_called_once()

    def test_fatal_pipeline_error_propagates_uncaught(self, ax):
        ax.imshow.side_effect = FatalPipelineError("unrecoverable")
        images = SimpleNamespace(spots_stdev_norm=np.zeros((2, 2)))
        spots = SimpleNamespace(has_spots=True, x=np.zeros((2, 2)), y=np.zeros((2, 2)))
        spot_labels = np.zeros((2, 2))

        with pytest.raises(FatalPipelineError):
            _panel_spot_detection(
                ax=ax,
                images=images,  # type: ignore[arg-type]
                spots=spots,  # type: ignore[arg-type]
                spot_labels=spot_labels,
            )

        # early exit: the title/axis code after the try/except never runs
        ax.set_title.assert_not_called()


class TestFlowToRGB:
    def test_2d_output_contract(self):
        rng = np.random.default_rng(0)
        flow_data = rng.standard_normal((6, 7, 3))

        out = _flow_to_rgb(flow_data=flow_data)

        assert out.shape == (6, 7, 3)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_3d_output_collapses_z_axis(self):
        rng = np.random.default_rng(0)
        flow_data = rng.standard_normal((9, 6, 7, 4))

        out = _flow_to_rgb(flow_data=flow_data)

        assert out.shape == (6, 7, 3)
        assert out.dtype == np.float32

    def test_3d_selects_highest_magnitude_z(self):
        vol = np.zeros((9, 6, 7, 4))
        vol[1, :, :, 2] = 3.0  # fy on the "loud" slice
        vol[1, :, :, 3] = 4.0  # fx on the "loud" slice
        # slice 0 stays all-zero → mean XY magnitude 0 → not selected

        equiv_2d = np.zeros((6, 7, 4))
        equiv_2d[:, :, 1] = 3.0  # fy for the 2D branch (index 1, not 2)
        equiv_2d[:, :, 2] = 4.0  # fx for the 2D branch (index 2, not 3)

        assert _flow_to_rgb(flow_data=vol) == pytest.approx(
            _flow_to_rgb(flow_data=equiv_2d)
        )

    def test_3d_ignores_stereographic_and_z_channels(self):
        vol = np.zeros((9, 6, 7, 4))
        vol[0, :, :, 0] = 100.0  # stereographic component — must be ignored
        vol[0, :, :, 1] = 100.0  # z-flow component — must be ignored
        # indices 2 (fy) and 3 (fx) stay 0.0

        out = _flow_to_rgb(flow_data=vol)

        assert out == pytest.approx(1.0)

    def test_zero_flow_is_white(self):
        flow_data = np.zeros((4, 5, 3))

        out = _flow_to_rgb(flow_data=flow_data)

        assert out == pytest.approx(1.0)

    @pytest.mark.parametrize("shape", [(4, 5), (4, 5, 2), (2, 4, 5, 2)])
    def test_bad_shape_raises_valueerror(self, shape):
        with pytest.raises(ValueError):
            _flow_to_rgb(flow_data=np.zeros(shape))


class TestPanelFlow:
    def test_none_details_draws_placeholder(self, ax):
        flow_details = None

        _panel_flow(ax=ax, flow_details=flow_details)  # type: ignore[arg-type]

        ax.text.assert_called_once()
        assert ax.text.call_args[0][2] == "No Flow Data Provided"
        ax.imshow.assert_not_called()
        ax.set_title.assert_called_with("Stereographic Flow")

    def test_flow_none_heatmap_2d_shows_heatmap(self, ax):
        flow_details = SimpleNamespace(
            flow=None, heatmap=np.arange(12).reshape(3, 4).astype(float)
        )

        _panel_flow(ax=ax, flow_details=flow_details)  # type: ignore[arg-type]

        ax.imshow.assert_called_once()
        args, kwargs = ax.imshow.call_args
        assert kwargs["cmap"] == "magma"
        assert args[0] is flow_details.heatmap
        ax.set_title.assert_called_with(
            "Probability Heatmap\n(flow unavailable — subpix disabled)"
        )

    def test_flow_none_heatmap_3d_max_projects(self, ax):
        heatmap = np.array(
            [
                [[9.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 9.0]],
                [[0.0, 9.0], [0.0, 0.0]],
            ]
        )  # (3, 2, 2)  ->  max(axis=0) == [[9, 9], [0, 9]]
        flow_details = SimpleNamespace(flow=None, heatmap=heatmap)

        _panel_flow(ax=ax, flow_details=flow_details)  # type: ignore[arg-type]

        ax.imshow.assert_called_once()
        args, kwargs = ax.imshow.call_args
        assert args[0].shape == (2, 2)
        assert args[0] == pytest.approx(heatmap.max(axis=0))
        assert kwargs["cmap"] == "magma"
        ax.set_title.assert_called_with(
            "Probability Heatmap\n(flow unavailable — subpix disabled)"
        )

    def test_flow_none_no_heatmap_text_fallback(self, ax):
        flow_details = SimpleNamespace(flow=None, heatmap=None)

        _panel_flow(ax=ax, flow_details=flow_details)  # type: ignore[arg-type]

        ax.text.assert_called_once()
        assert (
            ax.text.call_args[0][2]
            == "Flow unavailable\n(model has compute_flow=False)"
        )
        ax.imshow.assert_not_called()

    def test_non_ndarray_flow_text_fallback(self, ax):
        flow_details = SimpleNamespace(flow=[1, 2, 3], heatmap=None)

        _panel_flow(ax=ax, flow_details=flow_details)  # type: ignore[arg-type]

        ax.text.assert_called_once()
        assert (
            ax.text.call_args[0][2]
            == "Flow Render Error\nflow attribute is not a numpy array"
        )
        ax.imshow.assert_not_called()

    def test_success_path_calls_flow_to_rgb_and_imshow(self, mocker: MockerFixture, ax):
        flow_details = SimpleNamespace(flow=np.zeros((4, 5, 3)), heatmap=None)

        mock_flow = mocker.patch(
            "spot_detector.qc_figures._flow_to_rgb", return_value=mocker.sentinel.rgb
        )
        _panel_flow(ax=ax, flow_details=flow_details)  # type: ignore[arg-type]

        mock_flow.assert_called_once()
        assert mock_flow.call_args[0][0] is flow_details.flow
        ax.imshow.assert_called_once_with(mocker.sentinel.rgb)
        ax.text.assert_not_called()
        ax.set_title.assert_called_with("Stereographic Flow")

    def test_render_failure_falls_back_to_placeholder(self, mocker: MockerFixture, ax):
        ax.imshow.side_effect = RuntimeError("boom")
        flow_details = SimpleNamespace(flow=np.zeros((4, 5, 3)), heatmap=None)

        mocker.patch(
            "spot_detector.qc_figures._flow_to_rgb", return_value=mocker.sentinel.rgb
        )
        _panel_flow(ax=ax, flow_details=flow_details)  # type: ignore[arg-type]

        ax.text.assert_called_once()
        assert ax.text.call_args[0][2] == "Flow Render Error\nboom"

    def test_fatal_pipeline_error_propagates_uncaught(self, mocker: MockerFixture, ax):
        ax.imshow.side_effect = FatalPipelineError("unrecoverable")
        flow_details = SimpleNamespace(flow=np.zeros((4, 5, 3)), heatmap=None)

        mocker.patch(
            "spot_detector.qc_figures._flow_to_rgb", return_value=mocker.sentinel.rgb
        )
        with pytest.raises(FatalPipelineError):
            _panel_flow(ax=ax, flow_details=flow_details)  # type: ignore[arg-type]


class TestPanelZDistribution:
    # spots.has_spots = False
    def test_no_spots_draws_placeholder(self, ax):
        spots = SimpleNamespace(
            has_spots=False,
        )

        _panel_z_distribution(
            ax=ax,
            images=...,  # type: ignore[arg-type]
            spots=spots,  # type: ignore[arg-type]
            spot_labels=...,  # type: ignore[arg-type]
        )

        ax.text.assert_called_once()
        assert ax.text.call_args[0][2] == "No Spots Detected"
        ax.axis.assert_called_once_with("off")
        ax.hist.assert_not_called()

    class TestIs3dTrue:
        # spots.has_spots = True, spots.is_3d = True

        @pytest.fixture
        def valid_3d(self):
            # 3 spots: 1 background, 1 in obj-1, 1 in obj-2
            return dict(
                images=SimpleNamespace(
                    spot_image=np.zeros((5, 2, 2)),
                    masks_2d=np.array([[0, 1], [2, 0]]),
                ),
                spots=SimpleNamespace(
                    has_spots=True,
                    is_3d=True,
                    dz=0.5,
                    z_um=np.array([0.5, 1.0, 1.5]),
                ),
                spot_labels=np.array([0, 1, 2]),
            )

        def test_stacked_histogram_per_object(self, ax, valid_3d):
            _panel_z_distribution(ax=ax, **valid_3d)

            ax.hist.assert_called_once()
            args, kwargs = ax.hist.call_args
            assert [a.tolist() for a in args[0]] == [[0.5], [1.0], [1.5]]
            assert kwargs["stacked"] is True
            assert kwargs["label"] == ["Background", "Obj 1", "Obj 2"]
            assert kwargs["color"][0] == "lightgray"
            assert kwargs["bins"] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
            ax.set_title.assert_called_once_with("Z-Distribution Profile (µm)")

        def test_legend_shown_when_few_objects(self, ax, valid_3d):
            valid_3d["images"].masks_2d = np.arange(9).reshape(3, 3)  # n_obj = 8
            _panel_z_distribution(ax=ax, **valid_3d)

            ax.legend.assert_called_once()
            ax.text.assert_not_called()

        def test_legend_suppressed_when_many_objects(self, ax, valid_3d):
            valid_3d["images"].masks_2d = np.arange(12).reshape(
                3, 4
            )  # n_obj = 11, not < 10
            _panel_z_distribution(ax=ax, **valid_3d)

            ax.legend.assert_not_called()
            ax.text.assert_not_called()

        def test_render_failure_falls_back_to_placeholder(self, ax, valid_3d):
            ax.hist.side_effect = RuntimeError("boom")
            _panel_z_distribution(ax=ax, **valid_3d)

            ax.text.assert_called_once()
            assert ax.text.call_args[0][2] == "Failed to render"
            ax.set_title.assert_called_once()

        def test_fatal_pipeline_error_propagates_uncaught(self, ax, valid_3d):
            ax.hist.side_effect = FatalPipelineError("unrecoverable")

            with pytest.raises(FatalPipelineError):
                _panel_z_distribution(ax=ax, **valid_3d)

            # early exit: the title/axis code after the try/except never runs
            ax.set_title.assert_not_called()

    class TestIs3dFalse:
        # spots.has_spots = True, spots.is_3d = False
        def test_insufficient_spots_for_nnd(self, ax):
            spots = SimpleNamespace(
                has_spots=True, is_3d=False, coordinates=np.array([[1.0, 2.0]]), dx=1.0
            )

            _panel_z_distribution(
                ax=ax,
                images=...,  # type: ignore[arg-type]
                spots=spots,  # type: ignore[arg-type]
                spot_labels=...,  # type: ignore[arg-type]
            )

            ax.text.assert_called_once()
            assert ax.text.call_args[0][2] == "Insufficient Spots for NND"
            ax.grid.assert_any_call(False)
            ax.hist.assert_not_called()

        def test_nnd_histogram_and_kde_curve(self, ax):
            spots = SimpleNamespace(
                has_spots=True,
                is_3d=False,
                coordinates=np.array(
                    [[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [20.0, 0.0]]
                ),  # NND [1,1,4,15] has variance
                dx=1.0,
            )

            _panel_z_distribution(
                ax=ax,
                images=...,  # type: ignore[arg-type]
                spots=spots,  # type: ignore[arg-type]
                spot_labels=...,  # type: ignore[arg-type]
            )

            ax.hist.assert_called_once()
            _, kwargs = ax.hist.call_args
            assert kwargs["density"] is True
            ax.plot.assert_called_once()

        def test_kde_failure_drops_curve_keeps_histogram(self, ax):
            spots = SimpleNamespace(
                has_spots=True,
                is_3d=False,
                coordinates=np.array(
                    [[0.0, 0.0], [1.0, 0.0]]
                ),  # NND [1,1] → gaussian_kde raises on singular covariance
                dx=1.0,
            )

            _panel_z_distribution(
                ax=ax,
                images=...,  # type: ignore[arg-type]
                spots=spots,  # type: ignore[arg-type]
                spot_labels=...,  # type: ignore[arg-type]
            )

            ax.hist.assert_called_once()
            ax.plot.assert_not_called()

        def test_render_failure_falls_back_to_placeholder(self, ax):
            ax.hist.side_effect = RuntimeError("boom")
            spots = SimpleNamespace(
                has_spots=True,
                is_3d=False,
                coordinates=np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [20.0, 0.0]]),
                dx=1.0,
            )

            _panel_z_distribution(
                ax=ax,
                images=...,  # type: ignore[arg-type]
                spots=spots,  # type: ignore[arg-type]
                spot_labels=...,  # type: ignore[arg-type]
            )

            ax.text.assert_called_once()
            assert ax.text.call_args[0][2] == "Failed to render"
            ax.set_title.assert_called_once()

        def test_fatal_pipeline_error_propagates_uncaught(self, ax):
            ax.hist.side_effect = FatalPipelineError("unrecoverable")
            spots = SimpleNamespace(
                has_spots=True,
                is_3d=False,
                coordinates=np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [20.0, 0.0]]),
                dx=1.0,
            )

            with pytest.raises(FatalPipelineError):
                _panel_z_distribution(
                    ax=ax,
                    images=...,  # type: ignore[arg-type]
                    spots=spots,  # type: ignore[arg-type]
                    spot_labels=...,  # type: ignore[arg-type]
                )

            # early exit: the title/axis code after the try/except never runs
            ax.set_title.assert_not_called()


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
