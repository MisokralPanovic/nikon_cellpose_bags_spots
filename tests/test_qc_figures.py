from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from spot_detector import qc_figures
from spot_detector.qc_figures import (
    make_qc_figure,
    make_run_summary_figure,
    make_scene_summary_figure,
)

# =====================================================================
# make_qc_figure
# =====================================================================


class TestMakeQCFigure:
    @pytest.fixture
    def qc_common(self, tmp_path, make_config):
        return dict(
            condition="control",
            scene=1,
            out_path=tmp_path / "qc.png",
            spot_labels=np.array([1, 1, 2, 2, 0, 0]),
            dx=0.1,
            dz=0.5,
            config=make_config(
                detection={"prob_thresh": 0.4, "use_default_model": True}
            ),
        )

    def test_smoke_2d(self, qc_common, caplog):
        rng = np.random.default_rng(0)
        masks = np.zeros((16, 16), dtype=int)
        masks[2:6, 2:6] = 1  # object 1
        masks[9:14, 9:14] = 2  # object 2

        kwargs = qc_common | dict(
            mode="2d",
            segmentation_image=rng.random((16, 16), dtype=np.float32),
            spots_image=rng.random((16, 16), dtype=np.float32),
            masks=masks,
            coordinates=np.array(
                [
                    [3.0, 3.0],
                    [4.5, 6.0],
                    [10.0, 9.0],
                    [13.0, 12.0],
                    [7.0, 1.5],
                    [1.0, 14.0],
                ]
            ),  # (y, x)
            flow_details=SimpleNamespace(
                flow=rng.random((16, 16, 3), dtype=np.float32),  # Spotiflow 2D layout
                prob=np.array([0.95, 0.82, 0.91, 0.78, 0.35, 0.42]),
            ),
        )

        make_qc_figure(**kwargs)

        assert kwargs["out_path"].exists()
        assert "failed" not in caplog.text.lower()

    def test_smoke_3d(self, qc_common, caplog):
        rng = np.random.default_rng(1)
        masks = np.zeros((5, 16, 16), dtype=int)
        masks[1:4, 2:6, 2:6] = 1  # object 1
        masks[1:4, 9:14, 9:14] = 2  # object 2

        kwargs = qc_common | dict(
            mode="3d",
            segmentation_image=rng.random((5, 16, 16), dtype=np.float32),
            spots_image=rng.random((5, 16, 16), dtype=np.float32),
            masks=masks,
            coordinates=np.array(
                [
                    [0.0, 3.0, 3.0],
                    [1.0, 4.5, 6.0],
                    [2.0, 10.0, 9.0],
                    [3.0, 13.0, 12.0],
                    [4.0, 7.0, 1.5],
                    [2.0, 1.0, 14.0],
                ]
            ),  # (z, y, x)
            flow_details=SimpleNamespace(
                flow=rng.random(
                    (5, 16, 16, 4), dtype=np.float32
                ),  # Spotiflow 3D layout
                prob=np.array([0.95, 0.82, 0.91, 0.78, 0.35, 0.42]),
            ),
        )

        make_qc_figure(**kwargs)

        assert kwargs["out_path"].exists()
        assert "failed" not in caplog.text.lower()

    def test_dispatches_to_all_six_panels(self, qc_common, mocker: MockerFixture):
        panels = mocker.patch.multiple(
            "spot_detector.qc_figures",
            _panel_segemntation=mocker.DEFAULT,
            _panel_spot_detection=mocker.DEFAULT,
            _panel_flow=mocker.DEFAULT,
            _panel_z_distribution=mocker.DEFAULT,
            _panel_ecdf=mocker.DEFAULT,
            _panel_spotmap=mocker.DEFAULT,
        )
        mocker.patch("spot_detector.qc_figures.plt.savefig")
        subplots_spy = mocker.spy(qc_figures.plt, "subplots")

        kwargs = qc_common | dict(
            mode="2d",
            segmentation_image=np.zeros((8, 8), dtype=np.float32),
            spots_image=np.zeros((8, 8), dtype=np.float32),
            masks=np.zeros((8, 8), dtype=int),
            coordinates=np.empty((0, 2)),
            flow_details=None,
        )

        make_qc_figure(**kwargs)

        for mock in panels.values():
            mock.assert_called_once()

        axes_flat = subplots_spy.spy_return[1].flatten()  # spy_return is (fig, axes)
        assert panels["_panel_segemntation"].call_args.kwargs["ax"] is axes_flat[0]
        assert panels["_panel_spotmap"].call_args.kwargs["ax"] is axes_flat[5]

    def test_closes_figure(self, qc_common, mocker: MockerFixture):
        kwargs = qc_common | dict(
            mode="2d",
            segmentation_image=np.zeros((8, 8), dtype=np.float32),
            spots_image=np.zeros((8, 8), dtype=np.float32),
            masks=np.zeros((8, 8), dtype=int),
            coordinates=np.empty((0, 2)),
            flow_details=None,
        )
        mocker.patch("spot_detector.qc_figures.plt.savefig")
        close_mock = mocker.patch("spot_detector.qc_figures.plt.close")

        make_qc_figure(**kwargs)

        close_mock.assert_called_once()


# =====================================================================
# make_scene_summary_figure
# =====================================================================


class TestMakeSceneSummaryFigure:
    def test_smoke_2d(self, tmp_path):
        df = pd.DataFrame(
            {
                "Scene": [1, 1, 2],
                "Spot_Count": [20, 30, 40],
                "Area_um2": [200, 250, 300],
            }
        )
        out_path = tmp_path / "scene.png"

        make_scene_summary_figure(
            df=df, condition="Control", mode="2d", out_path=out_path
        )

        assert out_path.exists()
        assert df["Scene"].tolist() == [1, 1, 2]

    def test_smoke_3d(self, tmp_path):
        df = pd.DataFrame(
            {
                "Scene": [1, 1, 2],
                "Spot_Count": [20, 30, 40],
                "Volume_um3": [200, 250, 300],
            }
        )
        out_path = tmp_path / "scene.png"

        make_scene_summary_figure(
            df=df, condition="Control", mode="3d", out_path=out_path
        )

        assert out_path.exists()
        assert df["Scene"].tolist() == [1, 1, 2]


# =====================================================================
# make_run_summary_figure
# =====================================================================


class TestMakeRunSummaryFigure:
    def test_smoke_2d(self, tmp_path):
        df = pd.DataFrame(
            {
                "Condition": [
                    "Control",
                    "Control",
                    "Treated",
                    "Treated",
                    "Negative",
                    "Negative",
                ],
                "Spot_Count": [18, 22, 90, 110, 4, 6],
                "Area_um2": [200, 210, 240, 260, 300, 310],
                "Spot_Density_per_um2": [0.09, 0.11, 0.38, 0.42, 0.013, 0.020],
            }
        )
        out_path = tmp_path / "run.png"

        make_run_summary_figure(df=df, experiment="Exp1", mode="2d", out_path=out_path)

        assert out_path.exists()

    def test_smoke_3d(self, tmp_path):
        df = pd.DataFrame(
            {
                "Condition": [
                    "Control",
                    "Control",
                    "Treated",
                    "Treated",
                    "Negative",
                    "Negative",
                ],
                "Spot_Count": [18, 22, 90, 110, 4, 6],
                "Volume_um3": [200, 210, 240, 260, 300, 310],
                "Spot_Density_per_um3": [0.09, 0.11, 0.38, 0.42, 0.013, 0.020],
            }
        )
        out_path = tmp_path / "run.png"

        make_run_summary_figure(df=df, experiment="Exp1", mode="3d", out_path=out_path)

        assert out_path.exists()

    def test_3d_selects_volume_columns(self, tmp_path, mocker: MockerFixture):
        df = pd.DataFrame(
            {
                "Condition": ["Control", "Control", "Treated", "Treated"],
                "Spot_Count": [18, 22, 90, 110],
                "Volume_um3": [200, 210, 240, 260],
                "Area_um2": [20, 21, 24, 26],
                "Spot_Density_per_um3": [0.09, 0.11, 0.38, 0.42],
                "Spot_Density_per_um2": [0.9, 1.1, 3.8, 4.2],
            }
        )
        scatterplot = mocker.patch("spot_detector.qc_figures.sns.scatterplot")
        mocker.patch("spot_detector.qc_figures.plt.savefig")

        make_run_summary_figure(
            df=df, experiment="Exp1", mode="3d", out_path=tmp_path / "run.png"
        )
        panel_d = [
            c for c in scatterplot.call_args_list if c.kwargs.get("y") == "Spot_Count"
        ]
        assert len(panel_d) == 1
        assert panel_d[0].kwargs["x"] == "Volume_um3"
