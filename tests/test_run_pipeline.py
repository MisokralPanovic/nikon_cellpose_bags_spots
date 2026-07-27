import pytest
from pytest_mock import MockerFixture
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

from spot_detector.run_pipeline import run_pipeline, _process_file, _process_scene

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def base_config():
    return {
        "channels": {"segmentation_image": 0, "spot_image": 1},
        "segmentation": {"bin_factor": 4, "stitch_threshold": 0.4},
        "detection": {"prob_thresh": 0.5, "min_distance": 10},
    }


@pytest.fixture
def mock_img_2d(mocker: MockerFixture):
    """Fake BioImage with 2D dims, returning small synthetic stacks."""
    img = mocker.MagicMock()
    img.dims.order = "CYX"
    img.get_image_data.return_value = np.random.rand(20, 20).astype(np.float32)
    img.physical_pixel_sizes.X = 0.5
    img.physical_pixel_sizes.Z = 1.0
    return img


@pytest.fixture
def mock_models(mocker: MockerFixture):
    return mocker.MagicMock()


# =====================================================================
# _process_scene
# =====================================================================


class TestProcessScene:
    def test_2d_calls_segment_2d(
        self, mocker: MockerFixture, base_config, mock_img_2d, mock_models, tmp_path
    ):
        fake_masks = np.zeros((20, 20), dtype=int)
        fake_masks[5:10, 5:10] = 1
        mock_segment_2d = mocker.patch(
            "spot_detector.run_pipeline.segment_2d", return_value=fake_masks
        )
        mocker.patch(
            "spot_detector.run_pipeline.detect_spots_spotiflow",
            return_value=(
                np.array([[7, 7]]),
                SimpleNamespace(flow=np.zeros((5, 5, 3))),
            ),
        )
        mocker.patch("spot_detector.run_pipeline.make_qc_figure")

        result = _process_scene(
            img=mock_img_2d,
            config=base_config,
            do_3d=False,
            models=mock_models,
            mode="2d",
            condition="Control",
            filepath=Path("test.nd2"),
            experiment="exp1",
            scene=0,
            fig_dir=tmp_path,
        )

        mock_segment_2d.assert_called_once()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_3d_calls_segment_3d_with_stitch_threshold(
        self, mocker: MockerFixture, base_config, mock_models, tmp_path
    ):
        img = mocker.MagicMock()
        img.dims.order = "CZYX"
        img.get_image_data.return_value = np.random.rand(5, 20, 20).astype(np.float32)
        img.physical_pixel_sizes.X = 0.5
        img.physical_pixel_sizes.Z = 2.0

        fake_masks = np.zeros((5, 20, 20), dtype=int)
        fake_masks[:, 5:10, 5:10] = 1
        mock_segment_3d = mocker.patch(
            "spot_detector.run_pipeline.segment_3d", return_value=fake_masks
        )
        mocker.patch(
            "spot_detector.run_pipeline.detect_spots_spotiflow",
            return_value=(
                np.array([[2, 7, 7]]),
                SimpleNamespace(flow=np.zeros((5, 5, 4))),
            ),
        )
        mocker.patch("spot_detector.run_pipeline.make_qc_figure")

        result = _process_scene(
            img=img,
            config=base_config,
            do_3d=True,
            models=mock_models,
            mode="3d",
            condition="Treated",
            filepath=Path("test.nd2"),
            experiment="exp1",
            scene=2,
            fig_dir=tmp_path,
        )

        mock_segment_3d.assert_called_once()
        assert mock_segment_3d.call_args[1]["stitch_threshold"] == 0.4
        assert result["Scene"].iloc[0] == 2

    def test_qc_figure_called_with_correct_path_and_condition(
        self, mocker: MockerFixture, base_config, mock_img_2d, mock_models, tmp_path
    ):
        mocker.patch(
            "spot_detector.run_pipeline.segment_2d",
            return_value=np.zeros((20, 20), dtype=int),
        )
        mocker.patch(
            "spot_detector.run_pipeline.detect_spots_spotiflow",
            return_value=(
                np.array([[7, 7]]),
                SimpleNamespace(flow=np.zeros((5, 5, 3))),
            ),
        )
        mock_qc = mocker.patch("spot_detector.run_pipeline.make_qc_figure")

        _process_scene(
            img=mock_img_2d,
            config=base_config,
            do_3d=False,
            models=mock_models,
            mode="2d",
            condition="Control",
            filepath=Path("test.nd2"),
            experiment="exp1",
            scene=3,
            fig_dir=tmp_path,
        )

        mock_qc.assert_called_once()
        kwargs = mock_qc.call_args[1]
        assert kwargs["out_path"] == tmp_path / "Control_S03_2D_qc.png"
        assert kwargs["condition"] == "Control"

    def test_passes_filepath_name_to_measure_objects(
        self, mocker: MockerFixture, base_config, mock_img_2d, mock_models, tmp_path
    ):
        mocker.patch(
            "spot_detector.run_pipeline.segment_2d",
            return_value=np.zeros((20, 20), dtype=int),
        )
        mocker.patch(
            "spot_detector.run_pipeline.detect_spots_spotiflow",
            return_value=(
                np.array([[7, 7]]),
                SimpleNamespace(flow=np.zeros((5, 5, 3))),
            ),
        )
        mocker.patch("spot_detector.run_pipeline.make_qc_figure")
        mock_measure = mocker.patch(
            "spot_detector.run_pipeline.measure_objects",
            return_value=pd.DataFrame({"Object_Label": []}),
        )

        _process_scene(
            img=mock_img_2d,
            config=base_config,
            do_3d=False,
            models=mock_models,
            mode="2d",
            condition="Control",
            filepath=Path("/some/dir/Control_01.nd2"),
            experiment="exp1",
            scene=3,
            fig_dir=tmp_path,
        )

        kwargs = mock_measure.call_args[1]
        assert kwargs["filepath"] == "Control_01.nd2"  # .name, not full path


# =====================================================================
# _process_file
# =====================================================================


class TestProcessFile:
    def test_combines_multiple_scenes(
        self, mocker: MockerFixture, base_config, mock_models, tmp_path
    ):
        mock_img_instance = mocker.MagicMock()
        mock_img_instance.scenes = ["scene0", "scene1"]
        mocker.patch(
            "spot_detector.run_pipeline.BioImage", return_value=mock_img_instance
        )

        df_scene0 = pd.DataFrame({"Object_Label": [1], "Scene": [0]})
        df_scene1 = pd.DataFrame({"Object_Label": [1], "Scene": [1]})
        mock_process_scene = mocker.patch(
            "spot_detector.run_pipeline._process_scene",
            side_effect=[df_scene0, df_scene1],
        )
        mock_summary = mocker.patch(
            "spot_detector.run_pipeline.make_scene_summary_figure"
        )

        result = _process_file(
            filepath=Path("Control_01.nd2"),
            config=base_config,
            models=mock_models,
            mode="2d",
            do_3d=False,
            experiment="exp1",
            fig_dir=tmp_path,
            tab_dir=tmp_path,
            failures=[],
        )

        assert mock_process_scene.call_count == 2
        assert result is not None
        assert len(result) == 2
        mock_summary.assert_called_once()
        assert (tmp_path / "Control_objects_2D.csv").exists()

    def test_handles_corrupted_scene(
        self, mocker: MockerFixture, base_config, mock_models, tmp_path
    ):
        mock_img_instance = mocker.MagicMock()
        mock_img_instance.scenes = ["scene0", "scene1", "scene2"]
        mocker.patch(
            "spot_detector.run_pipeline.BioImage", return_value=mock_img_instance
        )

        df_scene0 = pd.DataFrame({"Object_Label": [1], "Scene": [0]})
        df_scene2 = pd.DataFrame({"Object_Label": [1], "Scene": [2]})
        mock_process_scene = mocker.patch(
            "spot_detector.run_pipeline._process_scene",
            side_effect=[df_scene0, Exception("boom"), df_scene2],
        )

        mocker.patch("spot_detector.run_pipeline.make_scene_summary_figure")

        failures = []

        result = _process_file(
            filepath=Path("Control_01.nd2"),
            config=base_config,
            models=mock_models,
            mode="2d",
            do_3d=False,
            experiment="exp1",
            fig_dir=tmp_path,
            tab_dir=tmp_path,
            failures=failures,
        )

        assert mock_process_scene.call_count == 3
        assert result is not None
        assert len(result) == 2
        assert failures[0]["Scene"] == 1

    def test_returns_none_when_no_scenes(
        self, mocker: MockerFixture, base_config, mock_models, tmp_path
    ):
        mock_img_instance = mocker.MagicMock()
        mock_img_instance.scenes = []
        mocker.patch(
            "spot_detector.run_pipeline.BioImage", return_value=mock_img_instance
        )

        result = _process_file(
            filepath=Path("Control_01.nd2"),
            config=base_config,
            models=mock_models,
            mode="2d",
            do_3d=False,
            experiment="exp1",
            fig_dir=tmp_path,
            tab_dir=tmp_path,
            failures=[],
        )

        assert result is None

    def test_derives_condition_from_filename(
        self, mocker: MockerFixture, base_config, mock_models, tmp_path
    ):
        mock_img_instance = mocker.MagicMock()
        mock_img_instance.scenes = ["scene0"]
        mocker.patch(
            "spot_detector.run_pipeline.BioImage", return_value=mock_img_instance
        )
        mocker.patch(
            "spot_detector.run_pipeline._process_scene",
            return_value=pd.DataFrame({"Object_Label": [1]}),
        )
        mocker.patch("spot_detector.run_pipeline.make_scene_summary_figure")

        _process_file(
            filepath=Path("Treated-DrugA_FOV3.nd2"),
            config=base_config,
            models=mock_models,
            mode="2d",
            do_3d=False,
            experiment="exp1",
            fig_dir=tmp_path,
            tab_dir=tmp_path,
            failures=[],
        )

        assert (tmp_path / "Treated-DrugA_objects_2D.csv").exists()


# =====================================================================
# run_pipeline (top-level orchestration)
# =====================================================================


class TestRunPipeline:
    def test_combines_multiple_files_and_derives_experiment_name(
        self, mocker: MockerFixture, tmp_path
    ):
        # layout: tmp_path / my_experiment / data / *.nd2
        project_root = tmp_path / "my_experiment"
        data_dir = project_root / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "Control_01.nd2").touch()
        (data_dir / "Treated_01.nd2").touch()

        out_dir = project_root / "output"
        config = {
            "mode": {"do_3d": False},
            "paths": {"raw_data_dir": str(data_dir), "out_dir": str(out_dir)},
        }

        mocker.patch(
            "spot_detector.run_pipeline.ModelBundle.load",
            return_value=mocker.MagicMock(),
        )

        df_a = pd.DataFrame({"Object_Label": [1], "Condition": ["Control"]})
        df_b = pd.DataFrame({"Object_Label": [1], "Condition": ["Treated"]})
        mocker.patch(
            "spot_detector.run_pipeline._process_file", side_effect=[df_a, df_b]
        )
        mock_run_summary = mocker.patch(
            "spot_detector.run_pipeline.make_run_summary_figure"
        )

        result = run_pipeline(config=config)

        assert result is not None
        assert len(result) == 2
        mock_run_summary.assert_called_once()

        # experiment name should be derived from data_folder.parent.name == "my_experiment"
        assert mock_run_summary.call_args[1]["experiment"] == "my_experiment"
        assert (out_dir / "tables" / "_run_objects_2D.csv").exists()

    def test_handles_corrupted_file(self, mocker: MockerFixture, tmp_path):
        # layout: tmp_path / my_experiment / data / *.nd2
        project_root = tmp_path / "my_experiment"
        data_dir = project_root / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "Control_01.nd2").touch()
        (data_dir / "Extra_01.nd2").touch()
        (data_dir / "Treated_01.nd2").touch()

        out_dir = project_root / "output"
        config = {
            "mode": {"do_3d": False},
            "paths": {"raw_data_dir": str(data_dir), "out_dir": str(out_dir)},
        }

        mocker.patch(
            "spot_detector.run_pipeline.ModelBundle.load",
            return_value=mocker.MagicMock(),
        )

        df_a = pd.DataFrame({"Object_Label": [1], "Condition": ["Control"]})
        df_b = pd.DataFrame({"Object_Label": [1], "Condition": ["Treated"]})
        mock_process_file = mocker.patch(
            "spot_detector.run_pipeline._process_file",
            side_effect=[df_a, Exception("boom"), df_b],
        )
        mocker.patch("spot_detector.run_pipeline.make_run_summary_figure")

        result = run_pipeline(config=config)

        assert mock_process_file.call_count == 3
        assert result is not None
        assert len(result) == 2

        assert (out_dir / "tables" / "_run_failures_2D.csv").exists()
        failures_df = pd.read_csv(out_dir / "tables" / "_run_failures_2D.csv")

        assert len(failures_df) == 1
        assert set(failures_df.columns) == {
            "Experiment",
            "Source_File",
            "Condition",
            "Scene",
            "Error",
            "Error_Type",
        }
        assert failures_df["Error"].iloc[0] == "boom"
        assert failures_df["Error_Type"].iloc[0] == "Exception"

    def test_returns_none_with_no_files_processed(
        self, mocker: MockerFixture, tmp_path
    ):
        project_root = tmp_path / "empty_experiment"
        data_dir = project_root / "data"
        data_dir.mkdir(parents=True)  # empty folder

        config = {
            "mode": {"do_3d": False},
            "paths": {
                "raw_data_dir": str(data_dir),
                "out_dir": str(project_root / "output"),
            },
        }
        mocker.patch(
            "spot_detector.run_pipeline.ModelBundle.load",
            return_value=mocker.MagicMock(),
        )

        result = run_pipeline(config=config)

        assert result is None

    def test_creates_output_directories(self, mocker: MockerFixture, tmp_path):
        project_root = tmp_path / "exp"
        data_dir = project_root / "data"
        data_dir.mkdir(parents=True)

        out_dir = project_root / "output"
        config = {
            "mode": {"do_3d": False},
            "paths": {"raw_data_dir": str(data_dir), "out_dir": str(out_dir)},
        }
        mocker.patch(
            "spot_detector.run_pipeline.ModelBundle.load",
            return_value=mocker.MagicMock(),
        )

        run_pipeline(config=config)

        assert (out_dir / "figures").is_dir()
        assert (out_dir / "tables").is_dir()
