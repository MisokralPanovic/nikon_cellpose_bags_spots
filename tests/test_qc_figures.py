"""
from types import SimpleNamespace

import numpy as np
import pytest
from pytest_mock import MockerFixture

from spot_detector.exceptions import FatalPipelineError
from spot_detector.qc_figures import (
    make_qc_figure,
    make_run_summary_figure,
    make_scene_summary_figure,
)
from spot_detector.qc_panels import (
    ImageData,
    SpotData,
    _flow_to_rgb,
    _panel_ecdf,
    _panel_flow,
    _panel_segemntation,
    _panel_spot_detection,
    _panel_spotmap,
    _panel_z_distribution,
)
"""
# =====================================================================
# Fixtures
# =====================================================================


# =====================================================================
# make_qc_figure
# =====================================================================


class TestMakeQCFigure:
    def test_smoke_2d(self):
        pass

    def test_smoke_3d(self):
        pass

    def test_dispatches_to_all_six_panels(self):
        pass

    def test_closes_figure(self):
        pass


# =====================================================================
# make_run_summary_figure
# =====================================================================


class TestMakeRunSummaryFigure:
    def test_smoke_2d(self):
        pass

    def test_smoke_3d(self):
        pass

    def test_3d_selects_voolume_columns(self):
        pass


# =====================================================================
# make_scene_summary_figure
# =====================================================================


class TestMakeSceneSummaryFigure:
    def test_smoke_2d(self):
        pass

    def test_smoke_3d(self):
        pass
