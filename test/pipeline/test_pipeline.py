#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline class tests suite.
"""

from pathlib import Path

import pytest

from medpipe.pipeline.pipeline import MedpipePipeline


class TestPipline:
    """Test class for the Pipline class"""

    @pytest.fixture
    def example_config_dir(self) -> Path:
        """Provide the location of the example configuration files."""
        base_dir = Path(__file__).parent.parent.parent

        return base_dir / "config-examples/"

    def test_create_pipeline(self, example_config_dir: Path) -> None:
        """Test successful pipeline creation."""
        pipe = MedpipePipeline(example_config_dir / "HGBc_config.toml", logger=None)
        assert pipe.version == "v0.1.1"
        assert pipe.predictor_algo == "HistGradientBoostingClassifier"
        assert pipe.calibrator_method == "IsotonicRegression"
        assert pipe.n_outcomes == 1
