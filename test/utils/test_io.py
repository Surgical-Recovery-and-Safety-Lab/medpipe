#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O functions and classes tests suite.
"""

from pathlib import Path

import pytest

from medpipe.utils.io import read_toml_configuration


class TestReadTOMLConfiguration:

    @pytest.fixture
    def example_config_dir(self) -> Path:
        """Provide the location of the example configuration files."""
        base_dir = Path(__file__).parent.parent.parent

        return base_dir / "config-examples/"

    def test_read_configuration(self, example_config_dir: Path) -> None:
        """Test successfull function call."""
        read_toml_configuration(example_config_dir / "HGBc_config.toml")
