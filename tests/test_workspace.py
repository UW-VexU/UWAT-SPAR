"""Every workspace package imports, and `spar` mounts each package's subcommands."""

import importlib

import pytest
from typer.testing import CliRunner

from spar_cli.main import app

PACKAGES = [
    "spar_schema",
    "spar_ingest",
    "spar_perception",
    "spar_state",
    "spar_data",
    "spar_gt",
    "spar_sim",
    "spar_learn",
    "spar_eval",
    "spar_cli",
]
SUBCOMMANDS = ["ingest", "perception", "state", "data", "gt", "sim", "learn", "eval"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    importlib.import_module(name)


def test_cli_mounts_every_package():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for name in SUBCOMMANDS:
        assert name in result.output
