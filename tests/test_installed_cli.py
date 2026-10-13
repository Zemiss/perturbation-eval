"""An installed checkout must work without the original parent workspace."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["evaluate", "evaluate.evaluate_gears"])
def test_installed_cli_runs_from_an_unrelated_directory(tmp_path, module):
    process = subprocess.run([sys.executable, "-m", module, "--help"], cwd=tmp_path,
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "--help" in process.stdout
