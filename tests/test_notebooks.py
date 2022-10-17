from __future__ import annotations

import sys
from pathlib import Path

import papermill as pm
import pytest


@pytest.fixture()
def common_kwargs(tmpdir):
    outputnb = tmpdir.join("output.ipynb")
    return {
        "output_path": str(outputnb),
        "kernel_name": f"python{sys.version_info.major}",
        "progress_bar": False,
    }


def test_intro(common_kwargs):
    execution_dir = Path.cwd() / "docs" / "usage"
    pm.execute_notebook(execution_dir / "intro.ipynb", **common_kwargs)
