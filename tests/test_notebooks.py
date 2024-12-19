from __future__ import annotations

import sys
from pathlib import Path

import papermill as pm
import pytest


@pytest.fixture
def common_kwargs(tmpdir):
    outputnb = tmpdir.join("output.ipynb")
    return {
        "output_path": str(outputnb),
        "kernel_name": f"python{sys.version_info.major}",
        "progress_bar": False,
    }


def test_object(common_kwargs):
    execution_dir = Path.cwd() / "docs" / "src"
    pm.execute_notebook(execution_dir / "object.ipynb", **common_kwargs)


def test_numpy(common_kwargs):
    execution_dir = Path.cwd() / "docs" / "src"
    pm.execute_notebook(execution_dir / "numpy.ipynb", **common_kwargs)


def test_awkward(common_kwargs):
    execution_dir = Path.cwd() / "docs" / "src"
    pm.execute_notebook(execution_dir / "awkward.ipynb", **common_kwargs)


def test_numba(common_kwargs):
    execution_dir = Path.cwd() / "docs" / "src"
    pm.execute_notebook(execution_dir / "numba.ipynb", **common_kwargs)


def test_sympy(common_kwargs):
    execution_dir = Path.cwd() / "docs" / "src"
    pm.execute_notebook(execution_dir / "sympy.ipynb", **common_kwargs)
