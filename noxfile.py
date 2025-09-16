#!/usr/bin/env -S uv run

# /// script
# dependencies = ["nox>=2025.2.9"]
# ///

from __future__ import annotations

from pathlib import Path

import nox

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|virtualenv"

DIR = Path(__file__).parent.resolve()
PYPROJECT = nox.project.load_toml(DIR / "pyproject.toml")
ALL_PYTHON = nox.project.python_versions(PYPROJECT)


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run the linter."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(reuse_venv=True, default=False)
def pylint(session: nox.Session) -> None:
    """Run pylint."""
    session.install("pylint")
    session.install("-e.")
    session.run("pylint", "src/vector/", *session.posargs)


@nox.session(reuse_venv=True, python=ALL_PYTHON)
def lite(session: nox.Session) -> None:
    """Run lightweight tests."""
    test_deps = nox.project.dependency_groups(PYPROJECT, "test")
    session.install("-e.", *test_deps)
    session.run("pytest", "--ignore", "tests/test_notebooks.py", *session.posargs)


@nox.session(reuse_venv=True, python=ALL_PYTHON)
def tests(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    test_deps = nox.project.dependency_groups(PYPROJECT, "test-all")
    session.install("-e.", *test_deps)
    session.run(
        "pytest",
        "--ignore",
        "tests/test_notebooks.py",
        *session.posargs,
    )


@nox.session(reuse_venv=True, python=ALL_PYTHON, default=False)
def coverage(session: nox.Session) -> None:
    """Run tests and compute coverage."""
    session.posargs.extend(["--cov=vector", "--cov-report=xml"])
    tests(session)


@nox.session(reuse_venv=True, python=ALL_PYTHON)
def doctests(session: nox.Session) -> None:
    """Run the doctests."""
    test_deps = nox.project.dependency_groups(PYPROJECT, "test-all")
    session.install("-e.", *test_deps)
    session.run("pytest", "--doctest-plus", "src/vector/", *session.posargs)


@nox.session(reuse_venv=True, default=False)
def notebooks(session: nox.Session) -> None:
    """Run the notebook tests"""
    test_deps = nox.project.dependency_groups(PYPROJECT, "test", "test-optional")
    session.install("-e.", *test_deps)
    session.install("jupyter")
    session.run("pytest", "tests/test_notebooks.py", *session.posargs)


@nox.session(reuse_venv=True, default=False)
def docs(session: nox.Session) -> None:
    """Build the docs. Pass "serve" to serve."""
    doc_deps = nox.project.dependency_groups(PYPROJECT, "docs", "test")
    session.install("-e.", doc_deps)
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8001/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8001", "-d", "_build/html")
        else:
            print("Unsupported argument to docs")


@nox.session(reuse_venv=True, default=False)
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    session.install("build")
    session.run("python", "-m", "build")


if __name__ == "__main__":
    nox.main()
