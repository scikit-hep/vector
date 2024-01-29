from __future__ import annotations

from pathlib import Path

import nox

ALL_PYTHONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]

nox.options.sessions = ["lint", "tests", "doctests"]


DIR = Path(__file__).parent.resolve()


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run the linter."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def pylint(session: nox.Session) -> None:
    """Run pylint."""
    session.install("pylint~=2.14.0")
    session.install("-e", ".")
    session.run("pylint", "src/vector/", *session.posargs)


@nox.session(python=ALL_PYTHONS, reuse_venv=True)
def tests(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-e", ".[awkward,test,test-extras]")
    session.run("pytest", "--ignore", "tests/test_notebooks.py", *session.posargs)


@nox.session(reuse_venv=True)
def coverage(session: nox.Session) -> None:
    """Run tests and compute coverage."""
    session.posargs.append("--cov=vector")
    tests(session)


@nox.session(reuse_venv=True)
def doctests(session: nox.Session) -> None:
    """Run the doctests."""
    session.install("-e", ".[awkward,test,test-extras]")
    session.run("pytest", "--doctest-plus", "src/vector/", *session.posargs)


@nox.session(python=ALL_PYTHONS, reuse_venv=True)
def notebooks(session: nox.Session) -> None:
    """Run the notebook tests"""
    session.install("-e", ".[awkward,test,test-extras]", "numba")
    session.install("jupyter", "papermill")
    session.run("pytest", "tests/test_notebooks.py", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the docs. Pass "serve" to serve."""
    session.install("-e", ".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8001/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8001", "-d", "_build/html")
        else:
            print("Unsupported argument to docs")


@nox.session
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    session.install("build")
    session.run("python", "-m", "build")
