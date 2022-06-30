"""
Use ``nox`` to run various test suites of vector locally.

Execute ``nox`` to run the default sessions ("lint", "tests", "doctests")
and ``nox --session <session>`` to run a specific session.
"""
import shutil
from pathlib import Path

import nox

ALL_PYTHONS = ["3.6", "3.7", "3.8", "3.9"]

nox.options.sessions = ["lint", "tests", "doctests"]


DIR = Path(__file__).parent.resolve()


@nox.session(reuse_venv=True)
def lint(session):
    """Run the linter."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(python=ALL_PYTHONS, reuse_venv=True)
def tests(session):
    """Run the unit and regular tests."""
    session.install("-e", ".[awkward,test,test-extras]")
    session.run("pytest", *session.posargs)


@nox.session(reuse_venv=True)
def doctests(session):
    """Run the doctests."""
    session.install("-e", ".[awkward,test,test-extras]")
    session.run("xdoctest", "./src/vector/", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session):
    """Build the docs. Pass "serve" to serve."""
    session.install("-e", ".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if session.posargs:
        if "serve" in session.posargs:
            print(  # noqa: T201
                "Launching docs at http://localhost:8001/ - use Ctrl-C to quit"
            )
            session.run("python", "-m", "http.server", "8001", "-d", "_build/html")
        else:
            print("Unsupported argument to docs")  # noqa: T201


@nox.session
def build(session):
    """Build an SDist and wheel."""
    build_p = DIR.joinpath("build")
    if build_p.exists():
        shutil.rmtree(build_p)

    session.install("build")
    session.run("python", "-m", "build")
