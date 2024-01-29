# Contributing to vector

If you are planning to develop `vector`, or want to use the latest commit of `vector` on your local machine,
you might want to install it from the source. This installation is not recommended for users who want to use
the stable version of `vector`. The steps below describe the installation process of `vector`'s latest commit. This page also describes how to test `vector`'s codebase and build `vector`'s documentation.

**Note**: [Scikit-HEP's developer information](https://scikit-hep.org/developer) is a general and much more detailed collection of documentation available for developing `Scikit-HEP` packages. This developer guide is specific to `vector`.

## Installing vector

We recommend using a virtual environment to install `vector`. This would isolate the library from your global `Python` environment, which would be beneficial for reproducing bugs, and the overall development of `vector`. The first step would be to clone `vector` -

```bash
git clone https://github.com/Scikit-hep/vector.git
```

and then we can change the current working directory and enter `vector` -

```bash
cd vector
```

### Creating a virtual environment

A virtual environment can be set up and activated using `venv` in both `UNIX` and `Windows` systems.

**UNIX**:

```bash
python3 -m venv .env
. .env/bin/activate
```

**Windows**:

```
python -m venv .env
.env\bin\activate
```

### Installation

The developer installation of `vector` comes with several options -

- `awkward`: installs [awkward](https://github.com/scikit-hep/awkward) along with `vector`
- `test`: the test dependencies
- `test-extras`: extra dependencies to run tests on a specific Python version and Operating System
- `docs`: extra dependencies to build and develop `vector`'s documentation
- `dev`: installs the `awkward` option + the `test` option + [numba](https://github.com/numba/numba)

These options can be used with `pip` with the editable (`-e`) mode of installation in the following way -

```bash
pip install -e .[dev,test]
```

For example, if you want to install the `docs` dependencies along with the dependencies included above, use -

```bash
pip install -e .[dev,test,docs]
```

Furthermore, `vector` can also be installed using `conda`. This installation also requires using a virtual environment -

```bash
conda env create
conda activate vector
conda config --env --add channels conda-forge  # Optional
```

### Adding vector for notebooks

`vector` can be added to the notebooks using the following commands -

```
python -m ipykernel install --user --name vector
```

## Activating pre-commit

`vector` uses a set of `pre-commit` hooks and the `pre-commit` bot to format, type-check, and prettify the codebase. The hooks can be installed locally using -

```bash
pre-commit install
```

This would run the checks every time a commit is created locally. The checks will only run on the files modified by that commit, but the checks can be triggered for all the files using -

```bash
pre-commit run --all-files
```

If you would like to skip the failing checks and push the code for further discussion, use the `--no-verify` option with `git commit`.

## Testing vector

`vector` is tested using `pytest` and `pytest-doctestplus`. `pytest` is responsible for testing the code, whose configuration is available in [pyproject.toml](https://github.com/scikit-hep/vector/blob/main/pyproject.toml).`pytest-doctestplus` is responsible for testing the examples available in every docstring, which prevents them from going stale. Additionally, `vector` also uses `pytest-cov` to calculate the coverage of these unit tests.

### Running tests locally

The tests can be executed using the `test` and `test-extras` dependencies of `vector` in the following way -

```bash
python -m pytest
```

To skip the notebook tests, use `--ignore=tests/test_notebooks.py`

### Running tests with coverage locally

The coverage value can be obtained while running the tests using `pytest-cov` in the following way -

```bash
python -m pytest --cov=vector tests/
```

### Running doctests

The doctests can be executed using the `test` dependencies of `vector` in the following way -

```bash
python -m pytest --doctest-plus src/vector/
```

or, one can run the doctests along with the unit tests in the following way -

```bash
python -m pytest --doctest-plus .
```

A much more detailed guide on testing with `pytest` for `Scikit-HEP` packages is available [here](https://scikit-hep.org/developer/pytest).

### Running notebook tests

The Notebook tests can be executed individually in the following way -

```bash
pytest tests/test_notebooks.py
```

## Documenting vector

`vector`'s documentation is mainly written in the form of [docstrings](https://peps.python.org/pep-0257) and [reStructurredText](https://docutils.sourceforge.io/docs/user/rst/quickref.html). The docstrings include the description, arguments, examples, return values, and attributes of a class or a function, and the `.rst` files enable us to render this documentation on `vector`'s documentation website.

`vector` primarily uses [Sphinx](https://www.sphinx-doc.org/en/master/) for rendering documentation on its website. The configuration file (`conf.py`) for `sphinx` can be found [here](https://github.com/scikit-hep/vector/blob/main/docs/conf.py). The documentation is deployed on [https://readthedocs.io]() [here](https://vector.readthedocs.io/en/latest/).

Ideally, with the addition of every new feature to `vector`, documentation should be added using comments, docstrings, and `.rst` files.

### Building documentation locally

The documentation is located in the `docs` folder of the main repository. This documentation can be generated using
the `docs` dependencies of `vector` in the following way -

```bash
cd docs/
make clean
make html
```

The commands executed above will clean any existing documentation build and create a new build under the `docs/_build`
folder. You can view this build in any browser by opening the `index.html` file.

## Nox

`vector` supports running various critical commands using [nox](https://github.com/wntrblm/nox) to make them less intimidating for new developers. All of these commands (or sessions in the language of `nox`) - `lint`, `tests`, `notebooks`, `doctests`, `docs`, and `build` - are defined in [noxfile.py](https://github.com/scikit-hep/vector/blob/main/noxfile.py).

`nox` can be installed via `pip` using -

```bash
pip install nox
```

The default sessions (`lint`, `tests`, and `doctests`) can be executed using -

```bash
nox
```

### Running pre-commit with nox

The `pre-commit` hooks can be run with `nox` in the following way -

```
nox -s lint
```

### Running tests with nox

Tests can be run with `nox` in the following way -

```
nox -s tests
```

Notebooks can be tested with `nox` in the following way -

```
nox -s notebooks
```

### Building documentation with nox

Docs can be built with `nox` in the following way -

```
nox -s docs
```

Use the following command if you want to deploy the docs on `localhost` -

```
nox -s docs -- serve
```
