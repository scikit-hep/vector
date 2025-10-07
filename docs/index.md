![](_images/LogoSrc.svg)

# Overview

[![DOI][zenodo-badge]][zenodo-link]
[![DOI][joss-badge]][joss-link]
[![Scikit-HEP][sk-badge]][sk-link]

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![pre-commit.ci status][pre-commit-badge]][pre-commit-link]
[![codecov percentage][codecov-badge]][codecov-link]

[![PyPI platforms][pypi-platforms]][pypi-link]
[![PyPI version][pypi-version]][pypi-link]
[![Conda latest release][conda-version]][conda-link]

[![LICENSE][license-badge]][license-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![Gitter][gitter-badge]][gitter-link]

## Installation

You can install Vector with [pip](https://pypi.org/project/vector/) and [conda](https://anaconda.org/conda-forge/vector).

```bash
pip install vector
```

## Introduction

Vector is a Python library for 2D and 3D spatial vectors, as well as 4D space-time vectors. It is especially intended for performing geometric calculations on _arrays of vectors_, rather than one vector at a time in a Python for loop.

Vector is part of the [Scikit-HEP project](https://scikit-hep.org/), High Energy Physics (HEP) tools in Python.

### Coordinate systems

Vectors may be expressed in any of these coordinate systems:

- the azimuthal plane may be Cartesian `x` `y` or polar `rho` ($\rho$) `phi` ($\phi$)
- the longitudinal axis may be Cartesian `z`, polar `theta` ($\theta$), or pseudorapidity `eta` ($\eta$)
- the temporal component for space-time vectors may be Cartesian `t` or proper time `tau` ($\tau$)

in any combination. (That is, 4D vectors have 2×3×2 = 12 distinct coordinate systems.)

![](_images/coordinate-systems.svg)

### Backends

Vectors may be included in any of these data types:

- [vector.obj](src/make_object.md) objects (pure Python)
- [NumPy structured arrays](https://numpy.org/doc/stable/user/basics.rec.html) of vectors
- [Awkward Arrays](https://awkward-array.org/) of vectors (possibly within variable-length lists or nested record structures)
- [SymPy expressions](https://www.sympy.org/en/index.html) for symbolic (non-numeric) manipulations
- In [Numba-compiled functions](https://numba.pydata.org/), with [vector.obj](src/make_object.md) objects or Awkward Arrays

Each of these "backends" provides the same suite of properties and methods, through a common "compute" library.

### Integrations

Optionally, the vector package provides integration with other libraries. Currently, this includes:

- [PyTree integrations](https://vector.readthedocs.io/en/latest/src/pytree.html) using the [optree](https://github.com/metaopt/optree) package.

### Geometric versus momentum

Finally, vectors come in two flavors:

- geometric: only one name for each property or method
- momentum: same property or method can be accessed with several synonyms, such as `pt` ($p_T$, transverse momentum) for the azimuthal magnitude `rho` ($\rho$) and `energy` and `mass` for the Cartesian time `t` and proper time `tau` ($\tau$).

### Familiar conventions

Names and coordinate conventions were chosen to align with [ROOT](https://root.cern/)'s [TLorentzVector](https://root.cern.ch/doc/master/classTLorentzVector.html) and [Math::LorentzVector](https://root.cern.ch/doc/master/classROOT_1_1Math_1_1LorentzVector.html), as well as [scikit-hep/math](https://github.com/scikit-hep/scikit-hep/tree/master/skhep/math), [uproot-methods TLorentzVector](https://github.com/scikit-hep/uproot3-methods/blob/master/uproot3_methods/classes/TLorentzVector.py), [henryiii/hepvector](https://github.com/henryiii/hepvector), and [coffea.nanoevents.methods.vector](https://coffea-hep.readthedocs.io/en/latest/modules/coffea.nanoevents.methods.vector.html).

Vector follows the $(-, -, -, +)$ (`x, y, z, t`) metric convention for Lorentz vectors. The $(-,-,-,+)$ metric convention for Lorentz vectors corresponds to the Minkowski metric:

```{math}
g_{\mu\nu} = \text{diag}(-1, -1, -1, +1)
```

For a Lorentz vector $p^\mu = (p_x, p_y, p_z, E)$, the squared norm (or invariant mass squared) is given by:

```{math}
p^\mu p_\mu = g_{\mu\nu} p^\mu p^\nu = E^2 - p_x^2 - p_y^2 - p_z^2
```

This convention is widely used in high-energy physics (HEP), including frameworks such as ROOT.

Further, the transformations and rotations (including boosts) are active, and the Euler angle conventions align with the formalisations in the [GenVector package](https://root.cern/topical/GenVector.pdf). More precisely, the implementation of `rotate_euler` uses the matrices defined in the [wikipedia article](https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix).

## Getting help

- Source code on GitHub: [scikit-hep/vector](https://github.com/scikit-hep/vector)
- Report bugs and request features on the [GitHub Issues page](https://github.com/scikit-hep/vector/issues)
- Ask questions on the [GitHub Discussions page](https://github.com/scikit-hep/vector/discussions)
- Real-time chat on Gitter: [Scikit-HEP/Vector](https://gitter.im/Scikit-HEP/vector)

## Contributing to Vector

If you want to contribute to Vector, [pull requests](https://github.com/scikit-hep/vector/pulls) are welcome!

Please install the latest version of the `main` branch from source or a fork:

```bash
git clone https://github.com/scikit-hep/vector.git
cd vector
pip install -e .
```

Refer to [CONTRIBUTING.md](https://github.com/scikit-hep/vector/blob/main/.github/CONTRIBUTING.md) for more.

## Citing Vector

To cite Vector, please use [![DOI][joss-badge]][joss-link]

```bib
@article{Chopra2025,
doi = {10.21105/joss.07791},
url = {https://doi.org/10.21105/joss.07791},
year = {2025}, publisher = {The Open Journal},
volume = {10},
number = {109},
pages = {7791},
author = {Saransh Chopra and Henry Schreiner and Eduardo Rodrigues and Jonas Eschle and Jim Pivarski},
title = {Vector: JIT-compilable mathematical manipulations of ragged Lorentz vectors},
journal = {Journal of Open Source Software}
}
```

## Documentation

```{toctree}
:maxdepth: 1
:caption: Tutorials
src/object.ipynb
src/numpy.ipynb
src/awkward.ipynb
src/numba.ipynb
src/sympy.ipynb
src/pytree.ipynb
```

```{toctree}
:maxdepth: 1
:caption: Vector constructors
src/make_object.md
src/make_numpy.md
src/make_awkward.md
src/make_sympy.md
```

```{toctree}
:maxdepth: 1
:caption: Vector functions
src/common.md
src/vector2d.md
src/vector3d.md
src/vector4d.md
src/momentum2d.md
src/momentum3d.md
src/momentum4d.md
```

```{toctree}
:maxdepth: 1
:caption: Integrations
src/pytree_api.md
```

```{toctree}
:maxdepth: 1
:caption: More ways to learn
src/talks.md
```

[actions-badge]: https://github.com/scikit-hep/vector/actions/workflows/ci.yml/badge.svg
[actions-link]: https://github.com/scikit-hep/vector/actions
[codecov-badge]: https://codecov.io/gh/scikit-hep/vector/branch/main/graph/badge.svg?token=YBv60ueORQ
[codecov-link]: https://codecov.io/gh/scikit-hep/vector
[conda-version]: https://img.shields.io/conda/vn/conda-forge/vector.svg
[conda-link]: https://github.com/conda-forge/vector-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]: https://github.com/scikit-hep/vector/discussions
[gitter-badge]: https://badges.gitter.im/Scikit-HEP/vector.svg
[gitter-link]: https://gitter.im/Scikit-HEP/vector?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
[joss-badge]: https://joss.theoj.org/papers/10.21105/joss.07791/status.svg
[joss-link]: https://doi.org/10.21105/joss.07791
[license-badge]: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
[license-link]: https://opensource.org/licenses/BSD-3-Clause
[pre-commit-badge]: https://results.pre-commit.ci/badge/github/scikit-hep/vector/main.svg
[pre-commit-link]: https://results.pre-commit.ci/repo/github/scikit-hep/vector
[pypi-link]: https://pypi.org/project/vector/
[pypi-platforms]: https://img.shields.io/pypi/pyversions/vector
[pypi-version]: https://badge.fury.io/py/vector.svg
[rtd-badge]: https://readthedocs.org/projects/vector/badge/?version=latest
[rtd-link]: https://vector.readthedocs.io/en/latest/?badge=latest
[sk-badge]: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
[sk-link]: https://scikit-hep.org/
[zenodo-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.15263860.svg
[zenodo-link]: https://zenodo.org/records/15263860
