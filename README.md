<img alt="Vector logo" width="50%" src="https://raw.githubusercontent.com/scikit-hep/vector/main/docs/_images/LogoSrc.svg"/>

# Vector: arrays of 2D, 3D, and Lorentz vectors

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![pre-commit.ci status][pre-commit-badge]][pre-commit-link]
[![codecov percentage][codecov-badge]][codecov-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![Gitter][gitter-badge]][gitter-link]

[![PyPI platforms][pypi-platforms]][pypi-link]
[![PyPI version][pypi-version]][pypi-link]
[![Conda latest release][conda-version]][conda-link]
[![DOI][zenodo-badge]][zenodo-link]
[![LICENSE][license-badge]][license-link]
[![Scikit-HEP][sk-badge]][sk-link]

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

<img alt="Diagram of coordinate systems" width="100%" src="https://raw.githubusercontent.com/scikit-hep/vector/main/docs/_images/coordinate-systems.svg"/>

### Backends

Vectors may be included in any of these data types:

- [vector.obj](https://vector.readthedocs.io/en/latest/src/make_object.html) objects (pure Python)
- [NumPy structured arrays](https://numpy.org/doc/stable/user/basics.rec.html) of vectors
- [Awkward Arrays](https://awkward-array.org/) of vectors (possibly within variable-length lists or nested record structures)
- [SymPy expressions](https://www.sympy.org/en/index.html) for symbolic (non-numeric) manipulations
- In [Numba-compiled functions](https://numba.pydata.org/), with [vector.obj](https://vector.readthedocs.io/en/latest/src/make_object.html) objects or Awkward Arrays

Each of these "backends" provides the same suite of properties and methods, through a common "compute" library.

### Geometric versus momentum

Finally, vectors come in two flavors:

- geometric: only one name for each property or method
- momentum: same property or method can be accessed with several synonyms, such as `pt` ($p_T$, transverse momentum) for the azimuthal magnitude `rho` ($\rho$) and `energy` and `mass` for the Cartesian time `t` and proper time `tau` ($\tau$).

### Familiar conventions

Names and coordinate conventions were chosen to align with [ROOT](https://root.cern/)'s [TLorentzVector](https://root.cern.ch/doc/master/classTLorentzVector.html) and [Math::LorentzVector](https://root.cern.ch/doc/master/classROOT_1_1Math_1_1LorentzVector.html), as well as [scikit-hep/math](https://github.com/scikit-hep/scikit-hep/tree/master/skhep/math), [uproot-methods TLorentzVector](https://github.com/scikit-hep/uproot3-methods/blob/master/uproot3_methods/classes/TLorentzVector.py), [henryiii/hepvector](https://github.com/henryiii/hepvector), and [coffea.nanoevents.methods.vector](https://coffea-hep.readthedocs.io/en/latest/modules/coffea.nanoevents.methods.vector.html).

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

## Documentation

### Tutorials

- [Vector objects](https://vector.readthedocs.io/en/latest/src/object.html)
- [NumPy arrays of vectors](https://vector.readthedocs.io/en/latest/src/numpy.html)
- [Awkward Arrays of vectors](https://vector.readthedocs.io/en/latest/src/awkward.html)
- [Compiling functions on vectors with Numba](https://vector.readthedocs.io/en/latest/src/numba.html)
- [Vector expressions with SymPy](https://vector.readthedocs.io/en/latest/src/sympy.html)

### Vector constructors

- [Making vector objects](https://vector.readthedocs.io/en/latest/src/make_object.html)
- [Making NumPy arrays of vectors](https://vector.readthedocs.io/en/latest/src/make_numpy.html)
- [Making Awkward Arrays of vectors](https://vector.readthedocs.io/en/latest/src/make_awkward.html)
- [Making SymPy vector expressions](https://vector.readthedocs.io/en/latest/src/make_sympy.html)

### Vector functions

- [Interface for all vectors](https://vector.readthedocs.io/en/latest/src/common.html)
- [Interface for 2D vectors](https://vector.readthedocs.io/en/latest/src/vector2d.html)
- [Interface for 3D vectors](https://vector.readthedocs.io/en/latest/src/vector3d.html)
- [Interface for 4D vectors](https://vector.readthedocs.io/en/latest/src/vector4d.html)
- [Interface for 2D momentum](https://vector.readthedocs.io/en/latest/src/momentum2d.html)
- [Interface for 3D momentum](https://vector.readthedocs.io/en/latest/src/momentum3d.html)
- [Interface for 4D momentum](https://vector.readthedocs.io/en/latest/src/momentum4d.html)

### More ways to learn

- [Presentations about Vector](https://vector.readthedocs.io/en/latest/src/talks.html)

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/jpivarski"><img src="https://avatars.githubusercontent.com/u/1852447?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jim Pivarski</b></sub></a><br /><a href="#maintenance-jpivarski" title="Maintenance">🚧</a> <a href="https://github.com/scikit-hep/vector/commits?author=jpivarski" title="Code">💻</a> <a href="https://github.com/scikit-hep/vector/commits?author=jpivarski" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/henryiii"><img src="https://avatars.githubusercontent.com/u/4616906?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Henry Schreiner</b></sub></a><br /><a href="#maintenance-henryiii" title="Maintenance">🚧</a> <a href="https://github.com/scikit-hep/vector/commits?author=henryiii" title="Code">💻</a> <a href="https://github.com/scikit-hep/vector/commits?author=henryiii" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/eduardo-rodrigues"><img src="https://avatars.githubusercontent.com/u/5013581?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Eduardo Rodrigues</b></sub></a><br /><a href="#maintenance-eduardo-rodrigues" title="Maintenance">🚧</a> <a href="https://github.com/scikit-hep/vector/commits?author=eduardo-rodrigues" title="Code">💻</a> <a href="https://github.com/scikit-hep/vector/commits?author=eduardo-rodrigues" title="Documentation">📖</a></td>
    <td align="center"><a href="http://lovelybuggies.com.cn/"><img src="https://avatars.githubusercontent.com/u/29083689?v=4?s=100" width="100px;" alt=""/><br /><sub><b>N!no</b></sub></a><br /><a href="https://github.com/scikit-hep/vector/commits?author=LovelyBuggies" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/pfackeldey"><img src="https://avatars.githubusercontent.com/u/18463582?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Peter Fackeldey</b></sub></a><br /><a href="https://github.com/scikit-hep/vector/commits?author=pfackeldey" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/kreczko"><img src="https://avatars.githubusercontent.com/u/1213276?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Luke Kreczko</b></sub></a><br /><a href="https://github.com/scikit-hep/vector/commits?author=kreczko" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/nsmith-"><img src="https://avatars.githubusercontent.com/u/6587412?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nicholas Smith</b></sub></a><br /><a href="#ideas-nsmith-" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/mayou36"><img src="https://avatars.githubusercontent.com/u/17454848?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jonas Eschle</b></sub></a><br /><a href="#ideas-mayou36" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the
[all-contributors](https://github.com/all-contributors/all-contributors)
specification. Contributions of any kind welcome! See
[CONTRIBUTING.md](./.github/CONTRIBUTING.md) for information on setting up a
development environment.

## Acknowledgements

This library was primarily developed by Saransh Chopra, Henry Schreiner, Jim Pivarski, Eduardo Rodrigues, and Jonas Eschle.

Support for this work was provided by the National Science Foundation cooperative agreement [OAC-1836650](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1836650) and [PHY-2323298](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2323298) (IRIS-HEP) and [OAC-1450377](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1450377) (DIANA/HEP). Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

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
[zenodo-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.7054478.svg
[zenodo-link]: https://doi.org/10.5281/zenodo.7054478
