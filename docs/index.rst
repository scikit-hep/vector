.. Vector documentation master file, created by
   sphinx-quickstart on Thu Mar 12 15:04:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

.. image:: _images/vector-logo.png

|Action status| |Documentation Status| |pre-commit.ci status| |coverage| |GitHub Discussion| |Gitter| |Code style: black|
|PyPI platforms| |PyPI version| |Conda latest releasetatus| |DOI| |License| |Scikit-HEP|

Overview
--------

Vector is a Python 3.6+ library for 2D, 3D, and `Lorentz vectors <https://en.wikipedia.org/wiki/Special_relativity#Physics_in_spacetime>`_, especially _arrays of vectors\_, to solve common physics problems in a NumPy-like way.

Main features of Vector:

* Pure Python with NumPy as its only dependency. This makes it easier to install.
* Vectors may be represented in a variety of coordinate systems: Cartesian, cylindrical, pseudorapidity, and any combination of these with time or proper time for Lorentz vectors. In all, there are 12 coordinate systems: {*x* - *y* vs *ρ* - *φ* in the azimuthal plane} × {*z* vs *θ* vs *η* longitudinally} × {*t* vs *τ* temporally}.
* Uses names and conventions set by `ROOT <https://root.cern/>`_'s `TLorentzVector <https://root.cern.ch/doc/master/classTLorentzVector.html>`_ and `Math::LorentzVector <https://root.cern.ch/doc/master/classROOT_1_1Math_1_1LorentzVector.html>`_, as well as `scikit-hep/math <https://github.com/scikit-hep/scikit-hep/tree/master/skhep/math>`_, `uproot-methods TLorentzVector <https://github.com/scikit-hep/uproot3-methods/blob/master/uproot3_methods/classes/TLorentzVector.py>`_, `henryiii/hepvector <https://github.com/henryiii/hepvector>`_, and `coffea.nanoevents.methods.vector <https://coffeateam.github.io/coffea/modules/coffea.nanoevents.methods.vector.html>`_.
* Implemented on a variety of backends:
   * pure Python objects
   * NumPy arrays of vectors (as a `structured array <https://numpy.org/doc/stable/user/basics.rec.html>`_ subclass)
   * `Awkward Arrays <https://awkward-array.org/>`_ of vectors
   * potential for more: CuPy, TensorFlow, Torch, JAX...
* NumPy/Awkward backends also implemented in `Numba <https://numba.pydata.org/>`_ for JIT-compiled calculations on vectors.
* Distinction between geometrical vectors, which have a minimum of attribute and method names, and vectors representing momentum, which have synonyms like ``pt`` = ``rho``, ``energy`` = ``t``, ``mass`` = ``tau``.

Installation
------------
Vector is available on `PyPI <https://pypi.org/project/vector/>`_ as well as on `conda <https://anaconda.org/conda-forge/vector>`_. The library can be installed using ``pip`` -

.. code-block::

   pip install vector

or using ``conda`` -

.. code-block::

   conda install -c conda-forge vector

Example gallery
---------------
``Vector`` has several examples covering the basics as well as some advanced usage of the library. The example gallery
covers almost all the features offered by ``vector`` and any new additions to the gallery are welcomed.

**Note**: Adding more examples and improving the existing examples for newcomers is still in progress.

.. toctree::
   :maxdepth: 1
   :caption: Example gallery

   usage/intro
   usage/structure
   usage/vector_design_prototype

Changes in vector's API
-----------------------
The ``changelog`` file describes the changes in ``vector``'s API and usage introduced in every new version. These changes can
be breaking changes or minor adjustments, hence one should go through this file and their existing codebase while updating ``vector``'s
version.

.. toctree::
   :maxdepth: 3
   :caption: Changes in vector's API

   changelog

Getting help
------------
* ``Vector``'s code is hosted on `GitHub <https://github.com/scikit-hep/vector>`_.
* If something is not working the way it should, or if you want to request a new feature, create a new `issue <https://github.com/scikit-hep/vector/issues>`_ on GitHub.
* To discuss something related to ``vector``, use the `discussions <https://github.com/scikit-hep/vector/discussions/>`_ tab on GitHub or ``vector``'s gitter (`Scikit-HEP/vector <https://gitter.im/Scikit-HEP/vector>`_) chat room.

Developing vector
-----------------
If you are planning to develop ``vector`` (thank you!), or if you want to use the latest commit of ``vector`` on your local machine,
you might want to install it from the source. The developer guide introduces a user to the various tasks a developer might perform,
including installing from source, testing the library, documenting the library, etc. For a general developer guide for all the
Scikit-HEP projects, refer to the `Scikit-HEP website <https://scikit-hep.org/developer>`_.

**Note**: The developer guide is still in progress.

API reference
-------------
The API reference details the functionality of each ``class`` and ``function`` present in ``vector``'s codebase.

.. toctree::
   :maxdepth: 8
   :caption: API Reference

   api/modules.rst

.. |Action status| image:: https://github.com/scikit-hep/vector/workflows/CI/badge.svg
   :target: https://github.com/scikit-hep/vector/actions
.. |Documentation Status| image:: https://readthedocs.org/projects/vector/badge/?version=latest
   :target: https://vector.readthedocs.io/en/latest/?badge=latest
.. |pre-commit.ci status| image:: https://results.pre-commit.ci/badge/github/scikit-hep/vector/develop.svg
   :target: https://results.pre-commit.ci/repo/github/scikit-hep/vector
.. |GitHub Discussion| image:: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
   :target: https://github.com/scikit-hep/vector/discussions
.. |Gitter| image:: https://badges.gitter.im/Scikit-HEP/vector.svg
   :target: https://gitter.im/Scikit-HEP/vector?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |PyPI platforms| image:: https://img.shields.io/pypi/pyversions/vector
   :target: https://pypi.org/project/vector/
.. |PyPI version| image:: https://badge.fury.io/py/vector.svg
   :target: https://pypi.org/project/vector/
.. |Conda latest releasetatus| image:: https://img.shields.io/conda/vn/conda-forge/vector.svg
   :target: https://github.com/conda-forge/vector-feedstock
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5942083.svg
   :target: https://doi.org/10.5281/zenodo.5942082
.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |Scikit-HEP| image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :target: https://scikit-hep.org/
.. |coverage| image:: https://codecov.io/gh/scikit-hep/vector/branch/main/graph/badge.svg?token=YBv60ueORQ
   :target: https://codecov.io/gh/scikit-hep/vector
