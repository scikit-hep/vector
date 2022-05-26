.. Vector documentation master file, created by
   sphinx-quickstart on Thu Mar 12 15:04:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

.. image:: _images/vector-logo.png

|Action status| |Documentation Status| |pre-commit.ci status| |GitHub Discussion| |Gitter| |Code style: black|
|PyPI platforms| |PyPI version| |Conda latest releasetatus| |DOI| |Scikit-HEP|

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
Below are some examples covering the majority of the features offered by ``vector``.

* `Introduction to Vector <usage/intro.ipynb>`_
* `Structur of Vector <usage/structure.ipynb>`_
* `Vector Design Prototype <usage/vector_design_prototype.ipynb>`_

.. toctree::
   :maxdepth: 1
   :caption: Example gallery
   :hidden:

   usage/intro
   usage/structure
   usage/vector_design_prototype

Developer guide
-----------------
If you are planning to develop ``vector``, or if you want to use the latest commit of ``vector`` on your local machine,
you might want to install it from the source. The developer guide is available here -

* `Developer guide <dev_guide.rst>`_

.. toctree::
   :maxdepth: 4
   :caption: Development of vector
   :hidden:

   dev_guide.rst

.. toctree::
   :maxdepth: 1
   :caption: Changelog

   changelog

.. toctree::
   :maxdepth: 5
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
.. |Conda latest releasetatus| image:: https://img.shields.io/conda/vn/conda-forge/decaylanguage.svg
   :target: https://github.com/conda-forge/decaylanguage-feedstock
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5942083.svg
   :target: https://doi.org/10.5281/zenodo.5942082
.. |Scikit-HEP| image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :target: https://scikit-hep.org/
