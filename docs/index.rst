.. Vector documentation master file, created by
   sphinx-quickstart on Thu Mar 12 15:04:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

Vector: vectorized 2D, 3D, and Lorentz vectors
==============================================

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
* Distinction between geometrical vectors, which have a minimum of attribute and method names, and vectors representing momentum, which have synonyms like `pt` = `rho`, `energy` = `t`, `mass` = `tau`.


.. toctree::
   :maxdepth: 1
   :caption: Using vector

   usage/intro
   usage/structure
   usage/vector_design_prototype
   changelog

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/modules.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
