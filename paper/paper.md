---
title: "Vector: mathematical manipulations of JIT-compilable ragged Lorentz vectors"
tags:
  - Python
  - vector algebra
  - high energy physics
authors:
  - name: Saransh Chopra
    orcid: 0000-0003-3046-7675
    equal-contrib: true
    affiliation: "1, 2"
  - name: Henry Schreiner
    orcid: 0000-0002-7833-783X
    equal-contrib: true
    affiliation: 2
  - name: Jim Pivarski
    orcid: 0000-0002-6649-343X
    equal-contrib: true
    corresponding: true
    affiliation: 2

affiliations:
  - name: University College London
    index: 1
  - name: Princeton University
    index: 2
date: 12 October 2024
bibliography: paper.bib
---

# Summary

Mathematical manipulations of vectors is a crucial component of data analysis
pipelines in high energy physics, enabling physicists to transform raw data
into meaningful results that can be visualized. More specifically, high energy
physicists work with 2D and 3D Euclidean vectors, and 4D Lorentz vectors that
can be used as physical quantities, such as position, momentum, and forces.
Given that high energy physics data is not uniform, the vector manipulation
frameworks or libraries are expected to work readily on non-uniform or ragged
data, data with variable-sized rows (or a nested data structure with variable-sized
entries); thus, the library is expected to perform operations on an entire
ragged structure in minimum passes. Furthermore, optimizing memory usage and
processing time has become essential with the increasing computational demands
at the LHC. Vector is a Python library for creating and manipulating 2D, 3D,
and Lorentz vectors, especially arrays of vectors, to solve common physics
problems in a NumPy-like [@harris:2020] way. The library enables physicists to
operate on high energy physics data in a high level language without
compromising speed. The library is already in use at LHC and is a part of
frameworks, like Coffea [@Gray:2023], employed by physicists across multiple
high energy physics experiments.

# Statement of need

Vector is one of the few Lorentz vector libraries providing a Pythonic interface
but a compiled (through Awkward Array [@Pivarski:2018]) computational backend.
Vector integrates seamlessly with the existing high energy physics
ecosystem and the broader scientific Python ecosystem, including libraries like
Dask [@rocklin:2015] and Numba [@lam:2015]. The library implements a variety of
backends for several purposes. Although vector was written with high energy
physics in mind, it is a general-purpose library that can be used for any
scientific or engineering application. The library houses a set of diverse
backends, 3 numerical backends for experimental physicists and 1 symbolic
backend for theoretical physicists. These backends include:

- a pure Python object (builtin) backend for scalar computations,
- a NumPy backend for computations on regular collection-type data,
- a SymPy [@Meurer:2017] backend for symbolic computations, and
- an Awkward backend for computations on ragged collection-type data

There also exists implementations of the Object and the Awkward backend in Numba
for just-in-time compilable operations. Further, support for JAX and Dask is
provided through the Awkward backend, which enables vector functionalities to
support automatic differentiation and parallel computing.

## Impact

Besides PyROOT's LorentzVectors and TLorentzVector [@root:2020], vector has
become a popular choice for mathematical manipulations in Python based high energy
physics analysis pipelines. Along with being utilized directly in
analysis pipelines at LHC and other experiments [@Kling:2023; @Held:2024; @Qu:2022],
the library is being used as a dependency in user-facing frameworks, such as,
Coffea, MadMiner [@Brehmer:2020], FastJet [@aryan:2023], Spyral [@spyral-utils:2024],
Weaver [@weaver-core:2024], and pylhe [@pylhe]. The library is also used in multiple
teaching materials for graduate courses and workshops. Finally, given the generic
nature of the library, it is often used in non high energy physics use cases.

# Acknowledgements

The work on vector was supported by NSF cooperative agreements OAC-1836650
(IRIS-HEP) and PHY-2323298 (IRIS-HEP). We would additionally like to thank the
contributors of vector and the Scikit-HEP community for their support.

# Reference
