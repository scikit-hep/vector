---
title: "Vector: JIT-compilable mathematical manipulations of ragged Lorentz vectors"
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
  - name: Eduardo Rodrigues
    orcid: 0000-0003-2846-7625
    affiliation: 3
  - name: Jonas Eschle
    orcid: 0000-0002-7312-3699
    affiliation: 4
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
  - name: University of Liverpool
    index: 3
  - name: Syracuse University
    index: 4
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
at the the Large Hadron Collider (LHC), world’s largest particle accelerator.
Vector is a Python library for creating and manipulating 2D, 3D,
and Lorentz vectors, especially arrays of vectors, to solve common physics
problems in a NumPy-like [@harris:2020] way. The library enables physicists to
operate on high energy physics data in a high level language without
compromising speed. The library is already in use at LHC and is a part of
frameworks, like Coffea [@Gray:2023], employed by physicists across multiple
high energy physics experiments.

# Statement of need

Vector is one of the few Lorentz vector libraries that offer
a Pythonic interface but a compiled computational backend, with the others
being Coffea's vector module (depends on vector), PyROOT [@root:2020]'s
LorentzVectors and TLorentzVector classes, and hepvector [@hepvector] (deprecated
in favor of vector). Although vector was written with high energy physics in mind,
it is a general-purpose library that can be used for any scientific or engineering
application. The library houses a set of diverse backends, three numerical backends
for experimental physicists and one symbolic backend for theoretical physicists.
These backends are:

- a pure Python object (builtin) backend for scalar computations,
- a NumPy backend for computations on regular collection-type data,
- a SymPy [@Meurer:2017] backend for symbolic computations, and
- an Awkward [@Pivarski:2018] backend for computations on ragged collection-type data

Moreover, vector is the first Lorentz vector library to offer multiple
computational backends, as well as both numerical and symbolic backends. Furthermore,
akin to PyROOT and LorentzVectorHEP.jl [@LorentzVectorHEP:2023], vector supports
just-in-time compilation through Numba extensions [@lam:2015], implemented for both
the Object and Awkward backends. Vector also includes support for JAX [@Bradbury:2018]
and Dask [@rocklin:2015] for the Awkward backend, enabling the library to support
automatic differentiation and parallel computing, which are required for introducing
automatic differentiation in Analysis Grand Challenge [@Held:2022sfw] and to meet
the computational needs of High Luminosity LHC [@Aberle:2749422].

## Impact

Besides PyROOT's LorentzVectors and TLorentzVector, vector has
become a popular choice for mathematical manipulations in Python based high energy
physics analysis pipelines. Along with being utilized directly in
analysis pipelines at LHC [@Kling:2023; @Held:2024; @Qu:2022], the library is also
being used in other high energy physics experiments and as a dependency in other
user-facing frameworks, such as, Coffea, MadMiner [@Brehmer:2020], FastJet
[@aryan:2023], Spyral [@spyral-utils:2024], Weaver [@weaver-core:2024], and pylhe
[@pylhe]. The library is also used in multiple teaching materials for graduate
courses and workshops. Finally, given the generic nature of the library, it is
often used in non high energy physics use cases.

# Acknowledgements

The work on vector was supported by NSF cooperative agreements OAC-1836650
(IRIS-HEP) and PHY-2323298 (IRIS-HEP). We would additionally like to thank the
contributors of vector and the Scikit-HEP community for their support.

# Reference
