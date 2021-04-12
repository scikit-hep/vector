#!/usr/bin/env python
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from setuptools import setup

extras = {
    "dev": [
        "awkward>=1.2.0",
        'numba>=0.50; python_version>="3.6"',
        "pytest>=4.6",
    ],
    "test": [
        "pytest>=4.6",
    ],
    "test_extras": [
        "uncompyle6",
        "spark-parser",
    ],
    "docs": [
        "nbsphinx",
        "recommonmark>=0.5.0",
        "Sphinx~=3.0",
        "sphinx_copybutton",
        "sphinx_book_theme~=0.0.42",
        "nbsphinx",
        "sphinx-math-dollar",
        "ipykernel",
        "awkward",
    ],
}

extras["all"] = sum(extras.values(), [])

setup(extras_require=extras)
