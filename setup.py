#!/usr/bin/env python
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from setuptools import setup

extras = {
    "dev": [
        "awkward1>=0.1.128",
        "uproot>=3.11",
        'numba>=0.46; python_version>="3"',
        "scikit-hep-testdata>=0.2.0",
    ],
    "test": ["pytest>=4.6"],
}

extras["all"] = sum(extras.values(), [])

setup(extras_require=extras)
