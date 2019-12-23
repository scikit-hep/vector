#!/usr/bin/env python
# Copyright (c) 2019, Eduardo Rodrigues and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import absolute_import
from __future__ import print_function

import os
import sys

from setuptools import setup


PYTHON_REQUIRES = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*"


def get_version():
    g = {}
    exec(open(os.path.join("vector", "_version.py")).read(), g)
    return g["__version__"]


setup(
    name="vector",
    author="Eduardo Rodrigues",
    author_email="eduardo.rodrigues@cern.ch",
    maintainer="The Scikit-HEP admins",
    maintainer_email="scikit-hep-admins@googlegroups.com",
    version=get_version(),
    description="Vector classes and utilities",
    long_description=open("README.md").read(),
    url="https://github.com/scikit-hep/vector",
    license="BSD 3-Clause License",
    packages=find_packages(),
    python_requires=PYTHON_REQUIRES,
    keywords=["vector"],
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 1 - Planning",
    ],
    platforms="Any",
)
