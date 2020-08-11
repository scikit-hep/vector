# -*- coding: utf-8 -*-
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import absolute_import, division, print_function

from vector.core import numpy as np

# Functions that return a value

# x: already included
# y: already included
# z: already included
# t: already included


def add(left, right):
    x = left.x + right.x
    y = left.y + right.y
    z = left.z + right.z
    t = left.t + right.t

    return x, y, z, t


def pt(vec):
    return np.sqrt(vec.x ** 2 + vec.y ** 2)


def eta(vec):
    return np.arcsinh(vec.z / np.sqrt(vec.x ** 2 + vec.y ** 2))


def phi(vec):
    return np.arctan2(vec.y, vec.x)


# mass
def mag(vec):
    return np.sqrt(vec.t ** 2 - vec.x ** 2 - vec.y ** 2 - vec.z ** 2)


# mass2
def mag2(vec):
    return vec.t ** 2 - vec.x ** 2 - vec.y ** 2 - vec.z ** 2
