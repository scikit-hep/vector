# Copyright (c) 2019-2020, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy as np

# Functions that return a value


def pt(rec):
    return np.sqrt(rec.x ** 2 + rec.y ** 2)


def eta(rec):
    return np.arcsinh(rec.z / np.sqrt(rec.x ** 2 + rec.y ** 2))


def phi(rec):
    return np.arctan2(rec.y, rec.x)


def mass(rec):
    return np.sqrt(rec.t ** 2 - rec.x ** 2 - rec.y ** 2 - rec.z ** 2)
