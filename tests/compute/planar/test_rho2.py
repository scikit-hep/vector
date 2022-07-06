# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import math

import pytest

import vector.backends.object


def test_xy():
    vec = vector.backends.object.VectorObject2D(
        vector.backends.object.AzimuthalObjectXY(3, 4)
    )
    assert vec.rho2 == pytest.approx(25)


def test_rhophi():
    vec = vector.backends.object.VectorObject2D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, math.atan2(4, 3))
    )
    assert vec.rho2 == pytest.approx(25)
