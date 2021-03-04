# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import math

import pytest

import vector.backends.object_


def test_xy():
    vec = vector.backends.object_.PlanarVectorObject(
        vector.backends.object_.AzimuthalObjectXY(3, 4)
    )
    assert pytest.approx(vec.rho, 5)


def test_rhophi():
    vec = vector.backends.object_.PlanarVectorObject(
        vector.backends.object_.AzimuthalObjectRhoPhi(5, math.atan2(4, 3))
    )
    assert pytest.approx(vec.rho, 5)
