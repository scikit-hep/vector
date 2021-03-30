# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import math

import pytest

import vector._backends.object_


def test_xy_xy():
    v1 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectXY(1, 2)
    )
    v2 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectXY(3, 4)
    )
    assert v1.deltaphi(v2) == pytest.approx(math.atan2(2, 1) - math.atan2(4, 3))


def test_xy_rhophi():
    v1 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectXY(1, 2)
    )
    v2 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectRhoPhi(3, 4)
    )
    assert v1.deltaphi(v2) == pytest.approx(math.atan2(2, 1) - 4)


def test_rhophi_xy():
    v1 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectRhoPhi(1, 2)
    )
    v2 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectXY(3, 4)
    )
    assert v1.deltaphi(v2) == pytest.approx(2 - math.atan2(4, 3))


def test_rhophi_rhophi():
    v1 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectRhoPhi(1, 2)
    )
    v2 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectRhoPhi(3, 4)
    )
    assert v1.deltaphi(v2) == pytest.approx(2 - 4)
