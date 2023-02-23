# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import math

import pytest

import vector.backends.object


def test_xy_xy():
    v1 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2)
    )
    v2 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4)
    )
    assert v1.deltaphi(v2) == pytest.approx(math.atan2(2, 1) - math.atan2(4, 3))


def test_xy_rhophi():
    v1 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2)
    )
    v2 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(3, 4)
    )
    assert v1.deltaphi(v2) == pytest.approx(math.atan2(2, 1) - 4)


def test_rhophi_xy():
    v1 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(1, 2)
    )
    v2 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4)
    )
    assert v1.deltaphi(v2) == pytest.approx(2 - math.atan2(4, 3))


def test_rhophi_rhophi():
    v1 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(1, 2)
    )
    v2 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(3, 4)
    )
    assert v1.deltaphi(v2) == pytest.approx(2 - 4)
