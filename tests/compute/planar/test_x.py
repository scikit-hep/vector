# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import math

import pytest

import vector.backends.object


def test_xy():
    vec = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4)
    )
    assert vec.x == pytest.approx(3)


def test_rhophi():
    vec = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, math.atan2(4, 3))
    )
    assert vec.x == pytest.approx(3)
