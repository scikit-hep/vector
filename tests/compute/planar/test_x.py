# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
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
