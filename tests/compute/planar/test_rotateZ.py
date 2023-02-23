# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy
import pytest

import vector.backends.numpy
import vector.backends.object


def test_xy():
    vec = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 0)
    )
    assert vec.rotateZ(0.1).x == pytest.approx(0.9950041652780258)
    assert vec.rotateZ(0.1).y == pytest.approx(0.09983341664682815)

    array = vector.backends.numpy.VectorNumpy2D(
        [(0, 0), (1, 0), (0, 1)], dtype=[("x", numpy.float64), ("y", numpy.float64)]
    )
    assert isinstance(array.rotateZ(0.1), vector.backends.numpy.VectorNumpy2D)
    out = array.rotateZ(0.1)
    assert out.dtype.names == ("x", "y")
    assert numpy.allclose(out.x, [0, 0.9950041652780258, -0.09983341664682815])
    assert numpy.allclose(out.y, [0, 0.09983341664682815, 0.9950041652780258])


def test_rhophi():
    vec = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(1, 0)
    )
    assert vec.rotateZ(0.1).rho == pytest.approx(1)
    assert vec.rotateZ(0.1).phi == pytest.approx(0.1)

    array = vector.backends.numpy.VectorNumpy2D(
        [(0, 0), (1, 0), (0, 1)], dtype=[("rho", numpy.float64), ("phi", numpy.float64)]
    )
    assert isinstance(array.rotateZ(0.1), vector.backends.numpy.VectorNumpy2D)
    out = array.rotateZ(0.1)
    assert out.dtype.names == ("rho", "phi")
    assert numpy.allclose(out.rho, [0, 1, 0])
    assert numpy.allclose(out.phi, [0.1, 0.1, 1.1])
