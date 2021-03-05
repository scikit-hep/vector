# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import math

import numpy

import vector.backends.numpy_


def test_xy():
    array = vector.backends.numpy_.PlanarVectorNumpy(
        [(0, 0), (0, 1), (3, 4)], dtype=[("x", numpy.float64), ("y", numpy.float64)]
    )
    assert numpy.allclose(array.x, [0, 0, 3])
    assert numpy.allclose(array.y, [0, 1, 4])
    assert numpy.allclose(array.rho, [0, 1, 5])
    assert numpy.allclose(array.phi, [0, math.atan2(1, 0), math.atan2(4, 3)])

    trans = vector.backends.numpy_.Transform2DNumpy(
        [(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)],
        dtype=[
            ("xx", numpy.float64),
            ("xy", numpy.float64),
            ("yx", numpy.float64),
            ("yy", numpy.float64),
        ],
    )
    out = trans(array)
    assert isinstance(out, vector.backends.numpy_.PlanarVectorNumpy)
    assert numpy.allclose(out.x, [0, 2, 11])
    assert numpy.allclose(out.y, [0, 4, 25])

    trans = vector.backends.numpy_.AzimuthalRotationNumpy(
        [(0,), (0,), (math.pi / 2)],
        dtype=[
            ("angle", numpy.float64),
        ],
    )
    assert not isinstance(trans.angle, vector.backends.numpy_.AzimuthalRotationNumpy)
    assert numpy.allclose(trans.angle, [0, 0, math.pi / 2])
    out = trans(array)
    assert isinstance(out, vector.backends.numpy_.PlanarVectorNumpy)
    assert numpy.allclose(out.x, [0, 0, -4])
    assert numpy.allclose(out.y, [0, 1, 3])


def test_rhophi():
    array = vector.backends.numpy_.PlanarVectorNumpy(
        [(0, 10), (1, math.atan2(1, 0)), (5, math.atan2(4, 3))],
        dtype=[("rho", numpy.float64), ("phi", numpy.float64)],
    )
    assert numpy.allclose(array.x, [0, 0, 3])
    assert numpy.allclose(array.y, [0, 1, 4])
    assert numpy.allclose(array.rho, [0, 1, 5])
    assert numpy.allclose(array.phi, [10, math.atan2(1, 0), math.atan2(4, 3)])
