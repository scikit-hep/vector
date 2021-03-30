# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import math

import numpy

import vector._backends.numpy_


def test_xy():
    array = vector._backends.numpy_.VectorNumpy2D(
        [(0, 0), (0, 1), (3, 4)], dtype=[("x", numpy.float64), ("y", numpy.float64)]
    )
    assert numpy.allclose(array.x, [0, 0, 3])
    assert numpy.allclose(array.y, [0, 1, 4])
    assert numpy.allclose(array.rho, [0, 1, 5])
    assert numpy.allclose(array.phi, [0, math.atan2(1, 0), math.atan2(4, 3)])


def test_rhophi():
    array = vector._backends.numpy_.VectorNumpy2D(
        [(0, 10), (1, math.atan2(1, 0)), (5, math.atan2(4, 3))],
        dtype=[("rho", numpy.float64), ("phi", numpy.float64)],
    )
    assert numpy.allclose(array.x, [0, 0, 3])
    assert numpy.allclose(array.y, [0, 1, 4])
    assert numpy.allclose(array.rho, [0, 1, 5])
    assert numpy.allclose(array.phi, [10, math.atan2(1, 0), math.atan2(4, 3)])
