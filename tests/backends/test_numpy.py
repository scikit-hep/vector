# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import math
import pickle

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


def test_pickle_vector_numpy_2d():
    array = vector._backends.numpy_.VectorNumpy2D(
        [(0, 0), (0, 1), (3, 4)], dtype=[("x", numpy.float64), ("y", numpy.float64)]
    )

    array_pickled = pickle.dumps(array)
    array_new = pickle.loads(array_pickled)

    assert numpy.allclose(array_new.x, array.x)
    assert numpy.allclose(array_new.y, array.y)


def test_pickle_momentum_numpy_2d():
    array = vector._backends.numpy_.MomentumNumpy2D(
        [(0, 0), (0, 1), (3, 4)], dtype=[("rho", numpy.float64), ("phi", numpy.float64)]
    )

    array_pickled = pickle.dumps(array)
    array_new = pickle.loads(array_pickled)

    assert numpy.allclose(array_new.rho, array.rho)
    assert numpy.allclose(array_new.phi, array.phi)


def test_pickle_vector_numpy_3d():
    array = vector._backends.numpy_.VectorNumpy3D(
        [(0, 0, 0), (0, 1, 1), (3, 4, 5)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )

    array_pickled = pickle.dumps(array)
    array_new = pickle.loads(array_pickled)

    assert numpy.allclose(array_new.x, array.x)
    assert numpy.allclose(array_new.y, array.y)
    assert numpy.allclose(array_new.z, array.z)


def test_pickle_momentum_numpy_3d():
    array = vector._backends.numpy_.MomentumNumpy3D(
        [(0, 0, 0), (0, 1, 1), (3, 4, 5)],
        dtype=[
            ("rho", numpy.float64),
            ("phi", numpy.float64),
            ("theta", numpy.float64),
        ],
    )

    array_pickled = pickle.dumps(array)
    array_new = pickle.loads(array_pickled)

    assert numpy.allclose(array_new.rho, array.rho)
    assert numpy.allclose(array_new.phi, array.phi)
    assert numpy.allclose(array_new.theta, array.theta)


def test_pickle_vector_numpy_4d():
    array = vector._backends.numpy_.VectorNumpy4D(
        [(0, 0, 0, 0), (0, 1, 1, 1), (3, 4, 5, 6)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )

    array_pickled = pickle.dumps(array)
    array_new = pickle.loads(array_pickled)

    assert numpy.allclose(array_new.x, array.x)
    assert numpy.allclose(array_new.y, array.y)
    assert numpy.allclose(array_new.z, array.z)
    assert numpy.allclose(array_new.t, array.t)


def test_pickle_momentum_numpy_4d():
    array = vector._backends.numpy_.MomentumNumpy4D(
        [(0, 0, 0, 0), (0, 1, 1, 1), (3, 4, 5, 6)],
        dtype=[
            ("rho", numpy.float64),
            ("phi", numpy.float64),
            ("theta", numpy.float64),
            ("tau", numpy.float64),
        ],
    )

    array_pickled = pickle.dumps(array)
    array_new = pickle.loads(array_pickled)

    assert numpy.allclose(array_new.rho, array.rho)
    assert numpy.allclose(array_new.phi, array.phi)
    assert numpy.allclose(array_new.theta, array.theta)
    assert numpy.allclose(array_new.tau, array.tau)
