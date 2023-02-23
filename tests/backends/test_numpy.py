# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import math
import pickle

import numpy
import pytest

import vector.backends.numpy


def test_dimension_conversion():
    # 2D -> 3D
    vec = vector.VectorNumpy2D(
        [(1.0, 1.0), (2.0, 2.0)],
        dtype=[("x", float), ("y", float)],
    )
    assert all(vec.to_Vector3D(z=1).z == 1)
    assert all(vec.to_Vector3D(eta=1).eta == 1)
    assert all(vec.to_Vector3D(theta=1).theta == 1)

    assert all(vec.to_Vector3D(z=1).x == vec.x)
    assert all(vec.to_Vector3D(z=1).y == vec.y)

    # 2D -> 4D
    assert all(vec.to_Vector4D(z=1, t=1).t == 1)
    assert all(vec.to_Vector4D(z=1, t=1).z == 1)
    assert all(vec.to_Vector4D(eta=1, t=1).eta == 1)
    assert all(vec.to_Vector4D(eta=1, t=1).t == 1)
    assert all(vec.to_Vector4D(theta=1, t=1).theta == 1)
    assert all(vec.to_Vector4D(theta=1, t=1).t == 1)
    assert all(vec.to_Vector4D(z=1, tau=1).z == 1)
    assert all(vec.to_Vector4D(z=1, tau=1).tau == 1)
    assert all(vec.to_Vector4D(eta=1, tau=1).eta == 1)
    assert all(vec.to_Vector4D(eta=1, tau=1).tau == 1)
    assert all(vec.to_Vector4D(theta=1, tau=1).theta == 1)
    assert all(vec.to_Vector4D(theta=1, tau=1).tau == 1)

    assert all(vec.to_Vector4D(z=1, t=1).x == vec.x)
    assert all(vec.to_Vector4D(z=1, t=1).y == vec.y)

    # 3D -> 4D
    vec = vector.VectorNumpy3D(
        [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        dtype=[("x", float), ("y", float), ("z", float)],
    )
    assert all(vec.to_Vector4D(t=1).t == 1)
    assert all(vec.to_Vector4D(tau=1).tau == 1)

    assert all(vec.to_Vector4D(t=1).x == vec.x)
    assert all(vec.to_Vector4D(t=1).y == vec.y)
    assert all(vec.to_Vector4D(t=1).z == vec.z)


def test_type_checks():
    with pytest.raises(TypeError):
        vector.backends.numpy.VectorNumpy2D(
            (0, 0), dtype=[("x", numpy.float64), ("y", numpy.timedelta64)]
        )

    with pytest.raises(TypeError):
        vector.backends.numpy.VectorNumpy2D(
            (0, 0), dtype=[("x", numpy.complex64), ("y", numpy.float64)]
        )

    with pytest.raises(TypeError):
        vector.backends.numpy.VectorNumpy2D(
            [(0, 0), (0, 1), (3, 4)],
            dtype=[("x", numpy.complex64), ("y", numpy.float64)],
        )

    with pytest.raises(TypeError):
        vector.backends.numpy.VectorNumpy3D(
            [([0], 0, 0), ([0], 1, 2), ([3], 4, 5)],
            dtype=[("x", list), ("y", numpy.float64), ("z", numpy.float64)],
        )

    with pytest.raises(TypeError):
        vector.backends.numpy.VectorNumpy4D(
            [(0, 0, 0, 0), (0, 1, 2, 3), (3, 4, 5, 6)],
            [
                ("x", complex),
                ("y", numpy.float64),
                ("z", numpy.float64),
                ("t", numpy.float64),
            ],
        )

    with pytest.raises(TypeError):
        vector.backends.numpy.MomentumNumpy2D(
            (0, 0), dtype=[("x", numpy.float64), ("y", numpy.timedelta64)]
        )

    with pytest.raises(TypeError):
        vector.backends.numpy.MomentumNumpy2D(
            (0, 0), dtype=[("x", numpy.complex64), ("y", numpy.float64)]
        )

    with pytest.raises(TypeError):
        vector.backends.numpy.MomentumNumpy3D(
            [(0, 0), (0, 1), (3, 4)],
            dtype=[("x", numpy.complex64), ("y", numpy.float64)],
        )

    with pytest.raises(TypeError):
        vector.backends.numpy.MomentumNumpy4D(
            [([0], 0, 0), ([0], 1, 2), ([3], 4, 5)],
            dtype=[("x", list), ("y", numpy.float64), ("z", numpy.float64)],
        )

    with pytest.raises(TypeError):
        vector.backends.numpy.MomentumNumpy4D(
            [(0, 0, 0, 0), (0, 1, 2, 3), (3, 4, 5, 6)],
            [
                ("x", complex),
                ("y", numpy.float64),
                ("z", numpy.float64),
                ("t", numpy.float64),
            ],
        )


def test_xy():
    array = vector.backends.numpy.VectorNumpy2D(
        [(0, 0), (0, 1), (3, 4)], dtype=[("x", numpy.float64), ("y", numpy.float64)]
    )
    assert numpy.allclose(array.x, [0, 0, 3])
    assert numpy.allclose(array.y, [0, 1, 4])
    assert numpy.allclose(array.rho, [0, 1, 5])
    assert numpy.allclose(array.phi, [0, math.atan2(1, 0), math.atan2(4, 3)])


def test_rhophi():
    array = vector.backends.numpy.VectorNumpy2D(
        [(0, 10), (1, math.atan2(1, 0)), (5, math.atan2(4, 3))],
        dtype=[("rho", numpy.float64), ("phi", numpy.float64)],
    )
    assert numpy.allclose(array.x, [0, 0, 3])
    assert numpy.allclose(array.y, [0, 1, 4])
    assert numpy.allclose(array.rho, [0, 1, 5])
    assert numpy.allclose(array.phi, [10, math.atan2(1, 0), math.atan2(4, 3)])


def test_pickle_vector_numpy_2d():
    array = vector.backends.numpy.VectorNumpy2D(
        [(0, 0), (0, 1), (3, 4)], dtype=[("x", numpy.float64), ("y", numpy.float64)]
    )

    array_pickled = pickle.dumps(array)
    array_new = pickle.loads(array_pickled)

    assert numpy.allclose(array_new.x, array.x)
    assert numpy.allclose(array_new.y, array.y)


def test_pickle_momentum_numpy_2d():
    array = vector.backends.numpy.MomentumNumpy2D(
        [(0, 0), (0, 1), (3, 4)], dtype=[("rho", numpy.float64), ("phi", numpy.float64)]
    )

    array_pickled = pickle.dumps(array)
    array_new = pickle.loads(array_pickled)

    assert numpy.allclose(array_new.rho, array.rho)
    assert numpy.allclose(array_new.phi, array.phi)


def test_pickle_vector_numpy_3d():
    array = vector.backends.numpy.VectorNumpy3D(
        [(0, 0, 0), (0, 1, 1), (3, 4, 5)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )

    array_pickled = pickle.dumps(array)
    array_new = pickle.loads(array_pickled)

    assert numpy.allclose(array_new.x, array.x)
    assert numpy.allclose(array_new.y, array.y)
    assert numpy.allclose(array_new.z, array.z)


def test_pickle_momentum_numpy_3d():
    array = vector.backends.numpy.MomentumNumpy3D(
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
    array = vector.backends.numpy.VectorNumpy4D(
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
    array = vector.backends.numpy.MomentumNumpy4D(
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
