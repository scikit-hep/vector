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


def test_sum_2d():
    v = vector.VectorNumpy2D(
        [[(1, 0.1), (4, 0.2), (0, 0)], [(1, 0.3), (4, 0.4), (1, 0.1)]],
        dtype=[("rho", numpy.float64), ("phi", numpy.float64)],
    )
    assert numpy.sum(v, axis=0, keepdims=False).allclose(
        vector.VectorNumpy2D(
            [
                [
                    (1.950340654403632, 0.3953536233081677),
                    (7.604510287376507, 2.3523506924148467),
                    (0.9950041652780258, 0.09983341664682815),
                ]
            ],
            dtype=[("x", numpy.float64), ("y", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=0, keepdims=True).allclose(
        vector.VectorNumpy2D(
            [
                (1.950340654403632, 0.3953536233081677),
                (7.604510287376507, 2.3523506924148467),
                (0.9950041652780258, 0.09983341664682815),
            ],
            dtype=[("x", numpy.float64), ("y", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=0).allclose(
        vector.VectorNumpy2D(
            [
                (1.950340654403632, 0.3953536233081677),
                (7.604510287376507, 2.3523506924148467),
                (0.9950041652780258, 0.09983341664682815),
            ],
            dtype=[("x", numpy.float64), ("y", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=0, keepdims=False).shape == (3,)
    assert numpy.sum(v, axis=0, keepdims=True).shape == (1, 3)
    assert numpy.sum(v, axis=0).shape == (3,)

    assert numpy.sum(v, axis=1, keepdims=True).allclose(
        vector.VectorNumpy2D(
            [[(4.91527048, 0.89451074)], [(5.63458463, 1.95302699)]],
            dtype=[("x", numpy.float64), ("y", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=1, keepdims=False).allclose(
        vector.VectorNumpy2D(
            [(4.91527048, 0.89451074), (5.63458463, 1.95302699)],
            dtype=[("x", numpy.float64), ("y", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=1).allclose(
        vector.VectorNumpy2D(
            [(4.91527048, 0.89451074), (5.63458463, 1.95302699)],
            dtype=[("x", numpy.float64), ("y", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=1, keepdims=False).shape == (2,)
    assert numpy.sum(v, axis=1, keepdims=True).shape == (2, 1)
    assert numpy.sum(v, axis=1).shape == (2,)


def test_sum_3d():
    v = vector.VectorNumpy3D(
        [
            [(1, 2, 0.1), (4, 5, 0.2), (0, 0, 0.04)],
            [(1, 2, 0.6), (4, 5, 1.3), (1, 1, 1.9)],
        ],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("theta", numpy.float64),
        ],
    )
    assert numpy.sum(v, axis=0, keepdims=True).allclose(
        vector.VectorNumpy3D(
            [
                [
                    (2.0, 4.0, 25.55454594),
                    (8.0, 10.0, 33.36521103),
                    (1.0, 1.0, -0.48314535),
                ]
            ],
            dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=0, keepdims=False).allclose(
        vector.VectorNumpy3D(
            [
                (2.0, 4.0, 25.55454594),
                (8.0, 10.0, 33.36521103),
                (1.0, 1.0, -0.48314535),
            ],
            dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=0).allclose(
        vector.VectorNumpy3D(
            [
                (2.0, 4.0, 25.55454594),
                (8.0, 10.0, 33.36521103),
                (1.0, 1.0, -0.48314535),
            ],
            dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=0, keepdims=False).shape == (3,)
    assert numpy.sum(v, axis=0, keepdims=True).shape == (1, 3)
    assert numpy.sum(v, axis=0).shape == (3,)

    assert numpy.sum(v, axis=1, keepdims=True).allclose(
        vector.VectorNumpy3D(
            [[(5.0, 7.0, 53.87369799)], [(6.0, 8.0, 4.56291362)]],
            dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=1, keepdims=False).allclose(
        vector.VectorNumpy3D(
            [(5.0, 7.0, 53.87369799), (6.0, 8.0, 4.56291362)],
            dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=1).allclose(
        vector.VectorNumpy3D(
            [(5.0, 7.0, 53.87369799), (6.0, 8.0, 4.56291362)],
            dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
        )
    )
    assert numpy.sum(v, axis=1, keepdims=False).shape == (2,)
    assert numpy.sum(v, axis=1, keepdims=True).shape == (2, 1)
    assert numpy.sum(v, axis=1).shape == (2,)


def test_sum_4d():
    v = vector.VectorNumpy4D(
        [
            [(1, 2, 3, 4), (4, 5, 6, 2), (0, 0, 0, 3)],
            [(1, 2, 3, 8), (4, 5, 6, 0), (1, 1, 1, 0)],
        ],
        dtype=[
            ("x", numpy.int64),
            ("y", numpy.int64),
            ("z", numpy.int64),
            ("t", numpy.int64),
        ],
    )
    assert numpy.sum(v, axis=0, keepdims=True).tolist() == [
        [(2, 4, 6, 12), (8, 10, 12, 2), (1, 1, 1, 3)]
    ]
    assert numpy.sum(v, axis=0, keepdims=False).tolist() == [
        (2, 4, 6, 12),
        (8, 10, 12, 2),
        (1, 1, 1, 3),
    ]
    assert numpy.sum(v, axis=0).tolist() == [
        (2, 4, 6, 12),
        (8, 10, 12, 2),
        (1, 1, 1, 3),
    ]
    assert numpy.sum(v, axis=0, keepdims=False).shape == (3,)
    assert numpy.sum(v, axis=0, keepdims=True).shape == (1, 3)
    assert numpy.sum(v, axis=0).shape == (3,)

    assert numpy.sum(v, axis=1, keepdims=True).tolist() == [
        [(5, 7, 9, 9)],
        [(6, 8, 10, 8)],
    ]
    assert numpy.sum(v, axis=1, keepdims=False).tolist() == [
        (5, 7, 9, 9),
        (6, 8, 10, 8),
    ]
    assert numpy.sum(v, axis=1).tolist() == [
        (5, 7, 9, 9),
        (6, 8, 10, 8),
    ]
    assert numpy.sum(v, axis=1, keepdims=False).shape == (2,)
    assert numpy.sum(v, axis=1, keepdims=True).shape == (2, 1)
    assert numpy.sum(v, axis=1).shape == (2,)


def test_count_nonzero_2d():
    v = vector.VectorNumpy2D(
        [
            [(1, 0.1), (4, 0.2), (0, 0)],
            [(1, 0.3), (4, 0.4), (1, 0.1)],
        ],
        dtype=[("rho", numpy.float64), ("phi", numpy.float64)],
    )
    assert numpy.count_nonzero(v, axis=1).tolist() == [2, 3]
    assert numpy.count_nonzero(v, axis=1, keepdims=True).tolist() == [[2], [3]]
    assert numpy.count_nonzero(v, axis=0).tolist() == [2, 2, 1]
    assert numpy.count_nonzero(v, axis=0, keepdims=True).tolist() == [[2, 2, 1]]


def test_count_nonzero_3d():
    v = vector.VectorNumpy3D(
        [
            [(1, 2, 0.1), (4, 5, 0.2), (0, 0, 0)],
            [(1, 2, 0.6), (4, 5, 1.3), (1, 1, 1.9)],
        ],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("theta", numpy.float64)],
    )
    assert numpy.count_nonzero(v, axis=1).tolist() == [2, 3]
    assert numpy.count_nonzero(v, axis=1, keepdims=True).tolist() == [[2], [3]]
    assert numpy.count_nonzero(v, axis=0).tolist() == [2, 2, 1]
    assert numpy.count_nonzero(v, axis=0, keepdims=True).tolist() == [[2, 2, 1]]


def test_count_nonzero_4d():
    v = vector.VectorNumpy4D(
        [
            [(1, 2, 3, 4), (4, 5, 6, 2), (0, 0, 0, 3)],
            [(1, 2, 3, 8), (4, 5, 6, 0), (1, 1, 1, 0)],
        ],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    assert numpy.count_nonzero(v, axis=1).tolist() == [3, 3]
    assert numpy.count_nonzero(v, axis=1, keepdims=True).tolist() == [[3], [3]]
    assert numpy.count_nonzero(v, axis=0).tolist() == [2, 2, 2]
    assert numpy.count_nonzero(v, axis=0, keepdims=True).tolist() == [[2, 2, 2]]

    v2 = vector.VectorNumpy4D(
        [
            [(1, 2, 3, 1), (4, 5, 6, 2), (0, 0, 0, 2)],
            [(1, 2, 3, 0), (4, 5, 6, 1), (0, 0, 0, 0)],
        ],
        dtype=[
            ("x", numpy.int64),
            ("y", numpy.int64),
            ("z", numpy.int64),
            ("t", numpy.int64),
        ],
    )
    assert numpy.count_nonzero(v2, axis=1).tolist() == [3, 2]
    assert numpy.count_nonzero(v2, axis=1, keepdims=True).tolist() == [[3], [2]]
    assert numpy.count_nonzero(v2, axis=0).tolist() == [2, 2, 1]
    assert numpy.count_nonzero(v2, axis=0, keepdims=True).tolist() == [[2, 2, 1]]


def test_demotion():
    v1 = vector.array(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
        },
    )
    v2 = vector.array(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 1.0, 1.0],
        },
    )
    v3 = vector.array(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 1.0, 1.0],
            "t": [16.0, 31.0, 46.0],
        },
    )

    v1_v2 = vector.array(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
        },
    )
    v2_v3 = vector.array(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
            "z": [10.0, 2.0, 2.0],
        },
    )
    v1_v3 = vector.array(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
        },
    )

    # order should not matter
    assert all(v1 + v2 == v1_v2)
    assert all(v2 + v1 == v1_v2)
    assert all(v1 + v3 == v1_v3)
    assert all(v3 + v1 == v1_v3)
    assert all(v2 + v3 == v2_v3)
    assert all(v3 + v2 == v2_v3)

    v1 = vector.array(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
        },
    )
    v2 = vector.array(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
            "pz": [5.0, 1.0, 1.0],
        },
    )
    v3 = vector.array(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
            "pz": [5.0, 1.0, 1.0],
            "t": [16.0, 31.0, 46.0],
        },
    )

    p_v1_v2 = vector.array(
        {
            "px": [20.0, 40.0, 60.0],
            "py": [-20.0, 40.0, 60.0],
        },
    )
    p_v2_v3 = vector.array(
        {
            "px": [20.0, 40.0, 60.0],
            "py": [-20.0, 40.0, 60.0],
            "pz": [10.0, 2.0, 2.0],
        },
    )
    p_v1_v3 = vector.array(
        {
            "px": [20.0, 40.0, 60.0],
            "py": [-20.0, 40.0, 60.0],
        },
    )

    # order should not matter
    assert all(v1 + v2 == p_v1_v2)
    assert all(v2 + v1 == p_v1_v2)
    assert all(v1 + v3 == p_v1_v3)
    assert all(v3 + v1 == p_v1_v3)
    assert all(v2 + v3 == p_v2_v3)
    assert all(v3 + v2 == p_v2_v3)

    v2 = vector.array(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 1.0, 1.0],
        },
    )

    # momentum + generic = generic
    assert all(v1 + v2 == v1_v2)
    assert all(v2 + v1 == v1_v2)
    assert all(v1 + v3 == v1_v3)
    assert all(v3 + v1 == v1_v3)
    assert all(v2 + v3 == v2_v3)
    assert all(v3 + v2 == v2_v3)
