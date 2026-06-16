# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import math
import pickle

import numpy
import pytest

import vector.backends.numpy


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


def test_setitem_boolean_mask_regression():
    # Bug 1: ``_setitem`` raised UnboundLocalError for non-momentum structured
    # assignment, and boolean/fancy masks silently discarded the writes.
    p = vector.array({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    p[numpy.array([True, False])] = vector.array({"x": [99.0], "y": [88.0]})
    assert p.x.tolist() == [99.0, 2.0]
    assert p.y.tolist() == [88.0, 4.0]

    # fancy index
    pf = vector.array({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    pf[numpy.array([0, 2])] = vector.array({"x": [10.0, 30.0], "y": [40.0, 60.0]})
    assert pf.x.tolist() == [10.0, 2.0, 30.0]
    assert pf.y.tolist() == [40.0, 5.0, 60.0]

    # slice still works
    ps = vector.array({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    ps[1:2] = vector.array({"x": [20.0], "y": [50.0]})
    assert ps.x.tolist() == [1.0, 20.0, 3.0]

    # momentum assignment with px/py names
    pm = vector.array({"px": [1.0, 2.0], "py": [3.0, 4.0]})
    pm[numpy.array([False, True])] = vector.array({"px": [7.0], "py": [8.0]})
    assert pm.px.tolist() == [1.0, 7.0]


def test_array_from_structured_array_regression():
    # Bug 2: ``vector.array`` of a single structured array ignored its fields and
    # returned a 2D vector regardless of dimensionality.
    sa = numpy.array(
        [(1.0, 2.0, 3.0, 4.0)],
        dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("t", "f8")],
    )
    assert isinstance(vector.array(sa), vector.backends.numpy.VectorNumpy4D)

    sa3 = numpy.array([(1.0, 2.0, 3.0)], dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    assert isinstance(vector.array(sa3), vector.backends.numpy.VectorNumpy3D)

    # momentum-eligible names route to the momentum class
    sap = numpy.array(
        [(1.0, 2.0, 3.0, 4.0)],
        dtype=[("px", "f8"), ("py", "f8"), ("pz", "f8"), ("E", "f8")],
    )
    out = vector.array(sap)
    assert isinstance(out, vector.backends.numpy.MomentumNumpy4D)


def test_coordinate_dtype_isolation_regression():
    # Bug 3: the coordinate classes used to overwrite a mutable ``dtype`` class
    # attribute on every construction, so every instance reported the dtype of
    # the most recent construction process-wide.
    a = vector.backends.numpy.AzimuthalNumpyXY(
        [(1, 1)], dtype=[("x", numpy.float32), ("y", numpy.float32)]
    )
    b = vector.backends.numpy.AzimuthalNumpyXY(
        [(1, 1)], dtype=[("x", numpy.float64), ("y", numpy.float64)]
    )
    assert a.dtype == numpy.dtype([("x", numpy.float32), ("y", numpy.float32)])
    assert b.dtype == numpy.dtype([("x", numpy.float64), ("y", numpy.float64)])
    # constructing b must not have changed a
    assert a.dtype["x"] == numpy.float32


def test_coordinate_eq_against_scalar_regression():
    # Bug 4: ``az == 5`` raised AttributeError (accessing ``other.dtype`` before
    # the isinstance check) instead of returning False.
    az = vector.backends.numpy.AzimuthalNumpyXY(
        [(1, 1)], dtype=[("x", float), ("y", float)]
    )
    assert (az == 5) is False
    assert (az != 5) is True

    lg = vector.backends.numpy.LongitudinalNumpyZ([(1,)], dtype=[("z", float)])
    assert (lg == 5) is False
    assert (lg != 5) is True

    # length mismatch must be False, not a raised ValueError
    az2 = vector.backends.numpy.AzimuthalNumpyXY(
        [(1, 1), (2, 2)], dtype=[("x", float), ("y", float)]
    )
    assert (az == az2) is False
    assert (az != az2) is True


def test_view_does_not_mutate_source_dtype_regression():
    # Bug 5: ``arr.view(MomentumNumpy2D)`` mutated the caller's shared dtype,
    # renaming arr.dtype.names from ('px', 'py') to ('x', 'y').
    arr = numpy.array([(1.0, 2.0)], dtype=[("px", "f8"), ("py", "f8")])
    v = arr.view(vector.MomentumNumpy2D)
    assert arr.dtype.names == ("px", "py")
    assert v.px.tolist() == [1.0]
    assert v.py.tolist() == [2.0]

    arr4 = numpy.array(
        [(1.0, 2.0, 3.0, 4.0)],
        dtype=[("px", "f8"), ("py", "f8"), ("pz", "f8"), ("E", "f8")],
    )
    v4 = arr4.view(vector.MomentumNumpy4D)
    assert arr4.dtype.names == ("px", "py", "pz", "E")
    assert v4.energy.tolist() == [4.0]


def test_getitem_non_canonical_field_order_regression():
    # Bug 6: ``_getitem`` indexed coordinate views positionally, returning wrong
    # coordinates for arrays whose fields are not in canonical order.
    nc = vector.array(
        numpy.array([(5.0, 1.0, 2.0)], dtype=[("z", "f8"), ("x", "f8"), ("y", "f8")])
    )
    assert isinstance(nc, vector.backends.numpy.VectorNumpy3D)
    elem = nc[0]
    assert elem.x == 1.0
    assert elem.y == 2.0
    assert elem.z == 5.0

    # longitudinal coordinate view (retains full dtype) indexed by name
    assert nc.longitudinal[0].z == 5.0

    nc4 = vector.array(
        numpy.array(
            [(4.0, 5.0, 1.0, 2.0)],
            dtype=[("t", "f8"), ("z", "f8"), ("x", "f8"), ("y", "f8")],
        )
    )
    assert isinstance(nc4, vector.backends.numpy.VectorNumpy4D)
    e4 = nc4[0]
    assert (e4.x, e4.y, e4.z, e4.t) == (1.0, 2.0, 5.0, 4.0)
    assert nc4.temporal[0].t == 4.0


def test_wrap_result_azimuthal_precedence_regression():
    # Bug 7: an operator-precedence error in ``_wrap_result`` meant a single
    # azimuthal return relied on a mis-bound isinstance check. Exercise an
    # operation that returns a 2D (azimuthal-only) result.
    v = vector.array({"x": [3.0], "y": [4.0]})
    rotated = v.rotateZ(0.0)
    assert isinstance(rotated, vector.backends.numpy.VectorNumpy2D)
    assert rotated.x.tolist() == pytest.approx([3.0])
    assert rotated.y.tolist() == pytest.approx([4.0])
