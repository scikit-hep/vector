# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import sys

import numpy
import pytest

import vector
import vector.backends.object_

numba = pytest.importorskip("numba")
pytest.importorskip("vector.backends.numba_object")


def test_namedtuples():
    @numba.njit
    def get_x(obj):
        return obj.x

    assert get_x(vector.backends.object_.AzimuthalObjectXY(1, 2.2)) == 1
    assert get_x(vector.backends.object_.AzimuthalObjectXY(1.1, 2)) == 1.1


def test_VectorObject2DType():
    # These tests verify that the reference counts for Python objects touched in
    # the lowered Numba code do not increase or decrease with the number of times
    # the function is run.

    @numba.njit
    def zero(obj):
        return None

    @numba.njit
    def one(obj):
        return obj

    @numba.njit
    def two(obj):
        return obj, obj

    obj = vector.obj(x=1, y=2)
    assert (sys.getrefcount(obj), sys.getrefcount(obj.azimuthal)) == (2, 2)

    class_refs = None
    for _ in range(10):
        zero(obj)
        assert (sys.getrefcount(obj), sys.getrefcount(obj.azimuthal)) == (2, 2)
        if class_refs is None:
            class_refs = sys.getrefcount(vector.backends.object_.VectorObject2D)
        assert class_refs + 1 == sys.getrefcount(vector.backends.object_.VectorObject2D)

    class_refs = None
    for _ in range(10):
        a = one(obj)
        assert (sys.getrefcount(obj), sys.getrefcount(obj.azimuthal)) == (2, 2)
        assert (sys.getrefcount(a), sys.getrefcount(a.azimuthal)) == (2, 2)
        if class_refs is None:
            class_refs = sys.getrefcount(vector.backends.object_.VectorObject2D)
        assert class_refs + 1 == sys.getrefcount(vector.backends.object_.VectorObject2D)

    class_refs = None
    for _ in range(10):
        a, b = two(obj)
        assert (sys.getrefcount(obj), sys.getrefcount(obj.azimuthal)) == (2, 2)
        assert (
            sys.getrefcount(a),
            sys.getrefcount(a.azimuthal),
            sys.getrefcount(b),
            sys.getrefcount(b.azimuthal),
        ) == (2, 2, 2, 2)
        if class_refs is None:
            class_refs = sys.getrefcount(vector.backends.object_.VectorObject2D)
        assert class_refs + 1 == sys.getrefcount(vector.backends.object_.VectorObject2D)


def test_VectorObject2D_constructor():
    @numba.njit
    def constructXY():
        return vector.backends.object_.VectorObject2D(
            vector.backends.object_.AzimuthalObjectXY(1.1, 2.2)
        )

    @numba.njit
    def constructRhoPhi():
        return vector.backends.object_.VectorObject2D(
            vector.backends.object_.AzimuthalObjectRhoPhi(1.1, 2.2)
        )

    out = constructXY()
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.x == pytest.approx(1.1)
    assert out.y == pytest.approx(2.2)

    out = constructRhoPhi()
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.rho == pytest.approx(1.1)
    assert out.phi == pytest.approx(2.2)


def test_property_float():
    @numba.njit
    def get_x(v):
        return v.x

    assert get_x(vector.obj(x=1.1, y=2)) == pytest.approx(1.1)


def test_method_float():
    @numba.njit
    def get_deltaphi(v1, v2):
        return v1.deltaphi(v2)

    assert get_deltaphi(vector.obj(x=1, y=0), vector.obj(x=0, y=1)) == pytest.approx(
        -numpy.pi / 2
    )


def test_method_vector():
    @numba.njit
    def get_rotateZ(v, angle):
        return v.rotateZ(angle)

    out = get_rotateZ(vector.obj(x=1, y=0), 0.1)
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.x == pytest.approx(0.9950041652780258)
    assert out.y == pytest.approx(0.09983341664682815)
