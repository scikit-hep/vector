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


def test_property_float():
    @numba.jit(nopython=True)
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
