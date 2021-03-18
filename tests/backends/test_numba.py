# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import sys

import pytest
import numpy

import vector


numba = pytest.importorskip("numba")


import vector.backends.numba_


def test_namedtuples():
    @numba.njit
    def get_x(obj):
        return obj.x

    assert get_x(vector.backends.object_.AzimuthalObjectXY(1, 2.2)) == 1
    assert get_x(vector.backends.object_.AzimuthalObjectXY(1.1, 2)) == 1.1


def test_VectorObject2DType():
    @numba.njit(debug=True)
    def zero(obj):
        return None

    @numba.njit(debug=True)
    def one(obj):
        return obj

    @numba.njit(debug=True)
    def two(obj):
        return obj, obj

    obj = vector.obj(x=1, y=2)
    assert (sys.getrefcount(obj), sys.getrefcount(obj.azimuthal)) == (2, 2)

    class_refs = None
    for i in range(10):
        zero(obj)
        assert (sys.getrefcount(obj), sys.getrefcount(obj.azimuthal)) == (2, 2)
        if class_refs is None:
            class_refs = sys.getrefcount(vector.backends.object_.VectorObject2D)
        assert class_refs + 1 == sys.getrefcount(vector.backends.object_.VectorObject2D)

    class_refs = None
    for i in range(10):
        a = one(obj)
        assert (sys.getrefcount(obj), sys.getrefcount(obj.azimuthal)) == (2, 2)
        assert (sys.getrefcount(a), sys.getrefcount(a.azimuthal)) == (2, 2)
        if class_refs is None:
            class_refs = sys.getrefcount(vector.backends.object_.VectorObject2D)
        assert class_refs + 1 == sys.getrefcount(vector.backends.object_.VectorObject2D)

    class_refs = None
    for i in range(10):
        a, b = two(obj)
        assert (sys.getrefcount(obj), sys.getrefcount(obj.azimuthal)) == (2, 2)
        assert (sys.getrefcount(a), sys.getrefcount(a.azimuthal), sys.getrefcount(b), sys.getrefcount(b.azimuthal)) == (2, 2, 2, 2)
        if class_refs is None:
            class_refs = sys.getrefcount(vector.backends.object_.VectorObject2D)
        assert class_refs + 1 == sys.getrefcount(vector.backends.object_.VectorObject2D)
