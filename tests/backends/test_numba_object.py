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


import vector.backends.numba_object  # noqa: E402


def test_namedtuples():
    @numba.njit
    def get_x(obj):
        return obj.x

    assert get_x(vector.backends.object_.AzimuthalObjectXY(1, 2.2)) == 1
    assert get_x(vector.backends.object_.AzimuthalObjectXY(1.1, 2)) == 1.1


def test_VectorObjectType():
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

    # These tests just check that the rest of the implementations are sane.

    obj = vector.obj(x=1, y=2)
    out = one(obj)
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)

    obj = vector.obj(px=1, py=2)
    out = one(obj)
    assert isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)

    obj = vector.obj(x=1, y=2, z=3)
    out = one(obj)
    assert isinstance(out, vector.backends.object_.VectorObject3D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(3)

    obj = vector.obj(px=1, py=2, pz=3)
    out = one(obj)
    assert isinstance(out, vector.backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(3)

    obj = vector.obj(x=1, y=2, z=3, t=4)
    out = one(obj)
    assert isinstance(out, vector.backends.object_.VectorObject4D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(3)
    assert out.t == pytest.approx(4)

    obj = vector.obj(px=1, py=2, pz=3, t=4)
    out = one(obj)
    assert isinstance(out, vector.backends.object_.MomentumObject4D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(3)
    assert out.t == pytest.approx(4)


def test_VectorObject_constructor():
    @numba.njit
    def vector_xy():
        return vector.backends.object_.VectorObject2D(
            vector.backends.object_.AzimuthalObjectXY(1, 2.2)
        )

    @numba.njit
    def vector_rhophi():
        return vector.backends.object_.VectorObject2D(
            vector.backends.object_.AzimuthalObjectRhoPhi(1, 2.2)
        )

    @numba.njit
    def momentum_xy():
        return vector.backends.object_.MomentumObject2D(
            vector.backends.object_.AzimuthalObjectXY(1, 2.2)
        )

    @numba.njit
    def momentum_rhophi():
        return vector.backends.object_.MomentumObject2D(
            vector.backends.object_.AzimuthalObjectRhoPhi(1, 2.2)
        )

    @numba.njit
    def vector_xyz():
        return vector.backends.object_.VectorObject3D(
            vector.backends.object_.AzimuthalObjectXY(1, 2.2),
            vector.backends.object_.LongitudinalObjectZ(3),
        )

    @numba.njit
    def momentum_xyz():
        return vector.backends.object_.MomentumObject3D(
            vector.backends.object_.AzimuthalObjectXY(1, 2.2),
            vector.backends.object_.LongitudinalObjectZ(3),
        )

    @numba.njit
    def vector_rhophitheta():
        return vector.backends.object_.VectorObject3D(
            vector.backends.object_.AzimuthalObjectRhoPhi(1, 2.2),
            vector.backends.object_.LongitudinalObjectTheta(3),
        )

    @numba.njit
    def momentum_rhophitheta():
        return vector.backends.object_.MomentumObject3D(
            vector.backends.object_.AzimuthalObjectRhoPhi(1, 2.2),
            vector.backends.object_.LongitudinalObjectTheta(3),
        )

    @numba.njit
    def vector_xyzt():
        return vector.backends.object_.VectorObject4D(
            vector.backends.object_.AzimuthalObjectXY(1, 2.2),
            vector.backends.object_.LongitudinalObjectZ(3),
            vector.backends.object_.TemporalObjectT(4),
        )

    @numba.njit
    def momentum_xyzt():
        return vector.backends.object_.MomentumObject4D(
            vector.backends.object_.AzimuthalObjectXY(1, 2.2),
            vector.backends.object_.LongitudinalObjectZ(3),
            vector.backends.object_.TemporalObjectT(4),
        )

    @numba.njit
    def vector_rhophietatau():
        return vector.backends.object_.VectorObject4D(
            vector.backends.object_.AzimuthalObjectRhoPhi(1, 2.2),
            vector.backends.object_.LongitudinalObjectEta(3),
            vector.backends.object_.TemporalObjectTau(4),
        )

    @numba.njit
    def momentum_rhophietatau():
        return vector.backends.object_.MomentumObject4D(
            vector.backends.object_.AzimuthalObjectRhoPhi(1, 2.2),
            vector.backends.object_.LongitudinalObjectEta(3),
            vector.backends.object_.TemporalObjectTau(4),
        )

    out = vector_xy()
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)

    out = vector_rhophi()
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)

    out = momentum_xy()
    assert isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)

    out = momentum_rhophi()
    assert isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)

    out = vector_xyz()
    assert isinstance(out, vector.backends.object_.VectorObject3D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)
    assert out.z == pytest.approx(3)

    out = momentum_xyz()
    assert isinstance(out, vector.backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)
    assert out.z == pytest.approx(3)

    out = vector_rhophitheta()
    assert isinstance(out, vector.backends.object_.VectorObject3D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)
    assert out.theta == pytest.approx(3)

    out = momentum_rhophitheta()
    assert isinstance(out, vector.backends.object_.MomentumObject3D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)
    assert out.theta == pytest.approx(3)

    out = vector_xyzt()
    assert isinstance(out, vector.backends.object_.VectorObject4D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)
    assert out.z == pytest.approx(3)
    assert out.t == pytest.approx(4)

    out = momentum_xyzt()
    assert isinstance(out, vector.backends.object_.MomentumObject4D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)
    assert out.z == pytest.approx(3)
    assert out.t == pytest.approx(4)

    out = vector_rhophietatau()
    assert isinstance(out, vector.backends.object_.VectorObject4D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)
    assert out.eta == pytest.approx(3)
    assert out.tau == pytest.approx(4)

    out = momentum_rhophietatau()
    assert isinstance(out, vector.backends.object_.MomentumObject4D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)
    assert out.eta == pytest.approx(3)
    assert out.tau == pytest.approx(4)


def test_factory():
    @numba.njit
    def vector_xy():
        return vector.obj(x=2, y=3.3)

    @numba.njit
    def momentum_xy():
        return vector.obj(px=2, py=3.3)

    @numba.njit
    def vector_rhophi():
        return vector.obj(rho=2, phi=3.3)

    @numba.njit
    def momentum_rhophi():
        return vector.obj(pt=2, phi=3.3)

    @numba.njit
    def vector_xyz():
        return vector.obj(x=2, y=3.3, z=5)

    @numba.njit
    def momentum_xyz():
        return vector.obj(x=2, y=3.3, pz=5)

    @numba.njit
    def vector_rhophieta():
        return vector.obj(rho=2, phi=3.3, eta=5)

    @numba.njit
    def momentum_rhophieta():
        return vector.obj(pt=2, phi=3.3, eta=5)

    @numba.njit
    def vector_xyztau():
        return vector.obj(x=2, y=3.3, z=5, tau=10)

    @numba.njit
    def momentum_xyztau():
        return vector.obj(x=2, y=3.3, z=5, m=10)

    @numba.njit
    def vector_rhophizt():
        return vector.obj(rho=2, phi=3.3, z=5, t=10)

    @numba.njit
    def momentum_rhophizt():
        return vector.obj(rho=2, phi=3.3, z=5, energy=10)

    out = vector_xy()
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)

    out = momentum_xy()
    assert isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)

    out = vector_rhophi()
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.rho == pytest.approx(2)
    assert out.phi == pytest.approx(3.3)

    out = momentum_rhophi()
    assert isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.pt == pytest.approx(2)
    assert out.phi == pytest.approx(3.3)

    out = vector_xyz()
    assert isinstance(out, vector.backends.object_.VectorObject3D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)
    assert out.z == pytest.approx(5)

    out = momentum_xyz()
    assert isinstance(out, vector.backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)
    assert out.z == pytest.approx(5)

    out = vector_rhophieta()
    assert isinstance(out, vector.backends.object_.VectorObject3D)
    assert out.rho == pytest.approx(2)
    assert out.phi == pytest.approx(3.3)
    assert out.eta == pytest.approx(5)

    out = momentum_rhophieta()
    assert isinstance(out, vector.backends.object_.MomentumObject3D)
    assert out.pt == pytest.approx(2)
    assert out.phi == pytest.approx(3.3)
    assert out.eta == pytest.approx(5)

    out = vector_xyztau()
    assert isinstance(out, vector.backends.object_.VectorObject4D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)
    assert out.z == pytest.approx(5)
    assert out.tau == pytest.approx(10)

    out = momentum_xyztau()
    assert isinstance(out, vector.backends.object_.MomentumObject4D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)
    assert out.z == pytest.approx(5)
    assert out.tau == pytest.approx(10)

    out = vector_rhophizt()
    assert isinstance(out, vector.backends.object_.VectorObject4D)
    assert out.rho == pytest.approx(2)
    assert out.phi == pytest.approx(3.3)
    assert out.z == pytest.approx(5)
    assert out.t == pytest.approx(10)

    out = momentum_rhophizt()
    assert isinstance(out, vector.backends.object_.MomentumObject4D)
    assert out.pt == pytest.approx(2)
    assert out.phi == pytest.approx(3.3)
    assert out.z == pytest.approx(5)
    assert out.E == pytest.approx(10)


def test_property_float():
    @numba.njit
    def get_x(v):
        return v.x

    @numba.njit
    def get_z(v):
        return v.z

    @numba.njit
    def get_t(v):
        return v.t

    @numba.njit
    def get_Et(v):
        return v.Et

    assert get_x(vector.obj(x=1.1, y=2)) == pytest.approx(1.1)
    assert get_x(vector.obj(px=1.1, py=2)) == pytest.approx(1.1)
    assert get_x(vector.obj(x=1.1, y=2, z=3)) == pytest.approx(1.1)
    assert get_x(vector.obj(px=1.1, py=2, pz=3)) == pytest.approx(1.1)
    assert get_x(vector.obj(x=1.1, y=2, z=3, t=4)) == pytest.approx(1.1)
    assert get_x(vector.obj(px=1.1, py=2, pz=3, E=4)) == pytest.approx(1.1)

    assert get_x(vector.obj(rho=1, phi=0)) == pytest.approx(1)
    assert get_x(vector.obj(rho=1, phi=numpy.pi / 4)) == pytest.approx(
        1 / numpy.sqrt(2)
    )
    assert get_x(vector.obj(rho=1, phi=numpy.pi / 2)) == pytest.approx(0)

    with pytest.raises(numba.TypingError):
        get_z(vector.obj(x=1, y=2))
    assert get_z(vector.obj(x=1, y=2, z=3)) == pytest.approx(3)
    assert get_z(vector.obj(px=1, py=2, pz=3)) == pytest.approx(3)

    with pytest.raises(numba.TypingError):
        get_t(vector.obj(x=1, y=2))
    with pytest.raises(numba.TypingError):
        get_t(vector.obj(x=1, y=2, z=3))
    assert get_t(vector.obj(x=1, y=2, z=3, t=4)) == pytest.approx(4)
    assert get_t(vector.obj(px=1, py=2, pz=3, E=4)) == pytest.approx(4)

    with pytest.raises(numba.TypingError):
        get_Et(vector.obj(x=1, y=2))
    with pytest.raises(numba.TypingError):
        get_Et(vector.obj(x=1, y=2, z=3))
    with pytest.raises(numba.TypingError):
        get_Et(vector.obj(x=1, y=2, z=3, t=4))
    assert get_Et(vector.obj(px=1, py=2, pz=3, E=4)) == pytest.approx(
        numpy.sqrt(4 ** 2 * (1 ** 2 + 2 ** 2) / (1 ** 2 + 2 ** 2 + 3 ** 2))
    )


def test_planar_method_float():
    @numba.njit
    def get_deltaphi(v1, v2):
        return v1.deltaphi(v2)

    assert get_deltaphi(vector.obj(x=1, y=0), vector.obj(x=0, y=1)) == pytest.approx(
        -numpy.pi / 2
    )
    assert get_deltaphi(vector.obj(x=1, y=0), vector.obj(px=0, py=1)) == pytest.approx(
        -numpy.pi / 2
    )
    assert get_deltaphi(
        vector.obj(px=1, py=0), vector.obj(px=0, py=1)
    ) == pytest.approx(-numpy.pi / 2)


def test_spatial_method_float():
    @numba.njit
    def get_deltaeta(v1, v2):
        return v1.deltaeta(v2)

    assert get_deltaeta(
        vector.obj(x=1, y=0, eta=2.5), vector.obj(x=0, y=1, eta=1)
    ) == pytest.approx(1.5)


def test_method_deltaphi():
    @numba.njit
    def get_deltaphi(v1, v2):
        return v1.deltaphi(v2)

    assert get_deltaphi(
        vector.obj(rho=1.1, phi=2.2), vector.obj(rho=3, phi=4)
    ) == pytest.approx(2.2 - 4)

    assert get_deltaphi(
        vector.obj(rho=1.1, phi=2.2), vector.obj(rho=3, phi=4, z=5)
    ) == pytest.approx(2.2 - 4)

    assert get_deltaphi(
        vector.obj(rho=1.1, phi=2.2, z=3.3), vector.obj(rho=3, phi=4)
    ) == pytest.approx(2.2 - 4)

    assert get_deltaphi(
        vector.obj(rho=1.1, phi=2.2, z=3.3), vector.obj(rho=3, phi=4, z=5)
    ) == pytest.approx(2.2 - 4)


def test_method_add():
    @numba.njit
    def get_add(v1, v2):
        return v1.add(v2)

    out = get_add(vector.obj(x=1.1, y=2.2), vector.obj(x=3, y=4))
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2), vector.obj(x=3, y=4, z=5))
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2, z=3.3), vector.obj(x=3, y=4))
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2, z=3.3), vector.obj(x=3, y=4, z=5))
    assert isinstance(out, vector.backends.object_.VectorObject3D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)
    assert out.z == pytest.approx(8.3)

    out = get_add(vector.obj(px=1.1, py=2.2), vector.obj(px=3, py=4))
    assert isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2), vector.obj(px=3, py=4, pz=5))
    assert isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2, pz=3.3), vector.obj(px=3, py=4))
    assert isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2, pz=3.3), vector.obj(px=3, py=4, pz=5))
    assert isinstance(out, vector.backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)
    assert out.z == pytest.approx(8.3)

    out = get_add(vector.obj(x=1.1, y=2.2), vector.obj(px=3, py=4))
    assert not isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2), vector.obj(px=3, py=4, pz=5))
    assert not isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2, z=3.3), vector.obj(px=3, py=4))
    assert not isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2, z=3.3), vector.obj(px=3, py=4, pz=5))
    assert not isinstance(out, vector.backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)
    assert out.z == pytest.approx(8.3)

    out = get_add(vector.obj(px=1.1, py=2.2), vector.obj(x=3, y=4))
    assert not isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2), vector.obj(x=3, y=4, z=5))
    assert not isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2, pz=3.3), vector.obj(x=3, y=4))
    assert not isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2, pz=3.3), vector.obj(x=3, y=4, z=5))
    assert not isinstance(out, vector.backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)
    assert out.z == pytest.approx(8.3)


def test_method_isparallel():
    @numba.njit
    def get_isparallel(v1, v2):
        return v1.is_parallel(v2)

    assert get_isparallel(vector.obj(px=1.1, py=2.2), vector.obj(px=2.2, py=4.4))

    assert get_isparallel(
        vector.obj(px=1.1, py=2.2, pz=3.3), vector.obj(px=2.2, py=4.4, pz=6.6)
    )


def test_method_rotateZ():
    @numba.njit
    def get_rotateZ(v, angle):
        return v.rotateZ(angle)

    out = get_rotateZ(vector.obj(x=1, y=0), 0.1)
    assert isinstance(out, vector.backends.object_.VectorObject2D)
    assert out.x == pytest.approx(0.9950041652780258)
    assert out.y == pytest.approx(0.09983341664682815)

    out = get_rotateZ(vector.obj(px=1, py=0), 0.1)
    assert isinstance(out, vector.backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(0.9950041652780258)
    assert out.y == pytest.approx(0.09983341664682815)
