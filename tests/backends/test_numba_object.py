# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import sys

import numpy
import pytest

import vector
import vector._backends.object_

numba = pytest.importorskip("numba")


import vector._backends.numba_object  # noqa: E402

pytestmark = pytest.mark.numba


def test_namedtuples():
    @numba.njit
    def get_x(obj):
        return obj.x

    assert get_x(vector._backends.object_.AzimuthalObjectXY(1, 2.2)) == 1
    assert get_x(vector._backends.object_.AzimuthalObjectXY(1.1, 2)) == 1.1


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
            class_refs = sys.getrefcount(vector._backends.object_.VectorObject2D)
        assert class_refs + 1 == sys.getrefcount(
            vector._backends.object_.VectorObject2D
        )

    class_refs = None
    for _ in range(10):
        a = one(obj)
        assert (sys.getrefcount(obj), sys.getrefcount(obj.azimuthal)) == (2, 2)
        assert (sys.getrefcount(a), sys.getrefcount(a.azimuthal)) == (2, 2)
        if class_refs is None:
            class_refs = sys.getrefcount(vector._backends.object_.VectorObject2D)
        assert class_refs + 1 == sys.getrefcount(
            vector._backends.object_.VectorObject2D
        )

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
            class_refs = sys.getrefcount(vector._backends.object_.VectorObject2D)
        assert class_refs + 1 == sys.getrefcount(
            vector._backends.object_.VectorObject2D
        )

    # These tests just check that the rest of the implementations are sane.

    obj = vector.obj(x=1, y=2)
    out = one(obj)
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)

    obj = vector.obj(px=1, py=2)
    out = one(obj)
    assert isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)

    obj = vector.obj(x=1, y=2, z=3)
    out = one(obj)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(3)

    obj = vector.obj(px=1, py=2, pz=3)
    out = one(obj)
    assert isinstance(out, vector._backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(3)

    obj = vector.obj(x=1, y=2, z=3, t=4)
    out = one(obj)
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(3)
    assert out.t == pytest.approx(4)

    obj = vector.obj(px=1, py=2, pz=3, t=4)
    out = one(obj)
    assert isinstance(out, vector._backends.object_.MomentumObject4D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(3)
    assert out.t == pytest.approx(4)


def test_VectorObject_constructor():
    @numba.njit
    def vector_xy():
        return vector._backends.object_.VectorObject2D(
            vector._backends.object_.AzimuthalObjectXY(1, 2.2)
        )

    @numba.njit
    def vector_rhophi():
        return vector._backends.object_.VectorObject2D(
            vector._backends.object_.AzimuthalObjectRhoPhi(1, 2.2)
        )

    @numba.njit
    def momentum_xy():
        return vector._backends.object_.MomentumObject2D(
            vector._backends.object_.AzimuthalObjectXY(1, 2.2)
        )

    @numba.njit
    def momentum_rhophi():
        return vector._backends.object_.MomentumObject2D(
            vector._backends.object_.AzimuthalObjectRhoPhi(1, 2.2)
        )

    @numba.njit
    def vector_xyz():
        return vector._backends.object_.VectorObject3D(
            vector._backends.object_.AzimuthalObjectXY(1, 2.2),
            vector._backends.object_.LongitudinalObjectZ(3),
        )

    @numba.njit
    def momentum_xyz():
        return vector._backends.object_.MomentumObject3D(
            vector._backends.object_.AzimuthalObjectXY(1, 2.2),
            vector._backends.object_.LongitudinalObjectZ(3),
        )

    @numba.njit
    def vector_rhophitheta():
        return vector._backends.object_.VectorObject3D(
            vector._backends.object_.AzimuthalObjectRhoPhi(1, 2.2),
            vector._backends.object_.LongitudinalObjectTheta(3),
        )

    @numba.njit
    def momentum_rhophitheta():
        return vector._backends.object_.MomentumObject3D(
            vector._backends.object_.AzimuthalObjectRhoPhi(1, 2.2),
            vector._backends.object_.LongitudinalObjectTheta(3),
        )

    @numba.njit
    def vector_xyzt():
        return vector._backends.object_.VectorObject4D(
            vector._backends.object_.AzimuthalObjectXY(1, 2.2),
            vector._backends.object_.LongitudinalObjectZ(3),
            vector._backends.object_.TemporalObjectT(4),
        )

    @numba.njit
    def momentum_xyzt():
        return vector._backends.object_.MomentumObject4D(
            vector._backends.object_.AzimuthalObjectXY(1, 2.2),
            vector._backends.object_.LongitudinalObjectZ(3),
            vector._backends.object_.TemporalObjectT(4),
        )

    @numba.njit
    def vector_rhophietatau():
        return vector._backends.object_.VectorObject4D(
            vector._backends.object_.AzimuthalObjectRhoPhi(1, 2.2),
            vector._backends.object_.LongitudinalObjectEta(3),
            vector._backends.object_.TemporalObjectTau(4),
        )

    @numba.njit
    def momentum_rhophietatau():
        return vector._backends.object_.MomentumObject4D(
            vector._backends.object_.AzimuthalObjectRhoPhi(1, 2.2),
            vector._backends.object_.LongitudinalObjectEta(3),
            vector._backends.object_.TemporalObjectTau(4),
        )

    out = vector_xy()
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)

    out = vector_rhophi()
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)

    out = momentum_xy()
    assert isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)

    out = momentum_rhophi()
    assert isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)

    out = vector_xyz()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)
    assert out.z == pytest.approx(3)

    out = momentum_xyz()
    assert isinstance(out, vector._backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)
    assert out.z == pytest.approx(3)

    out = vector_rhophitheta()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)
    assert out.theta == pytest.approx(3)

    out = momentum_rhophitheta()
    assert isinstance(out, vector._backends.object_.MomentumObject3D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)
    assert out.theta == pytest.approx(3)

    out = vector_xyzt()
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)
    assert out.z == pytest.approx(3)
    assert out.t == pytest.approx(4)

    out = momentum_xyzt()
    assert isinstance(out, vector._backends.object_.MomentumObject4D)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2.2)
    assert out.z == pytest.approx(3)
    assert out.t == pytest.approx(4)

    out = vector_rhophietatau()
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)
    assert out.eta == pytest.approx(3)
    assert out.tau == pytest.approx(4)

    out = momentum_rhophietatau()
    assert isinstance(out, vector._backends.object_.MomentumObject4D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(2.2)
    assert out.eta == pytest.approx(3)
    assert out.tau == pytest.approx(4)


def test_projections():
    @numba.njit
    def to_Vector2D(x):
        return x.to_Vector2D()

    @numba.njit
    def to_Vector3D(x):
        return x.to_Vector3D()

    @numba.njit
    def to_Vector4D(x):
        return x.to_Vector4D()

    assert isinstance(
        to_Vector2D(vector.obj(x=1.1, y=2.2)), vector._backends.object_.VectorObject2D
    )
    assert isinstance(
        to_Vector2D(vector.obj(x=1.1, y=2.2, z=3.3)),
        vector._backends.object_.VectorObject2D,
    )
    assert isinstance(
        to_Vector2D(vector.obj(x=1.1, y=2.2, z=3.3, t=4.4)),
        vector._backends.object_.VectorObject2D,
    )
    assert isinstance(
        to_Vector2D(vector.obj(px=1.1, py=2.2)),
        vector._backends.object_.MomentumObject2D,
    )
    assert isinstance(
        to_Vector2D(vector.obj(px=1.1, py=2.2, pz=3.3)),
        vector._backends.object_.MomentumObject2D,
    )
    assert isinstance(
        to_Vector2D(vector.obj(px=1.1, py=2.2, pz=3.3, E=4.4)),
        vector._backends.object_.MomentumObject2D,
    )

    assert isinstance(
        to_Vector3D(vector.obj(x=1.1, y=2.2)), vector._backends.object_.VectorObject3D
    )
    assert isinstance(
        to_Vector3D(vector.obj(x=1.1, y=2.2, z=3.3)),
        vector._backends.object_.VectorObject3D,
    )
    assert isinstance(
        to_Vector3D(vector.obj(x=1.1, y=2.2, z=3.3, t=4.4)),
        vector._backends.object_.VectorObject3D,
    )
    assert isinstance(
        to_Vector3D(vector.obj(px=1.1, py=2.2)),
        vector._backends.object_.MomentumObject3D,
    )
    assert isinstance(
        to_Vector3D(vector.obj(px=1.1, py=2.2, pz=3.3)),
        vector._backends.object_.MomentumObject3D,
    )
    assert isinstance(
        to_Vector3D(vector.obj(px=1.1, py=2.2, pz=3.3, E=4.4)),
        vector._backends.object_.MomentumObject3D,
    )

    assert isinstance(
        to_Vector4D(vector.obj(x=1.1, y=2.2)), vector._backends.object_.VectorObject4D
    )
    assert isinstance(
        to_Vector4D(vector.obj(x=1.1, y=2.2, z=3.3)),
        vector._backends.object_.VectorObject4D,
    )
    assert isinstance(
        to_Vector4D(vector.obj(x=1.1, y=2.2, z=3.3, t=4.4)),
        vector._backends.object_.VectorObject4D,
    )
    assert isinstance(
        to_Vector4D(vector.obj(px=1.1, py=2.2)),
        vector._backends.object_.MomentumObject4D,
    )
    assert isinstance(
        to_Vector4D(vector.obj(px=1.1, py=2.2, pz=3.3)),
        vector._backends.object_.MomentumObject4D,
    )
    assert isinstance(
        to_Vector4D(vector.obj(px=1.1, py=2.2, pz=3.3, E=4.4)),
        vector._backends.object_.MomentumObject4D,
    )


def test_conversions():
    @numba.njit
    def to_xy(x):
        return x.to_xy()

    @numba.njit
    def to_rhophi(x):
        return x.to_rhophi()

    @numba.njit
    def to_xyz(x):
        return x.to_xyz()

    @numba.njit
    def to_rhophiz(x):
        return x.to_rhophiz()

    @numba.njit
    def to_xytheta(x):
        return x.to_xytheta()

    @numba.njit
    def to_rhophitheta(x):
        return x.to_rhophitheta()

    @numba.njit
    def to_xyeta(x):
        return x.to_xyeta()

    @numba.njit
    def to_rhophieta(x):
        return x.to_rhophieta()

    @numba.njit
    def to_xyzt(x):
        return x.to_xyzt()

    @numba.njit
    def to_rhophizt(x):
        return x.to_rhophizt()

    @numba.njit
    def to_xythetat(x):
        return x.to_xythetat()

    @numba.njit
    def to_rhophithetat(x):
        return x.to_rhophithetat()

    @numba.njit
    def to_xyetat(x):
        return x.to_xyetat()

    @numba.njit
    def to_rhophietat(x):
        return x.to_rhophietat()

    @numba.njit
    def to_xyztau(x):
        return x.to_xyztau()

    @numba.njit
    def to_rhophiztau(x):
        return x.to_rhophiztau()

    @numba.njit
    def to_xythetatau(x):
        return x.to_xythetatau()

    @numba.njit
    def to_rhophithetatau(x):
        return x.to_rhophithetatau()

    @numba.njit
    def to_xyetatau(x):
        return x.to_xyetatau()

    @numba.njit
    def to_rhophietatau(x):
        return x.to_rhophietatau()

    for v in (
        vector.obj(x=1.1, y=2.2),
        vector.obj(px=1.1, py=2.2),
        vector.obj(x=1.1, y=2.2, z=3.3),
        vector.obj(px=1.1, py=2.2, pz=3.3),
        vector.obj(x=1.1, y=2.2, z=3.3, t=4.4),
        vector.obj(px=1.1, py=2.2, pz=3.3, E=4.4),
    ):
        print(v)

        out = to_xy(v)
        assert isinstance(out, vector._backends.object_.VectorObject2D)
        if isinstance(v, vector._methods.Momentum):
            assert isinstance(out, vector._backends.object_.MomentumObject2D)
        assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
        assert out.x == pytest.approx(1.1)
        assert out.y == pytest.approx(2.2)

        out = to_rhophi(v)
        assert isinstance(out, vector._backends.object_.VectorObject2D)
        if isinstance(v, vector._methods.Momentum):
            assert isinstance(out, vector._backends.object_.MomentumObject2D)
        assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectRhoPhi)
        assert out.x == pytest.approx(1.1)
        assert out.y == pytest.approx(2.2)

        out = to_xyz(v)
        assert isinstance(out, vector._backends.object_.VectorObject3D)
        if isinstance(v, vector._methods.Momentum):
            assert isinstance(out, vector._backends.object_.MomentumObject3D)
        assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
        assert isinstance(
            out.longitudinal, vector._backends.object_.LongitudinalObjectZ
        )
        assert out.x == pytest.approx(1.1)
        assert out.y == pytest.approx(2.2)
        if isinstance(v, vector._backends.object_.VectorObject2D):
            assert out.z == pytest.approx(0)
        else:
            assert out.z == pytest.approx(3.3)

        out = to_rhophiz(v)
        assert isinstance(out, vector._backends.object_.VectorObject3D)
        if isinstance(v, vector._methods.Momentum):
            assert isinstance(out, vector._backends.object_.MomentumObject3D)
        assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectRhoPhi)
        assert isinstance(
            out.longitudinal, vector._backends.object_.LongitudinalObjectZ
        )
        assert out.x == pytest.approx(1.1)
        assert out.y == pytest.approx(2.2)
        if isinstance(v, vector._backends.object_.VectorObject2D):
            assert out.z == pytest.approx(0)
        else:
            assert out.z == pytest.approx(3.3)

        out = to_xytheta(v)
        assert isinstance(out, vector._backends.object_.VectorObject3D)
        if isinstance(v, vector._methods.Momentum):
            assert isinstance(out, vector._backends.object_.MomentumObject3D)
        assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
        assert isinstance(
            out.longitudinal, vector._backends.object_.LongitudinalObjectTheta
        )
        assert out.x == pytest.approx(1.1)
        assert out.y == pytest.approx(2.2)
        if isinstance(v, vector._backends.object_.VectorObject2D):
            assert out.theta == pytest.approx(0)
        else:
            assert out.z == pytest.approx(3.3)

        out = to_rhophitheta(v)
        assert isinstance(out, vector._backends.object_.VectorObject3D)
        if isinstance(v, vector._methods.Momentum):
            assert isinstance(out, vector._backends.object_.MomentumObject3D)
        assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectRhoPhi)
        assert isinstance(
            out.longitudinal, vector._backends.object_.LongitudinalObjectTheta
        )
        assert out.x == pytest.approx(1.1)
        assert out.y == pytest.approx(2.2)
        if isinstance(v, vector._backends.object_.VectorObject2D):
            assert out.theta == pytest.approx(0)
        else:
            assert out.z == pytest.approx(3.3)

        out = to_xyeta(v)
        assert isinstance(out, vector._backends.object_.VectorObject3D)
        if isinstance(v, vector._methods.Momentum):
            assert isinstance(out, vector._backends.object_.MomentumObject3D)
        assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
        assert isinstance(
            out.longitudinal, vector._backends.object_.LongitudinalObjectEta
        )
        assert out.x == pytest.approx(1.1)
        assert out.y == pytest.approx(2.2)
        if isinstance(v, vector._backends.object_.VectorObject2D):
            assert out.eta == pytest.approx(0)
        else:
            assert out.z == pytest.approx(3.3)

        out = to_rhophietatau(v)
        assert isinstance(out, vector._backends.object_.VectorObject4D)
        if isinstance(v, vector._methods.Momentum):
            assert isinstance(out, vector._backends.object_.MomentumObject4D)
        assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectRhoPhi)
        assert isinstance(
            out.longitudinal, vector._backends.object_.LongitudinalObjectEta
        )
        assert isinstance(out.temporal, vector._backends.object_.TemporalObjectTau)
        assert out.x == pytest.approx(1.1)
        assert out.y == pytest.approx(2.2)
        if isinstance(v, vector._backends.object_.VectorObject2D):
            assert out.eta == pytest.approx(0)
        else:
            assert out.z == pytest.approx(3.3)
        if isinstance(
            v,
            (
                vector._backends.object_.VectorObject2D,
                vector._backends.object_.VectorObject3D,
            ),
        ):
            assert out.tau == pytest.approx(0)
        else:
            assert out.t == pytest.approx(4.4)


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
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)

    out = momentum_xy()
    assert isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)

    out = vector_rhophi()
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.rho == pytest.approx(2)
    assert out.phi == pytest.approx(3.3)

    out = momentum_rhophi()
    assert isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.pt == pytest.approx(2)
    assert out.phi == pytest.approx(3.3)

    out = vector_xyz()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)
    assert out.z == pytest.approx(5)

    out = momentum_xyz()
    assert isinstance(out, vector._backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)
    assert out.z == pytest.approx(5)

    out = vector_rhophieta()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.rho == pytest.approx(2)
    assert out.phi == pytest.approx(3.3)
    assert out.eta == pytest.approx(5)

    out = momentum_rhophieta()
    assert isinstance(out, vector._backends.object_.MomentumObject3D)
    assert out.pt == pytest.approx(2)
    assert out.phi == pytest.approx(3.3)
    assert out.eta == pytest.approx(5)

    out = vector_xyztau()
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)
    assert out.z == pytest.approx(5)
    assert out.tau == pytest.approx(10)

    out = momentum_xyztau()
    assert isinstance(out, vector._backends.object_.MomentumObject4D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(3.3)
    assert out.z == pytest.approx(5)
    assert out.tau == pytest.approx(10)

    out = vector_rhophizt()
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.rho == pytest.approx(2)
    assert out.phi == pytest.approx(3.3)
    assert out.z == pytest.approx(5)
    assert out.t == pytest.approx(10)

    out = momentum_rhophizt()
    assert isinstance(out, vector._backends.object_.MomentumObject4D)
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
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2), vector.obj(x=3, y=4, z=5))
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2, z=3.3), vector.obj(x=3, y=4))
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2, z=3.3), vector.obj(x=3, y=4, z=5))
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)
    assert out.z == pytest.approx(8.3)

    out = get_add(vector.obj(px=1.1, py=2.2), vector.obj(px=3, py=4))
    assert isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2), vector.obj(px=3, py=4, pz=5))
    assert isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2, pz=3.3), vector.obj(px=3, py=4))
    assert isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2, pz=3.3), vector.obj(px=3, py=4, pz=5))
    assert isinstance(out, vector._backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)
    assert out.z == pytest.approx(8.3)

    out = get_add(vector.obj(x=1.1, y=2.2), vector.obj(px=3, py=4))
    assert not isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2), vector.obj(px=3, py=4, pz=5))
    assert not isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2, z=3.3), vector.obj(px=3, py=4))
    assert not isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(x=1.1, y=2.2, z=3.3), vector.obj(px=3, py=4, pz=5))
    assert not isinstance(out, vector._backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)
    assert out.z == pytest.approx(8.3)

    out = get_add(vector.obj(px=1.1, py=2.2), vector.obj(x=3, y=4))
    assert not isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2), vector.obj(x=3, y=4, z=5))
    assert not isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2, pz=3.3), vector.obj(x=3, y=4))
    assert not isinstance(out, vector._backends.object_.MomentumObject2D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)

    out = get_add(vector.obj(px=1.1, py=2.2, pz=3.3), vector.obj(x=3, y=4, z=5))
    assert not isinstance(out, vector._backends.object_.MomentumObject3D)
    assert out.x == pytest.approx(4.1)
    assert out.y == pytest.approx(6.2)
    assert out.z == pytest.approx(8.3)


def test_method_isparallel():
    @numba.njit
    def get_isparallel(v1, v2):
        return v1.is_parallel(v2)

    assert get_isparallel(vector.obj(x=1.1, y=2.2), vector.obj(x=2.2, y=4.4))

    assert get_isparallel(vector.obj(x=1.1, y=2.2), vector.obj(px=2.2, py=4.4))

    assert get_isparallel(vector.obj(px=1.1, py=2.2), vector.obj(px=2.2, py=4.4))

    assert get_isparallel(vector.obj(x=1.1, y=2.2), vector.obj(x=2.2, y=4.4, z=0.0))

    assert get_isparallel(vector.obj(x=1.1, y=2.2, z=0.0), vector.obj(x=2.2, y=4.4))

    assert get_isparallel(
        vector.obj(x=1.1, y=2.2, z=3.3), vector.obj(x=2.2, y=4.4, z=6.6)
    )


def test_method_isclose():
    @numba.njit
    def get_isclose(v1, v2):
        return v1.isclose(v2)

    assert get_isclose(vector.obj(x=1.1, y=2.2), vector.obj(x=1.1, y=2.2))

    assert get_isclose(vector.obj(x=1.1, y=2.2, z=3.3), vector.obj(x=1.1, y=2.2, z=3.3))

    assert get_isclose(
        vector.obj(x=1.1, y=2.2, z=3.3, t=4.4), vector.obj(x=1.1, y=2.2, z=3.3, t=4.4)
    )


def test_method_rotateZ():
    @numba.njit
    def get_rotateZ(v, angle):
        return v.rotateZ(angle)

    out = get_rotateZ(vector.obj(x=1, y=0), 0.1)
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(0.9950041652780258)
    assert out.y == pytest.approx(0.09983341664682815)

    out = get_rotateZ(vector.obj(rho=1, phi=0), 0.1)
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(0.1)

    out = get_rotateZ(vector.obj(x=1, y=0, z=2.2), 0.1)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(0.9950041652780258)
    assert out.y == pytest.approx(0.09983341664682815)
    assert out.z == pytest.approx(2.2)


def test_method_transform2D():
    @numba.njit
    def get_transform2D(v, obj):
        return v.transform2D(obj)

    obj = numba.typed.Dict()
    obj["xx"] = numpy.cos(0.1)
    obj["xy"] = -numpy.sin(0.1)
    obj["yx"] = numpy.sin(0.1)
    obj["yy"] = numpy.cos(0.1)

    out = get_transform2D(vector.obj(x=1, y=0), obj)
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(0.9950041652780258)
    assert out.y == pytest.approx(0.09983341664682815)

    out = get_transform2D(vector.obj(rho=1, phi=0), obj)
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.rho == pytest.approx(1)
    assert out.phi == pytest.approx(0.1)

    out = get_transform2D(vector.obj(x=1, y=0, z=2.2), obj)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(0.9950041652780258)
    assert out.y == pytest.approx(0.09983341664682815)
    assert out.z == pytest.approx(2.2)


def test_method_unit():
    @numba.njit
    def get_unit(v):
        return v.unit()

    out = get_unit(vector.obj(x=1, y=1))
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(1 / numpy.sqrt(2))
    assert out.y == pytest.approx(1 / numpy.sqrt(2))

    out = get_unit(vector.obj(x=1, y=1, z=1))
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(1 / numpy.sqrt(3))
    assert out.y == pytest.approx(1 / numpy.sqrt(3))
    assert out.z == pytest.approx(1 / numpy.sqrt(3))

    out = get_unit(vector.obj(x=1, y=1, z=1, t=1))
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.x == pytest.approx(1 / numpy.sqrt(2))
    assert out.y == pytest.approx(1 / numpy.sqrt(2))
    assert out.z == pytest.approx(1 / numpy.sqrt(2))
    assert out.t == pytest.approx(1 / numpy.sqrt(2))


def test_method_scale():
    @numba.njit
    def get_scale(v):
        return v.scale(2)

    out = get_scale(vector.obj(x=1.1, y=2.2))
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(2.2)
    assert out.y == pytest.approx(4.4)

    out = get_scale(vector.obj(x=1.1, y=2.2, z=3.3))
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(2.2)
    assert out.y == pytest.approx(4.4)
    assert out.z == pytest.approx(6.6)

    out = get_scale(vector.obj(x=1.1, y=2.2, z=3.3, t=4.4))
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.x == pytest.approx(2.2)
    assert out.y == pytest.approx(4.4)
    assert out.z == pytest.approx(6.6)
    assert out.t == pytest.approx(8.8)


def test_method_cross():
    @numba.njit
    def get_cross(v1, v2):
        return v1.cross(v2)

    out = get_cross(vector.obj(x=0.1, y=0.2, z=0.3), vector.obj(x=0.4, y=0.5, z=0.6))
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
    assert isinstance(out.longitudinal, vector._backends.object_.LongitudinalObjectZ)
    assert (out.x, out.y, out.z) == pytest.approx((-0.03, 0.06, -0.03))

    out = get_cross(
        vector.obj(x=0.1, y=0.2, z=0.3, t=999), vector.obj(x=0.4, y=0.5, z=0.6)
    )
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
    assert isinstance(out.longitudinal, vector._backends.object_.LongitudinalObjectZ)
    assert (out.x, out.y, out.z) == pytest.approx((-0.03, 0.06, -0.03))

    out = get_cross(
        vector.obj(x=0.1, y=0.2, z=0.3), vector.obj(x=0.4, y=0.5, z=0.6, t=999)
    )
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
    assert isinstance(out.longitudinal, vector._backends.object_.LongitudinalObjectZ)
    assert (out.x, out.y, out.z) == pytest.approx((-0.03, 0.06, -0.03))

    out = get_cross(
        vector.obj(x=0.1, y=0.2, z=0.3, t=999), vector.obj(x=0.4, y=0.5, z=0.6, t=999)
    )
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
    assert isinstance(out.longitudinal, vector._backends.object_.LongitudinalObjectZ)
    assert (out.x, out.y, out.z) == pytest.approx((-0.03, 0.06, -0.03))


def test_method_rotateX():
    @numba.njit
    def get_rotateX(v):
        return v.rotateX(0.25)

    out = get_rotateX(vector.obj(x=0.1, y=0.2, z=0.3))
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(0.1)
    assert out.y == pytest.approx(0.1195612965657721)
    assert out.z == pytest.approx(0.340154518364098)

    out = get_rotateX(vector.obj(x=0.1, y=0.2, z=0.3, t=0.4))
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.x == pytest.approx(0.1)
    assert out.y == pytest.approx(0.1195612965657721)
    assert out.z == pytest.approx(0.340154518364098)
    assert out.t == pytest.approx(0.4)


def test_method_rotateY():
    @numba.njit
    def get_rotateY(v):
        return v.rotateY(0.25)

    out = get_rotateY(vector.obj(x=0.1, y=0.2, z=0.3))
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(0.17111242994742137)
    assert out.y == pytest.approx(0.2)
    assert out.z == pytest.approx(0.2659333305877411)

    out = get_rotateY(vector.obj(x=0.1, y=0.2, z=0.3, t=0.4))
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.x == pytest.approx(0.17111242994742137)
    assert out.y == pytest.approx(0.2)
    assert out.z == pytest.approx(0.2659333305877411)
    assert out.t == pytest.approx(0.4)


def test_method_rotate_axis():
    @numba.njit
    def get_rotate_axis(vec, axis):
        return vec.rotate_axis(axis, 0.25)

    axis = vector.obj(x=0.1, y=0.2, z=0.3)
    vec = vector.obj(x=0.4, y=0.5, z=0.6)
    out = get_rotate_axis(vec, axis)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(0.37483425404335763)
    assert out.y == pytest.approx(0.5383405688588193)
    assert out.z == pytest.approx(0.5828282027463345)

    axis = vector.obj(x=0.1, y=0.2, z=0.3)
    vec = vector.obj(x=0.4, y=0.5, z=0.6, t=999)
    out = get_rotate_axis(vec, axis)
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.x == pytest.approx(0.37483425404335763)
    assert out.y == pytest.approx(0.5383405688588193)
    assert out.z == pytest.approx(0.5828282027463345)
    assert out.t == pytest.approx(999)

    axis = vector.obj(x=0.1, y=0.2, z=0.3, t=999)
    vec = vector.obj(x=0.4, y=0.5, z=0.6)
    out = get_rotate_axis(vec, axis)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(0.37483425404335763)
    assert out.y == pytest.approx(0.5383405688588193)
    assert out.z == pytest.approx(0.5828282027463345)

    axis = vector.obj(x=0.1, y=0.2, z=0.3, t=999)
    vec = vector.obj(x=0.4, y=0.5, z=0.6, t=999)
    out = get_rotate_axis(vec, axis)
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.x == pytest.approx(0.37483425404335763)
    assert out.y == pytest.approx(0.5383405688588193)
    assert out.z == pytest.approx(0.5828282027463345)
    assert out.t == pytest.approx(999)


def test_method_rotate_euler():
    @numba.njit
    def get_rotate_euler(vec, phi, theta, psi):
        return vec.rotate_euler(phi, theta, psi, order="zxz")

    vec = vector.obj(x=0.4, y=0.5, z=0.6)
    out = get_rotate_euler(vec, 0.1, 0.2, 0.3)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(0.5956646364506655)
    assert out.y == pytest.approx(0.409927258162962)
    assert out.z == pytest.approx(0.4971350761081869)


def test_method_rotate_quaternion():
    @numba.njit
    def get_rotate_quaternion(vec, u, i, j, k):
        return vec.rotate_quaternion(u, i, j, k)

    vec = vector.obj(x=0.5, y=0.6, z=0.7)
    out = get_rotate_quaternion(vec, 0.1, 0.2, 0.3, 0.4)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(0.078)
    assert out.y == pytest.approx(0.18)
    assert out.z == pytest.approx(0.246)


def test_method_transform3D():
    @numba.njit
    def get_transform3D(v, obj):
        return v.transform3D(obj)

    obj = numba.typed.Dict()
    obj["xx"] = numpy.cos(0.1)
    obj["xy"] = -numpy.sin(0.1)
    obj["xz"] = 0
    obj["yx"] = numpy.sin(0.1)
    obj["yy"] = numpy.cos(0.1)
    obj["yz"] = 0
    obj["zx"] = 0
    obj["zy"] = 0
    obj["zz"] = 1

    out = get_transform3D(vector.obj(x=1, y=0, z=99), obj)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(0.9950041652780258)
    assert out.y == pytest.approx(0.09983341664682815)
    assert out.z == pytest.approx(99)


def test_method_boost():
    @numba.njit
    def get_boost_p4(vec, p4):
        return vec.boost_p4(p4)

    @numba.njit
    def get_boost_beta3(vec, beta3):
        return vec.boost_beta3(beta3)

    @numba.njit
    def get_boost(vec, booster):
        return vec.boost(booster)

    out = get_boost_p4(vector.obj(x=1, y=2, z=3, t=4), vector.obj(x=5, y=6, z=7, t=15))
    assert out.x == pytest.approx(3.5537720741941676)
    assert out.y == pytest.approx(5.0645264890330015)
    assert out.z == pytest.approx(6.575280903871835)
    assert out.t == pytest.approx(9.138547120755076)

    out = get_boost_beta3(
        vector.obj(x=1, y=2, z=3, t=4), vector.obj(x=5 / 15, y=6 / 15, z=7 / 15)
    )
    assert out.x == pytest.approx(3.5537720741941676)
    assert out.y == pytest.approx(5.0645264890330015)
    assert out.z == pytest.approx(6.575280903871835)
    assert out.t == pytest.approx(9.138547120755076)

    out = get_boost(vector.obj(x=1, y=2, z=3, t=4), vector.obj(x=5, y=6, z=7, t=15))
    assert out.x == pytest.approx(3.5537720741941676)
    assert out.y == pytest.approx(5.0645264890330015)
    assert out.z == pytest.approx(6.575280903871835)
    assert out.t == pytest.approx(9.138547120755076)

    out = get_boost(
        vector.obj(x=1, y=2, z=3, t=4), vector.obj(x=5 / 15, y=6 / 15, z=7 / 15)
    )
    assert out.x == pytest.approx(3.5537720741941676)
    assert out.y == pytest.approx(5.0645264890330015)
    assert out.z == pytest.approx(6.575280903871835)
    assert out.t == pytest.approx(9.138547120755076)


def test_method_boostX():
    @numba.njit
    def get_boostX_beta(vec, beta):
        return vec.boostX(beta=beta)

    @numba.njit
    def get_boostX_gamma(vec, gamma):
        return vec.boostX(gamma=gamma)

    out = get_boostX_beta(vector.obj(x=3, y=2, z=1, t=4), -0.9428090415820634)
    assert out.x == pytest.approx(-2.313708498984761)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(1)
    assert out.t == pytest.approx(3.5147186257614287)

    out = get_boostX_gamma(vector.obj(x=3, y=2, z=1, t=4), -3)
    assert out.x == pytest.approx(-2.313708498984761)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(1)
    assert out.t == pytest.approx(3.5147186257614287)


def test_method_to_beta3():
    @numba.njit
    def get_to_beta3(vec):
        return vec.to_beta3()

    out = get_to_beta3(vector.obj(x=3, y=4, z=10, t=20))
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_method_transform4D():
    @numba.njit
    def get_transform4D(v, obj):
        return v.transform4D(obj)

    obj = numba.typed.Dict()
    obj["xx"] = numpy.cos(0.1)
    obj["xy"] = -numpy.sin(0.1)
    obj["xz"] = 0
    obj["xt"] = 0
    obj["yx"] = numpy.sin(0.1)
    obj["yy"] = numpy.cos(0.1)
    obj["yz"] = 0
    obj["yt"] = 0
    obj["zx"] = 0
    obj["zy"] = 0
    obj["zz"] = 1
    obj["zt"] = 0
    obj["tx"] = 0
    obj["ty"] = 0
    obj["tz"] = 0
    obj["tt"] = 1

    out = get_transform4D(vector.obj(x=1, y=0, z=99, t=123), obj)
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert out.x == pytest.approx(0.9950041652780258)
    assert out.y == pytest.approx(0.09983341664682815)
    assert out.z == pytest.approx(99)
    assert out.t == pytest.approx(123)


def test_method_timespacelight_like():
    @numba.njit
    def get_is_timelike(v):
        return v.is_timelike()

    @numba.njit
    def get_is_spacelike(v):
        return v.is_spacelike()

    @numba.njit
    def get_is_lightlike(v):
        return v.is_lightlike()

    assert get_is_timelike(vector.obj(x=3, y=4, z=0, t=10))
    assert get_is_lightlike(vector.obj(x=3, y=4, z=0, t=5))
    assert get_is_spacelike(vector.obj(x=3, y=4, z=0, t=2))


def test_momentum_alias():
    @numba.njit
    def get_px(v):
        return v.px

    assert get_px(vector.obj(px=3, py=4)) == pytest.approx(3)
    with pytest.raises(numba.TypingError):
        get_px(vector.obj(x=3, y=4))


def test_operator_abs():
    @numba.njit
    def get_abs(v):
        return abs(v)

    assert get_abs(vector.obj(x=3, y=4)) == pytest.approx(5)

    assert get_abs(vector.obj(x=3, y=4, z=5)) == pytest.approx(numpy.sqrt(50))

    assert get_abs(vector.obj(x=3, y=4, z=5, tau=100)) == pytest.approx(100)


def test_operator_neg():
    @numba.njit
    def get_neg(v):
        return -v

    @numba.njit
    def get_pos(v):
        return +v

    out = get_neg(vector.obj(x=3, y=4))
    assert out.x == -3
    assert out.y == -4

    out = get_pos(vector.obj(x=3, y=4))
    assert out.x == 3
    assert out.y == 4


def test_operator_bool():
    @numba.njit
    def get_true(v):
        return bool(v)

    assert not get_true(vector.obj(x=0, y=0))
    assert get_true(vector.obj(x=0, y=0.1))

    assert not get_true(vector.obj(x=0, y=0, z=0))
    assert get_true(vector.obj(x=0, y=0, z=0.1))

    assert not get_true(vector.obj(x=0, y=0, z=0, t=0))
    assert get_true(vector.obj(x=0, y=0, z=0.1, t=0))
    assert get_true(vector.obj(x=0, y=0, z=0, t=0.1))
    assert get_true(vector.obj(x=0, y=0, z=10, t=10))


def test_operator_truth():
    @numba.njit
    def get_true(v):
        return True if v else False

    @numba.njit
    def get_false(v):
        return False if v else True

    assert not get_true(vector.obj(x=0, y=0))
    assert get_true(vector.obj(x=0, y=0.1))

    assert not get_true(vector.obj(x=0, y=0, z=0))
    assert get_true(vector.obj(x=0, y=0, z=0.1))

    assert not get_true(vector.obj(x=0, y=0, z=0, t=0))
    assert get_true(vector.obj(x=0, y=0, z=0.1, t=0))
    assert get_true(vector.obj(x=0, y=0, z=0, t=0.1))
    assert get_true(vector.obj(x=0, y=0, z=10, t=10))

    assert get_false(vector.obj(x=0, y=0))
    assert not get_false(vector.obj(x=0, y=0.1))

    assert get_false(vector.obj(x=0, y=0, z=0))
    assert not get_false(vector.obj(x=0, y=0, z=0.1))

    assert get_false(vector.obj(x=0, y=0, z=0, t=0))
    assert not get_false(vector.obj(x=0, y=0, z=0.1, t=0))
    assert not get_false(vector.obj(x=0, y=0, z=0, t=0.1))
    assert not get_false(vector.obj(x=0, y=0, z=10, t=10))


def test_operator_eq():
    @numba.njit
    def get_eq(v1, v2):
        return v1 == v2

    @numba.njit
    def get_ne(v1, v2):
        return v1 != v2

    assert get_eq(vector.obj(x=3, y=4), vector.obj(px=3, py=4))
    assert not get_eq(vector.obj(x=3, y=4), vector.obj(x=4, y=3))
    with pytest.raises(numba.TypingError):
        get_eq(vector.obj(x=3, y=4), vector.obj(x=3, y=4, z=99))

    assert not get_ne(vector.obj(x=3, y=4), vector.obj(px=3, py=4))
    assert get_ne(vector.obj(x=3, y=4), vector.obj(x=4, y=3))
    with pytest.raises(numba.TypingError):
        get_ne(vector.obj(x=3, y=4), vector.obj(x=3, y=4, z=99))


def test_operator_add():
    @numba.njit
    def get_add(v1, v2):
        return v1 + v2

    @numba.njit
    def get_subtract(v1, v2):
        return v1 - v2

    out = get_add(vector.obj(x=1, y=2), vector.obj(px=3, py=4))
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(4)
    assert out.y == pytest.approx(6)

    out = get_add(vector.obj(x=1, y=2), vector.obj(x=3, y=4, z=5))
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(4)
    assert out.y == pytest.approx(6)

    out = get_add(vector.obj(x=1, y=2, z=0), vector.obj(x=3, y=4, z=5))
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert out.x == pytest.approx(4)
    assert out.y == pytest.approx(6)
    assert out.z == pytest.approx(5)

    out = get_subtract(vector.obj(x=1, y=2), vector.obj(px=3, py=4))
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(-2)
    assert out.y == pytest.approx(-2)


def test_operator_mul():
    @numba.njit
    def get_mul(a, b):
        return a * b

    @numba.njit
    def get_div(a, b):
        return a / b

    out = get_mul(vector.obj(x=1, y=2), 2)
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(4)

    out = get_mul(2, vector.obj(x=1, y=2))
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(2)
    assert out.y == pytest.approx(4)

    out = get_div(vector.obj(x=1, y=2), 2)
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(0.5)
    assert out.y == pytest.approx(1)

    with pytest.raises(numba.TypingError):
        get_div(2, vector.obj(x=1, y=2))


def test_operator_pow():
    @numba.njit
    def get_pow(a, b):
        return a ** b

    @numba.njit
    def get_square(a):
        return a ** 2

    assert get_pow(vector.obj(x=1, y=2), 4) == pytest.approx(25)

    assert get_square(vector.obj(x=1, y=2)) == pytest.approx(5)


def test_operator_matmul():
    @numba.njit
    def get_matmul(v1, v2):
        return v1 @ v2

    assert get_matmul(vector.obj(x=1, y=2), vector.obj(x=3, y=4)) == pytest.approx(11)


def test_numpy_functions():
    @numba.njit
    def get_absolute(v):
        return numpy.absolute(v)

    assert get_absolute(vector.obj(x=3, y=4)) == pytest.approx(5)

    @numba.njit
    def get_add(v1, v2):
        return numpy.add(v1, v2)

    out = get_add(vector.obj(x=3, y=4), vector.obj(x=10, y=20))
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(13)
    assert out.y == pytest.approx(24)
