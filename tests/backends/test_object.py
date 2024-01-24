# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy
import pytest

import vector


def test_dimension_conversion():
    # 2D -> 3D
    vec = vector.VectorObject2D(x=1, y=2)
    assert vec.to_Vector3D(z=1).z == 1
    assert vec.to_Vector3D(eta=1).eta == 1
    assert vec.to_Vector3D(theta=1).theta == 1

    assert vec.to_Vector3D(z=1).x == vec.x
    assert vec.to_Vector3D(z=1).y == vec.y

    # 2D -> 4D
    assert vec.to_Vector4D(z=1, t=1).z == 1
    assert vec.to_Vector4D(z=1, t=1).t == 1
    assert vec.to_Vector4D(eta=1, t=1).eta == 1
    assert vec.to_Vector4D(eta=1, t=1).t == 1
    assert vec.to_Vector4D(theta=1, t=1).theta == 1
    assert vec.to_Vector4D(theta=1, t=1).t == 1
    assert vec.to_Vector4D(z=1, tau=1).z == 1
    assert vec.to_Vector4D(z=1, tau=1).tau == 1
    assert vec.to_Vector4D(eta=1, tau=1).eta == 1
    assert vec.to_Vector4D(eta=1, tau=1).tau == 1
    assert vec.to_Vector4D(theta=1, tau=1).theta == 1
    assert vec.to_Vector4D(theta=1, tau=1).tau == 1

    assert vec.to_Vector4D(z=1, t=1).x == vec.x
    assert vec.to_Vector4D(z=1, t=1).y == vec.y

    # 3D -> 4D
    vec = vector.VectorObject3D(x=1, y=2, z=3)
    assert vec.to_Vector4D(t=1).t == 1
    assert vec.to_Vector4D(tau=1).tau == 1

    assert vec.to_Vector4D(t=1).x == vec.x
    assert vec.to_Vector4D(t=1).y == vec.y
    assert vec.to_Vector4D(t=1).z == vec.z


def test_constructors_2D():
    vec = vector.VectorObject2D(x=1, y=2)
    assert vec.x == 1
    assert vec.y == 2

    vec = vector.VectorObject2D(rho=1, phi=2)
    assert vec.rho == 1
    assert vec.phi == 2

    vec = vector.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2)
    )
    assert vec.x == 1
    assert vec.y == 2

    with pytest.raises(TypeError):
        vector.VectorObject2D(rho=1, wow=2)

    with pytest.raises(TypeError):
        vector.VectorObject2D(rho=complex(1, 2), wow=2)

    with pytest.raises(TypeError):
        vector.VectorObject2D()

    vec = vector.MomentumObject2D(px=1, py=2)
    assert vec.px == 1
    assert vec.py == 2
    assert vec.x == 1
    assert vec.y == 2

    vec = vector.MomentumObject2D(x=1, py=2)
    assert vec.px == 1
    assert vec.py == 2
    assert vec.x == 1
    assert vec.y == 2

    vec = vector.MomentumObject2D(pt=1, phi=2)
    assert vec.pt == 1
    assert vec.phi == 2
    assert vec.rho == 1
    assert vec.phi == 2

    vec = vector.MomentumObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2)
    )
    assert vec.px == 1
    assert vec.py == 2
    assert vec.x == 1
    assert vec.y == 2

    with pytest.raises(TypeError):
        vector.MomentumObject2D(rho=1, wow=2)

    with pytest.raises(TypeError):
        vector.MomentumObject2D(rho=False, wow=2)

    with pytest.raises(TypeError):
        vector.MomentumObject2D()


def test_constructors_3D():
    vec = vector.VectorObject3D(x=1, y=2, z=3)
    assert vec.x == 1
    assert vec.y == 2
    assert vec.z == 3

    vec = vector.VectorObject3D(x=1, y=2, eta=3)
    assert vec.x == 1
    assert vec.y == 2
    assert vec.eta == 3

    vec = vector.VectorObject3D(x=1, y=2, theta=3)
    assert vec.x == 1
    assert vec.y == 2
    assert vec.theta == 3

    vec = vector.VectorObject3D(rho=1, phi=2, z=3)
    assert vec.rho == 1
    assert vec.phi == 2
    assert vec.z == 3

    vec = vector.VectorObject3D(rho=1, phi=2, eta=3)
    assert vec.rho == 1
    assert vec.phi == 2
    assert vec.eta == 3

    vec = vector.VectorObject3D(rho=1, phi=2, theta=3)
    assert vec.rho == 1
    assert vec.phi == 2
    assert vec.theta == 3

    vec = vector.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(3),
    )
    assert vec.x == 1
    assert vec.y == 2
    assert vec.z == 3

    with pytest.raises(TypeError):
        vector.VectorObject3D(rho=1, wow=2, z=3)

    with pytest.raises(TypeError):
        vector.VectorObject3D()

    with pytest.raises(TypeError):
        vector.VectorObject3D(x=complex(1, 2), y=2, z=3)

    vec = vector.MomentumObject3D(px=1, py=2, pz=3)
    assert vec.px == 1
    assert vec.py == 2
    assert vec.pz == 3
    assert vec.x == 1
    assert vec.y == 2
    assert vec.z == 3

    vec = vector.MomentumObject3D(px=1, py=2, eta=3)
    assert vec.px == 1
    assert vec.py == 2
    assert vec.eta == 3
    assert vec.x == 1
    assert vec.y == 2

    vec = vector.MomentumObject3D(px=1, py=2, theta=3)
    assert vec.px == 1
    assert vec.py == 2
    assert vec.theta == 3
    assert vec.x == 1
    assert vec.y == 2

    vec = vector.MomentumObject3D(pt=1, phi=2, pz=3)
    assert vec.pt == 1
    assert vec.phi == 2
    assert vec.pz == 3
    assert vec.rho == 1
    assert vec.z == 3

    vec = vector.MomentumObject3D(pt=1, phi=2, eta=3)
    assert vec.pt == 1
    assert vec.phi == 2
    assert vec.eta == 3
    assert vec.rho == 1

    vec = vector.MomentumObject3D(pt=1, phi=2, theta=3)
    assert vec.pt == 1
    assert vec.phi == 2
    assert vec.theta == 3
    assert vec.rho == 1

    vec = vector.MomentumObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(3),
    )
    assert vec.px == 1
    assert vec.py == 2
    assert vec.pz == 3
    assert vec.x == 1
    assert vec.y == 2
    assert vec.z == 3

    with pytest.raises(TypeError):
        vector.MomentumObject3D(rho=1, wow=2, pz=3)

    with pytest.raises(TypeError):
        vector.MomentumObject3D()

    with pytest.raises(TypeError):
        vector.MomentumObject3D(x=complex(1, 2), y=2, z=3)


def test_array_casting():
    obj = vector.obj(x=1, y=1)
    assert isinstance(obj, vector.VectorObject2D)
    assert isinstance(numpy.asanyarray(obj), vector.VectorNumpy2D)
    assert numpy.asanyarray(obj).shape == ()

    obj = vector.obj(px=1, py=1)
    assert isinstance(obj, vector.MomentumObject2D)
    assert isinstance(numpy.asanyarray(obj), vector.MomentumNumpy2D)
    assert numpy.asanyarray(obj).shape == ()

    obj = vector.obj(x=1, y=1, z=1)
    assert isinstance(obj, vector.VectorObject3D)
    assert isinstance(numpy.asanyarray(obj), vector.VectorNumpy3D)
    assert numpy.asanyarray(obj).shape == ()

    obj = vector.obj(px=1, py=1, pz=1)
    assert isinstance(obj, vector.MomentumObject3D)
    assert isinstance(numpy.asanyarray(obj), vector.MomentumNumpy3D)
    assert numpy.asanyarray(obj).shape == ()

    obj = vector.obj(x=1, y=1, z=1, t=1)
    assert isinstance(obj, vector.VectorObject4D)
    assert isinstance(numpy.asanyarray(obj), vector.VectorNumpy4D)
    assert numpy.asanyarray(obj).shape == ()

    obj = vector.obj(px=1, py=1, pz=1, E=1)
    assert isinstance(obj, vector.MomentumObject4D)
    assert isinstance(numpy.asanyarray(obj), vector.MomentumNumpy4D)
    assert numpy.asanyarray(obj).shape == ()

    with pytest.raises(TypeError):
        vector.obj(x=1, y=[1, 2])

    with pytest.raises(TypeError):
        vector.obj(x=1, y=complex(1, 2))

    with pytest.raises(TypeError):
        vector.obj(x=1, y=False)


def test_demotion():
    v1 = vector.obj(x=0.1, y=0.2)
    v2 = vector.obj(x=1, y=2, z=3)
    v3 = vector.obj(x=10, y=20, z=30, t=40)

    # order should not matter
    assert v1 + v2 == vector.obj(x=1.1, y=2.2)
    assert v2 + v1 == vector.obj(x=1.1, y=2.2)
    assert v1 + v3 == vector.obj(x=10.1, y=20.2)
    assert v3 + v1 == vector.obj(x=10.1, y=20.2)
    assert v2 + v3 == vector.obj(x=11, y=22, z=33)
    assert v3 + v2 == vector.obj(x=11, y=22, z=33)

    v1 = vector.obj(px=0.1, py=0.2)
    v2 = vector.obj(px=1, py=2, pz=3)
    v3 = vector.obj(px=10, py=20, pz=30, t=40)

    # order should not matter
    assert v1 + v2 == vector.obj(px=1.1, py=2.2)
    assert v2 + v1 == vector.obj(px=1.1, py=2.2)
    assert v1 + v3 == vector.obj(px=10.1, py=20.2)
    assert v3 + v1 == vector.obj(px=10.1, py=20.2)
    assert v2 + v3 == vector.obj(px=11, py=22, pz=33)
    assert v3 + v2 == vector.obj(px=11, py=22, pz=33)

    v1 = vector.obj(px=0.1, py=0.2)
    v2 = vector.obj(x=1, y=2, z=3)
    v3 = vector.obj(px=10, py=20, pz=30, t=40)

    # momentum + generic = generic
    assert v1 + v2 == vector.obj(x=1.1, y=2.2)
    assert v2 + v1 == vector.obj(x=1.1, y=2.2)
    assert v1 + v3 == vector.obj(px=10.1, py=20.2)
    assert v3 + v1 == vector.obj(px=10.1, py=20.2)
    assert v2 + v3 == vector.obj(x=11, y=22, z=33)
    assert v3 + v2 == vector.obj(x=11, y=22, z=33)
