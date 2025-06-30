# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy
import pytest

import vector


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

    for coord in (
        "xy",
        "rhophi",
    ):
        with pytest.raises(TypeError):
            getattr(vector.VectorObject2D, "from_" + coord)(complex(1, 2), 2)

        with pytest.raises(TypeError):
            getattr(vector.MomentumObject2D, "from_" + coord)(complex(1, 2), 2)


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

    for coord in (
        "xyz",
        "xytheta",
        "xyeta",
        "rhophiz",
        "rhophitheta",
        "rhophieta",
    ):
        with pytest.raises(TypeError):
            getattr(vector.VectorObject3D, "from_" + coord)(complex(1, 2), 2, 3)

        with pytest.raises(TypeError):
            getattr(vector.MomentumObject3D, "from_" + coord)(complex(1, 2), 2, 3)


def test_constructors_4D():
    vec = vector.VectorObject4D(x=1, y=2, z=3, t=4)
    assert vec.x == 1
    assert vec.y == 2
    assert vec.z == 3
    assert vec.t == 4

    vec = vector.VectorObject4D(x=1, y=2, eta=3, t=4)
    assert vec.x == 1
    assert vec.y == 2
    assert vec.eta == 3
    assert vec.t == 4

    vec = vector.VectorObject4D(x=1, y=2, theta=3, t=4)
    assert vec.x == 1
    assert vec.y == 2
    assert vec.theta == 3
    assert vec.t == 4

    vec = vector.VectorObject4D(rho=1, phi=2, z=3, t=4)
    assert vec.rho == 1
    assert vec.phi == 2
    assert vec.z == 3
    assert vec.t == 4

    vec = vector.VectorObject4D(rho=1, phi=2, eta=3, t=4)
    assert vec.rho == 1
    assert vec.phi == 2
    assert vec.eta == 3
    assert vec.t == 4

    vec = vector.VectorObject4D(rho=1, phi=2, theta=3, t=4)
    assert vec.rho == 1
    assert vec.phi == 2
    assert vec.theta == 3
    assert vec.t == 4

    vec = vector.VectorObject4D(x=1, y=2, z=3, tau=4)
    assert vec.x == 1
    assert vec.y == 2
    assert vec.z == 3
    assert vec.tau == 4

    vec = vector.VectorObject4D(x=1, y=2, eta=3, tau=4)
    assert vec.x == 1
    assert vec.y == 2
    assert vec.eta == 3
    assert vec.tau == 4

    vec = vector.VectorObject4D(x=1, y=2, theta=3, tau=4)
    assert vec.x == 1
    assert vec.y == 2
    assert vec.theta == 3
    assert vec.tau == 4

    vec = vector.VectorObject4D(rho=1, phi=2, z=3, tau=4)
    assert vec.rho == 1
    assert vec.phi == 2
    assert vec.z == 3
    assert vec.tau == 4

    vec = vector.VectorObject4D(rho=1, phi=2, eta=3, tau=4)
    assert vec.rho == 1
    assert vec.phi == 2
    assert vec.eta == 3
    assert vec.tau == 4

    vec = vector.VectorObject4D(rho=1, phi=2, theta=3, tau=4)
    assert vec.rho == 1
    assert vec.phi == 2
    assert vec.theta == 3
    assert vec.tau == 4

    vec = vector.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(3),
        temporal=vector.backends.object.TemporalObjectT(4),
    )
    assert vec.x == 1
    assert vec.y == 2
    assert vec.z == 3
    assert vec.t == 4

    with pytest.raises(TypeError):
        vector.VectorObject4D(rho=1, wow=2, z=3, t=4)

    with pytest.raises(TypeError):
        vector.VectorObject4D()

    with pytest.raises(TypeError):
        vector.VectorObject4D(x=complex(1, 2), y=2, z=3, t=4)

    vec = vector.MomentumObject4D(px=1, py=2, pz=3, E=4)
    assert vec.px == 1
    assert vec.py == 2
    assert vec.pz == 3
    assert vec.E == 4

    vec = vector.MomentumObject4D(px=1, py=2, eta=3, E=4)
    assert vec.px == 1
    assert vec.py == 2
    assert vec.eta == 3
    assert vec.E == 4

    vec = vector.MomentumObject4D(px=1, py=2, theta=3, E=4)
    assert vec.px == 1
    assert vec.py == 2
    assert vec.theta == 3
    assert vec.E == 4

    vec = vector.MomentumObject4D(pt=1, phi=2, pz=3, E=4)
    assert vec.pt == 1
    assert vec.phi == 2
    assert vec.pz == 3
    assert vec.E == 4

    vec = vector.MomentumObject4D(pt=1, phi=2, eta=3, E=4)
    assert vec.pt == 1
    assert vec.phi == 2
    assert vec.eta == 3
    assert vec.E == 4

    vec = vector.MomentumObject4D(pt=1, phi=2, theta=3, E=4)
    assert vec.pt == 1
    assert vec.phi == 2
    assert vec.theta == 3
    assert vec.E == 4

    vec = vector.MomentumObject4D(px=1, py=2, pz=3, M=4)
    assert vec.px == 1
    assert vec.py == 2
    assert vec.pz == 3
    assert vec.M == 4

    vec = vector.MomentumObject4D(px=1, py=2, eta=3, tau=4)
    assert vec.px == 1
    assert vec.py == 2
    assert vec.eta == 3
    assert vec.M == 4

    vec = vector.MomentumObject4D(px=1, py=2, theta=3, tau=4)
    assert vec.px == 1
    assert vec.py == 2
    assert vec.theta == 3
    assert vec.M == 4

    vec = vector.MomentumObject4D(pt=1, phi=2, pz=3, tau=4)
    assert vec.pt == 1
    assert vec.phi == 2
    assert vec.pz == 3
    assert vec.M == 4

    vec = vector.MomentumObject4D(pt=1, phi=2, eta=3, tau=4)
    assert vec.pt == 1
    assert vec.phi == 2
    assert vec.eta == 3
    assert vec.M == 4

    vec = vector.MomentumObject4D(pt=1, phi=2, theta=3, tau=4)
    assert vec.pt == 1
    assert vec.phi == 2
    assert vec.theta == 3
    assert vec.M == 4

    vec = vector.MomentumObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(3),
        temporal=vector.backends.object.TemporalObjectT(4),
    )
    assert vec.px == 1
    assert vec.py == 2
    assert vec.pz == 3
    assert vec.x == 1
    assert vec.y == 2
    assert vec.z == 3
    assert vec.t == 4

    with pytest.raises(TypeError):
        vector.MomentumObject4D(rho=1, wow=2, pz=3, t=4)

    with pytest.raises(TypeError):
        vector.MomentumObject4D()

    with pytest.raises(TypeError):
        vector.MomentumObject4D(x=complex(1, 2), y=2, z=3, t=4)

    for coord in (
        "xyzt",
        "xythetat",
        "xyetat",
        "rhophizt",
        "rhophithetat",
        "rhophietat",
        "xyztau",
        "xythetatau",
        "xyetatau",
        "rhophiztau",
        "rhophithetatau",
        "rhophietatau",
    ):
        with pytest.raises(TypeError):
            getattr(vector.VectorObject4D, "from_" + coord)(complex(1, 2), 2, 3, 4)

        with pytest.raises(TypeError):
            getattr(vector.MomentumObject4D, "from_" + coord)(complex(1, 2), 2, 3, 4)


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
