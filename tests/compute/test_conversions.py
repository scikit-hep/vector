# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector.backends.numpy
import vector.backends.object


def test_VectorObject2D():
    v = vector.obj(x=1, y=2)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.object.VectorObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.object.VectorObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(0)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.object.VectorObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(0)
    assert tv.t == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.object.VectorObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.object.VectorObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert getattr(tv, longitudinal) == pytest.approx(0)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.object.VectorObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert getattr(tv, longitudinal) == pytest.approx(0)
                assert getattr(tv, temporal) == pytest.approx(0)


def test_MomentumObject2D():
    v = vector.obj(px=1, py=2)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.object.MomentumObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.object.MomentumObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(0)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.object.MomentumObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(0)
    assert tv.t == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.object.MomentumObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.object.MomentumObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert getattr(tv, longitudinal) == pytest.approx(0)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.object.MomentumObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert getattr(tv, longitudinal) == pytest.approx(0)
                assert getattr(tv, temporal) == pytest.approx(0)


def test_VectorNumpy2D():
    v = vector.array({"x": [1, 1, 1], "y": [2, 2, 2]})
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.numpy.VectorNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.numpy.VectorNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(0)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.numpy.VectorNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(0)
    assert tv.t[0] == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.numpy.VectorNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.numpy.VectorNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert getattr(tv, longitudinal)[0] == pytest.approx(0)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.numpy.VectorNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert getattr(tv, longitudinal)[0] == pytest.approx(0)
                assert getattr(tv, temporal)[0] == pytest.approx(0)


def test_MomentumNumpy2D():
    v = vector.array({"px": [1, 1, 1], "py": [2, 2, 2]})
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.numpy.MomentumNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.numpy.MomentumNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(0)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.numpy.MomentumNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(0)
    assert tv.t[0] == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.numpy.MomentumNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.numpy.MomentumNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert getattr(tv, longitudinal)[0] == pytest.approx(0)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.numpy.MomentumNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert getattr(tv, longitudinal)[0] == pytest.approx(0)
                assert getattr(tv, temporal)[0] == pytest.approx(0)


def test_VectorObject3D():
    v = vector.obj(x=1, y=2, z=3)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.object.VectorObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.object.VectorObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.object.VectorObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    assert tv.t == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.object.VectorObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.object.VectorObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert tv.z == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.object.VectorObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert tv.z == pytest.approx(3)
                assert getattr(tv, temporal) == pytest.approx(0)


def test_MomentumObject3D():
    v = vector.obj(px=1, py=2, pz=3)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.object.MomentumObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.object.MomentumObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.object.MomentumObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    assert tv.t == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.object.MomentumObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.object.MomentumObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert tv.z == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.object.MomentumObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert tv.z == pytest.approx(3)
                assert getattr(tv, temporal) == pytest.approx(0)


def test_VectorNumpy3D():
    v = vector.array({"x": [1, 1, 1], "y": [2, 2, 2], "z": [3, 3, 3]})
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.numpy.VectorNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.numpy.VectorNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.numpy.VectorNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    assert tv.t[0] == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.numpy.VectorNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.numpy.VectorNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert tv.z[0] == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.numpy.VectorNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert tv.z[0] == pytest.approx(3)
                assert getattr(tv, temporal)[0] == pytest.approx(0)


def test_MomentumNumpy3D():
    v = vector.array({"px": [1, 1, 1], "py": [2, 2, 2], "pz": [3, 3, 3]})
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.numpy.MomentumNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.numpy.MomentumNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.numpy.MomentumNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    assert tv.t[0] == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.numpy.MomentumNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.numpy.MomentumNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert tv.z[0] == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.numpy.MomentumNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert tv.z[0] == pytest.approx(3)
                assert getattr(tv, temporal)[0] == pytest.approx(0)


def test_VectorObject4D():
    v = vector.obj(x=1, y=2, z=3, t=4)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.object.VectorObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.object.VectorObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.object.VectorObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    assert tv.t == pytest.approx(4)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.object.VectorObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.object.VectorObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert tv.z == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.object.VectorObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert tv.z == pytest.approx(3)
                assert tv.t == pytest.approx(4)


def test_MomentumObject4D():
    v = vector.obj(px=1, py=2, pz=3, E=4)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.object.MomentumObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.object.MomentumObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.object.MomentumObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    assert tv.t == pytest.approx(4)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.object.MomentumObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.object.MomentumObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert tv.z == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.object.MomentumObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert tv.z == pytest.approx(3)
                assert tv.t == pytest.approx(4)


def test_VectorNumpy4D():
    v = vector.array({"x": [1, 1, 1], "y": [2, 2, 2], "z": [3, 3, 3], "t": [4, 4, 4]})
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.numpy.VectorNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.numpy.VectorNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.numpy.VectorNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    assert tv.t[0] == pytest.approx(4)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.numpy.VectorNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.numpy.VectorNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert tv.z[0] == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.numpy.VectorNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert tv.z[0] == pytest.approx(3)
                assert tv.t[0] == pytest.approx(4)


def test_MomentumNumpy4D():
    v = vector.array(
        {"px": [1, 1, 1], "py": [2, 2, 2], "pz": [3, 3, 3], "E": [4, 4, 4]}
    )
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.backends.numpy.MomentumNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.backends.numpy.MomentumNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.backends.numpy.MomentumNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    assert tv.t[0] == pytest.approx(4)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.backends.numpy.MomentumNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.backends.numpy.MomentumNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert tv.z[0] == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.backends.numpy.MomentumNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert tv.z[0] == pytest.approx(3)
                assert tv.t[0] == pytest.approx(4)


def test_conversion_with_coords_object():
    # 2D -> 3D
    vec = vector.VectorObject2D(x=1, y=2)
    assert vec.to_Vector3D(z=1).z == 1
    assert vec.to_Vector3D(eta=1).eta == 1
    assert vec.to_Vector3D(theta=1).theta == 1

    # test alias
    assert vec.to_3D(z=1).x == vec.x
    assert vec.to_3D(z=1).y == vec.y

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

    # test alias
    assert vec.to_4D(z=1, t=1).x == vec.x
    assert vec.to_4D(z=1, t=1).y == vec.y

    # 3D -> 4D
    vec = vector.VectorObject3D(x=1, y=2, z=3)

    # test alias
    assert vec.to_4D(t=1).t == 1
    assert vec.to_4D(tau=1).tau == 1

    assert vec.to_Vector4D(t=1).x == vec.x
    assert vec.to_Vector4D(t=1).y == vec.y
    assert vec.to_Vector4D(t=1).z == vec.z

    # check if momentum coords work
    vec = vector.MomentumObject2D(px=1, py=2)
    assert vec.to_Vector3D(pz=1).pz == 1

    # test both alias and original methods
    assert vec.to_4D(pz=1, m=1).pz == 1
    assert vec.to_4D(pz=1, m=1).m == 1
    assert vec.to_4D(pz=1, mass=1).mass == 1
    assert vec.to_4D(pz=1, M=1).M == 1
    assert vec.to_Vector4D(pz=1, e=1).e == 1
    assert vec.to_Vector4D(pz=1, energy=1).energy == 1
    assert vec.to_Vector4D(pz=1, E=1).E == 1

    vec = vector.MomentumObject3D(px=1, py=2, pz=3)

    # test both alias and original methods
    assert vec.to_4D(m=1).m == 1
    assert vec.to_4D(mass=1).mass == 1
    assert vec.to_4D(M=1).M == 1
    assert vec.to_Vector4D(e=1).e == 1
    assert vec.to_Vector4D(energy=1).energy == 1
    assert vec.to_Vector4D(E=1).E == 1


def test_conversion_with_coords_numpy():
    # 2D -> 3D
    vec = vector.VectorNumpy2D(
        [(1.0, 1.0), (2.0, 2.0)],
        dtype=[("x", float), ("y", float)],
    )
    assert all(vec.to_Vector3D(z=1).z == 1)
    assert all(vec.to_Vector3D(eta=1).eta == 1)
    assert all(vec.to_Vector3D(theta=1).theta == 1)

    # test alias
    assert all(vec.to_3D(z=1).x == vec.x)
    assert all(vec.to_3D(z=1).y == vec.y)

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

    # test alias
    assert all(vec.to_4D(z=1, t=1).x == vec.x)
    assert all(vec.to_4D(z=1, t=1).y == vec.y)

    # 3D -> 4D
    vec = vector.VectorNumpy3D(
        [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        dtype=[("x", float), ("y", float), ("z", float)],
    )
    assert all(vec.to_Vector4D(t=1).t == 1)
    assert all(vec.to_Vector4D(tau=1).tau == 1)

    # test alias
    assert all(vec.to_4D(t=1).x == vec.x)
    assert all(vec.to_4D(t=1).y == vec.y)
    assert all(vec.to_4D(t=1).z == vec.z)

    # check if momentum coords work
    vec = vector.MomentumNumpy2D(
        [(1.0, 1.0), (2.0, 2.0)],
        dtype=[("px", float), ("py", float)],
    )
    assert all(vec.to_Vector3D(pz=1).pz == 1)

    # test both alias and original methods
    assert all(vec.to_4D(pz=1, m=1).pz == 1)
    assert all(vec.to_4D(pz=1, m=1).m == 1)
    assert all(vec.to_4D(pz=1, mass=1).mass == 1)
    assert all(vec.to_4D(pz=1, M=1).M == 1)
    assert all(vec.to_Vector4D(pz=1, e=1).e == 1)
    assert all(vec.to_Vector4D(pz=1, energy=1).energy == 1)
    assert all(vec.to_Vector4D(pz=1, E=1).E == 1)

    vec = vector.MomentumNumpy3D(
        [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        dtype=[("px", float), ("py", float), ("pz", float)],
    )

    # test both alias and original methods
    assert all(vec.to_4D(m=1).m == 1)
    assert all(vec.to_4D(mass=1).mass == 1)
    assert all(vec.to_4D(M=1).M == 1)
    assert all(vec.to_Vector4D(e=1).e == 1)
    assert all(vec.to_Vector4D(energy=1).energy == 1)
    assert all(vec.to_Vector4D(E=1).E == 1)


def test_like_object():
    v1 = vector.obj(x=0.1, y=0.2)
    v2 = vector.obj(x=1, y=2, z=3)
    v3 = vector.obj(x=10, y=20, z=30, t=40)

    # 2D + 3D.like(2D) = 2D
    assert v1 + v2.like(v1) == vector.obj(x=1.1, y=2.2)
    assert v2.like(v1) + v1 == vector.obj(x=1.1, y=2.2)
    # 2D + 4D.like(2D) = 2D
    assert v1 + v3.like(v1) == vector.obj(x=10.1, y=20.2)
    assert v3.like(v1) + v1 == vector.obj(x=10.1, y=20.2)
    # 3D + 2D.like(3D) = 3D
    assert v2 + v1.like(v2) == vector.obj(x=1.1, y=2.2, z=3)
    assert v1.like(v2) + v2 == vector.obj(x=1.1, y=2.2, z=3)
    # 3D + 4D.like(3D) = 3D
    assert v2 + v3.like(v2) == vector.obj(x=11, y=22, z=33)
    assert v3.like(v2) + v2 == vector.obj(x=11, y=22, z=33)
    # 4D + 2D.like(4D) = 4D
    assert v3 + v1.like(v3) == vector.obj(x=10.1, y=20.2, z=30.0, t=40.0)
    assert v1.like(v3) + v3 == vector.obj(x=10.1, y=20.2, z=30.0, t=40.0)
    # 4D + 3D.like(4D) = 4D
    assert v3 + v2.like(v3) == vector.obj(x=11, y=22, z=33, t=40)
    assert v2.like(v3) + v3 == vector.obj(x=11, y=22, z=33, t=40)

    v1 = vector.obj(px=0.1, py=0.2)
    v2 = vector.obj(px=1, py=2, pz=3)
    v3 = vector.obj(px=10, py=20, pz=30, t=40)

    # order should not matter
    # 2D + 3D.like(2D) = 2D
    assert v1 + v2.like(v1) == vector.obj(px=1.1, py=2.2)
    assert v2.like(v1) + v1 == vector.obj(px=1.1, py=2.2)
    # 2D + 4D.like(2D) = 2D
    assert v1 + v3.like(v1) == vector.obj(px=10.1, py=20.2)
    assert v3.like(v1) + v1 == vector.obj(px=10.1, py=20.2)
    # 3D + 2D.like(3D) = 3D
    assert v2 + v1.like(v2) == vector.obj(px=1.1, py=2.2, pz=3)
    assert v1.like(v2) + v2 == vector.obj(px=1.1, py=2.2, pz=3)
    # 3D + 4D.like(3D) = 3D
    assert v2 + v3.like(v2) == vector.obj(px=11, py=22, pz=33)
    assert v3.like(v2) + v2 == vector.obj(px=11, py=22, pz=33)
    # 4D + 2D.like(4D) = 4D
    assert v3 + v1.like(v3) == vector.obj(px=10.1, py=20.2, pz=30.0, E=40.0)
    assert v1.like(v3) + v3 == vector.obj(px=10.1, py=20.2, pz=30.0, E=40.0)
    # 4D + 3D.like(4D) = 4D
    assert v3 + v2.like(v3) == vector.obj(px=11, py=22, pz=33, E=40)
    assert v2.like(v3) + v3 == vector.obj(px=11, py=22, pz=33, E=40)

    v1 = vector.obj(px=0.1, py=0.2)
    v2 = vector.obj(x=1, y=2, z=3)
    v3 = vector.obj(px=10, py=20, pz=30, t=40)


def test_like_numpy():
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
    v2_v1 = vector.array(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
            "z": [5.0, 1.0, 1.0],
        },
    )
    v2_v3 = vector.array(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
            "z": [10.0, 2.0, 2.0],
        },
    )
    v3_v2 = vector.array(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
            "z": [10.0, 2.0, 2.0],
            "t": [16.0, 31.0, 46.0],
        },
    )
    v1_v3 = vector.array(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
            "z": [5.0, 1.0, 1.0],
            "t": [16.0, 31.0, 46.0],
        },
    )

    # 2D + 3D.like(2D) = 2D
    assert all(v1 + v2.like(v1) == v1_v2)
    assert all(v2.like(v1) + v1 == v1_v2)
    # 2D + 4D.like(2D) = 2D
    assert all(v1 + v3.like(v1) == v1_v2)
    assert all(v3.like(v1) + v1 == v1_v2)
    # 3D + 2D.like(3D) = 3D
    assert all(v2 + v1.like(v2) == v2_v1)
    assert all(v1.like(v2) + v2 == v2_v1)
    # 3D + 4D.like(3D) = 3D
    assert all(v2 + v3.like(v2) == v2_v3)
    assert all(v3.like(v2) + v2 == v2_v3)
    # 4D + 2D.like(4D) = 4D
    assert all(v3 + v1.like(v3) == v1_v3)
    assert all(v1.like(v3) + v3 == v1_v3)
    # 4D + 3D.like(4D) = 4D
    assert all(v3 + v2.like(v3) == v3_v2)
    assert all(v2.like(v3) + v3 == v3_v2)

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

    pv1_v2 = vector.array(
        {
            "px": [20.0, 40.0, 60.0],
            "py": [-20.0, 40.0, 60.0],
        },
    )
    pv2_v1 = vector.array(
        {
            "px": [20.0, 40.0, 60.0],
            "py": [-20.0, 40.0, 60.0],
            "pz": [5.0, 1.0, 1.0],
        },
    )
    pv2_v3 = vector.array(
        {
            "px": [20.0, 40.0, 60.0],
            "py": [-20.0, 40.0, 60.0],
            "pz": [10.0, 2.0, 2.0],
        },
    )
    pv3_v2 = vector.array(
        {
            "px": [20.0, 40.0, 60.0],
            "py": [-20.0, 40.0, 60.0],
            "pz": [10.0, 2.0, 2.0],
            "t": [16.0, 31.0, 46.0],
        },
    )
    pv1_v3 = vector.array(
        {
            "px": [20.0, 40.0, 60.0],
            "py": [-20.0, 40.0, 60.0],
            "pz": [5.0, 1.0, 1.0],
            "t": [16.0, 31.0, 46.0],
        },
    )

    # 2D + 3D.like(2D) = 2D
    assert all(v1 + v2.like(v1) == pv1_v2)
    assert all(v2.like(v1) + v1 == pv1_v2)
    # 2D + 4D.like(2D) = 2D
    assert all(v1 + v3.like(v1) == pv1_v2)
    assert all(v3.like(v1) + v1 == pv1_v2)
    # 3D + 2D.like(3D) = 3D
    assert all(v2 + v1.like(v2) == pv2_v1)
    assert all(v1.like(v2) + v2 == pv2_v1)
    # 3D + 4D.like(3D) = 3D
    assert all(v2 + v3.like(v2) == pv2_v3)
    assert all(v3.like(v2) + v2 == pv2_v3)
    # 4D + 2D.like(4D) = 4D
    assert all(v3 + v1.like(v3) == pv1_v3)
    assert all(v1.like(v3) + v3 == pv1_v3)
    # 4D + 3D.like(4D) = 4D
    assert all(v3 + v2.like(v3) == pv3_v2)
    assert all(v2.like(v3) + v3 == pv3_v2)


def test_momentum_preservation_object():
    v1 = vector.obj(px=0.1, py=0.2)
    v2 = vector.obj(x=1, y=2, z=3)
    v3 = vector.obj(px=10, py=20, pz=30, t=40)

    # momentum + generic = momentum
    # 2D + 3D.like(2D) = 2D
    assert isinstance(v1 + v2.like(v1), vector.MomentumObject2D)
    assert isinstance(v2.like(v1) + v1, vector.MomentumObject2D)
    # 2D + 4D.like(2D) = 2D
    assert isinstance(v1 + v3.like(v1), vector.MomentumObject2D)
    assert isinstance(v3.like(v1) + v1, vector.MomentumObject2D)
    # 3D + 2D.like(3D) = 3D
    assert isinstance(v2 + v1.like(v2), vector.MomentumObject3D)
    assert isinstance(v1.like(v2) + v2, vector.MomentumObject3D)
    # 3D + 4D.like(3D) = 3D
    assert isinstance(v2 + v3.like(v2), vector.MomentumObject3D)
    assert isinstance(v3.like(v2) + v2, vector.MomentumObject3D)
    # 4D + 2D.like(4D) = 4D
    assert isinstance(v3 + v1.like(v3), vector.MomentumObject4D)
    assert isinstance(v1.like(v3) + v3, vector.MomentumObject4D)
    # 4D + 3D.like(4D) = 4D
    assert isinstance(v3 + v2.like(v3), vector.MomentumObject4D)
    assert isinstance(v2.like(v3) + v3, vector.MomentumObject4D)


def test_momentum_preservation_numpy():
    v1 = vector.array(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
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
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
            "pz": [5.0, 1.0, 1.0],
            "t": [16.0, 31.0, 46.0],
        },
    )

    # momentum + generic = momentum
    # 2D + 3D.like(2D) = 2D
    assert isinstance(v1 + v2.like(v1), vector.MomentumNumpy2D)
    assert isinstance(v2.like(v1) + v1, vector.MomentumNumpy2D)
    # 2D + 4D.like(2D) = 2D
    assert isinstance(v1 + v3.like(v1), vector.MomentumNumpy2D)
    assert isinstance(v3.like(v1) + v1, vector.MomentumNumpy2D)
    # 3D + 2D.like(3D) = 3D
    assert isinstance(v2 + v1.like(v2), vector.MomentumNumpy3D)
    assert isinstance(v1.like(v2) + v2, vector.MomentumNumpy3D)
    # 3D + 4D.like(3D) = 3D
    assert isinstance(v2 + v3.like(v2), vector.MomentumNumpy3D)
    assert isinstance(v3.like(v2) + v2, vector.MomentumNumpy3D)
    # 4D + 2D.like(4D) = 4D
    assert isinstance(v3 + v1.like(v3), vector.MomentumNumpy4D)
    assert isinstance(v1.like(v3) + v3, vector.MomentumNumpy4D)
    # 4D + 3D.like(4D) = 4D
    assert isinstance(v3 + v2.like(v3), vector.MomentumNumpy4D)
    assert isinstance(v2.like(v3) + v3, vector.MomentumNumpy4D)
