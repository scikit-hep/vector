# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

import vector._backends.numpy_
import vector._backends.object_


def test_VectorObject2D():
    v = vector.obj(x=1, y=2)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.object_.VectorObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.object_.VectorObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(0)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.object_.VectorObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(0)
    assert tv.t == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.object_.VectorObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.object_.VectorObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert getattr(tv, longitudinal) == pytest.approx(0)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.object_.VectorObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert getattr(tv, longitudinal) == pytest.approx(0)
                assert getattr(tv, temporal) == pytest.approx(0)


def test_MomentumObject2D():
    v = vector.obj(px=1, py=2)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.object_.MomentumObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.object_.MomentumObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(0)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.object_.MomentumObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(0)
    assert tv.t == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.object_.MomentumObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.object_.MomentumObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert getattr(tv, longitudinal) == pytest.approx(0)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.object_.MomentumObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert getattr(tv, longitudinal) == pytest.approx(0)
                assert getattr(tv, temporal) == pytest.approx(0)


def test_VectorNumpy2D():
    v = vector.array({"x": [1, 1, 1], "y": [2, 2, 2]})
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.numpy_.VectorNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.numpy_.VectorNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(0)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.numpy_.VectorNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(0)
    assert tv.t[0] == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.numpy_.VectorNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.numpy_.VectorNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert getattr(tv, longitudinal)[0] == pytest.approx(0)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.numpy_.VectorNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert getattr(tv, longitudinal)[0] == pytest.approx(0)
                assert getattr(tv, temporal)[0] == pytest.approx(0)


def test_MomentumNumpy2D():
    v = vector.array({"px": [1, 1, 1], "py": [2, 2, 2]})
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.numpy_.MomentumNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.numpy_.MomentumNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(0)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.numpy_.MomentumNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(0)
    assert tv.t[0] == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.numpy_.MomentumNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.numpy_.MomentumNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert getattr(tv, longitudinal)[0] == pytest.approx(0)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.numpy_.MomentumNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert getattr(tv, longitudinal)[0] == pytest.approx(0)
                assert getattr(tv, temporal)[0] == pytest.approx(0)


def test_VectorObject3D():
    v = vector.obj(x=1, y=2, z=3)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.object_.VectorObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.object_.VectorObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.object_.VectorObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    assert tv.t == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.object_.VectorObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.object_.VectorObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert tv.z == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.object_.VectorObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert tv.z == pytest.approx(3)
                assert getattr(tv, temporal) == pytest.approx(0)


def test_MomentumObject3D():
    v = vector.obj(px=1, py=2, pz=3)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.object_.MomentumObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.object_.MomentumObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.object_.MomentumObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    assert tv.t == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.object_.MomentumObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.object_.MomentumObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert tv.z == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.object_.MomentumObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert tv.z == pytest.approx(3)
                assert getattr(tv, temporal) == pytest.approx(0)


def test_VectorNumpy3D():
    v = vector.array({"x": [1, 1, 1], "y": [2, 2, 2], "z": [3, 3, 3]})
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.numpy_.VectorNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.numpy_.VectorNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.numpy_.VectorNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    assert tv.t[0] == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.numpy_.VectorNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.numpy_.VectorNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert tv.z[0] == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.numpy_.VectorNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert tv.z[0] == pytest.approx(3)
                assert getattr(tv, temporal)[0] == pytest.approx(0)


def test_MomentumNumpy3D():
    v = vector.array({"px": [1, 1, 1], "py": [2, 2, 2], "pz": [3, 3, 3]})
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.numpy_.MomentumNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.numpy_.MomentumNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.numpy_.MomentumNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    assert tv.t[0] == pytest.approx(0)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.numpy_.MomentumNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.numpy_.MomentumNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert tv.z[0] == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.numpy_.MomentumNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert tv.z[0] == pytest.approx(3)
                assert getattr(tv, temporal)[0] == pytest.approx(0)


def test_VectorObject4D():
    v = vector.obj(x=1, y=2, z=3, t=4)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.object_.VectorObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.object_.VectorObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.object_.VectorObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    assert tv.t == pytest.approx(4)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.object_.VectorObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.object_.VectorObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert tv.z == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.object_.VectorObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert tv.z == pytest.approx(3)
                assert tv.t == pytest.approx(4)


def test_MomentumObject4D():
    v = vector.obj(px=1, py=2, pz=3, E=4)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.object_.MomentumObject2D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.object_.MomentumObject3D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.object_.MomentumObject4D)
    assert tv.x == pytest.approx(1)
    assert tv.y == pytest.approx(2)
    assert tv.z == pytest.approx(3)
    assert tv.t == pytest.approx(4)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.object_.MomentumObject2D)
        assert tv.x == pytest.approx(1)
        assert tv.y == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.object_.MomentumObject3D)
            assert tv.x == pytest.approx(1)
            assert tv.y == pytest.approx(2)
            assert tv.z == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.object_.MomentumObject4D)
                assert tv.x == pytest.approx(1)
                assert tv.y == pytest.approx(2)
                assert tv.z == pytest.approx(3)
                assert tv.t == pytest.approx(4)


def test_VectorNumpy4D():
    v = vector.array({"x": [1, 1, 1], "y": [2, 2, 2], "z": [3, 3, 3], "t": [4, 4, 4]})
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.numpy_.VectorNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.numpy_.VectorNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.numpy_.VectorNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    assert tv.t[0] == pytest.approx(4)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.numpy_.VectorNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.numpy_.VectorNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert tv.z[0] == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.numpy_.VectorNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert tv.z[0] == pytest.approx(3)
                assert tv.t[0] == pytest.approx(4)


def test_MomentumNumpy4D():
    v = vector.array(
        {"px": [1, 1, 1], "py": [2, 2, 2], "pz": [3, 3, 3], "E": [4, 4, 4]}
    )
    tv = v.to_Vector2D()
    assert isinstance(tv, vector._backends.numpy_.MomentumNumpy2D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    tv = v.to_Vector3D()
    assert isinstance(tv, vector._backends.numpy_.MomentumNumpy3D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    tv = v.to_Vector4D()
    assert isinstance(tv, vector._backends.numpy_.MomentumNumpy4D)
    assert tv.x[0] == pytest.approx(1)
    assert tv.y[0] == pytest.approx(2)
    assert tv.z[0] == pytest.approx(3)
    assert tv.t[0] == pytest.approx(4)

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector._backends.numpy_.MomentumNumpy2D)
        assert tv.x[0] == pytest.approx(1)
        assert tv.y[0] == pytest.approx(2)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector._backends.numpy_.MomentumNumpy3D)
            assert tv.x[0] == pytest.approx(1)
            assert tv.y[0] == pytest.approx(2)
            assert tv.z[0] == pytest.approx(3)

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector._backends.numpy_.MomentumNumpy4D)
                assert tv.x[0] == pytest.approx(1)
                assert tv.y[0] == pytest.approx(2)
                assert tv.z[0] == pytest.approx(3)
                assert tv.t[0] == pytest.approx(4)
