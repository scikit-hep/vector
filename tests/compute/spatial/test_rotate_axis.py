# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy
import pytest

import vector._backends.numpy_
import vector._backends.object_


def test_spatial_object():
    axis = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2),
        vector._backends.object_.LongitudinalObjectZ(0.3),
    )
    vec = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(0.4, 0.5),
        vector._backends.object_.LongitudinalObjectZ(0.6),
    )
    out = vec.rotate_axis(axis, 0.25)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
    assert isinstance(out.longitudinal, vector._backends.object_.LongitudinalObjectZ)
    assert out.x == pytest.approx(0.37483425404335763)
    assert out.y == pytest.approx(0.5383405688588193)
    assert out.z == pytest.approx(0.5828282027463345)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            taxis, tvec = (
                getattr(axis, "to_" + t1)(),
                getattr(vec, "to_" + t2)(),
            )
            out = tvec.rotate_axis(taxis, 0.25)
            assert isinstance(out, vector._backends.object_.VectorObject3D)
            assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
            assert isinstance(
                out.longitudinal, vector._backends.object_.LongitudinalObjectZ
            )
            assert out.x == pytest.approx(0.37483425404335763)
            assert out.y == pytest.approx(0.5383405688588193)
            assert out.z == pytest.approx(0.5828282027463345)


def test_spatial_numpy():
    axis = vector._backends.numpy_.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    vec = vector._backends.numpy_.VectorNumpy3D(
        [(0.4, 0.5, 0.6)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    out = vec.rotate_axis(axis, 0.25)
    assert isinstance(out, vector._backends.numpy_.VectorNumpy3D)
    assert out.dtype.names == ("x", "y", "z")
    assert out[0].x == pytest.approx(0.37483425404335763)
    assert out[0].y == pytest.approx(0.5383405688588193)
    assert out[0].z == pytest.approx(0.5828282027463345)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            taxis, tvec = (
                getattr(axis, "to_" + t1)(),
                getattr(vec, "to_" + t2)(),
            )
            out = tvec.rotate_axis(taxis, 0.25)
            assert isinstance(out, vector._backends.numpy_.VectorNumpy3D)
            assert out.dtype.names == ("x", "y", "z")
            assert out[0].x == pytest.approx(0.37483425404335763)
            assert out[0].y == pytest.approx(0.5383405688588193)
            assert out[0].z == pytest.approx(0.5828282027463345)


def test_lorentz_object():
    axis = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2),
        vector._backends.object_.LongitudinalObjectZ(0.3),
        vector._backends.object_.TemporalObjectT(99),
    )
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(0.4, 0.5),
        vector._backends.object_.LongitudinalObjectZ(0.6),
        vector._backends.object_.TemporalObjectT(99),
    )
    out = vec.rotate_axis(axis, 0.25)
    assert isinstance(out, vector._backends.object_.VectorObject4D)
    assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
    assert isinstance(out.longitudinal, vector._backends.object_.LongitudinalObjectZ)
    assert hasattr(out, "temporal")
    assert out.x == pytest.approx(0.37483425404335763)
    assert out.y == pytest.approx(0.5383405688588193)
    assert out.z == pytest.approx(0.5828282027463345)

    for t1 in (
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
        for t2 in (
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
            taxis, tvec = (
                getattr(axis, "to_" + t1)(),
                getattr(vec, "to_" + t2)(),
            )
            out = tvec.rotate_axis(taxis, 0.25)
            assert isinstance(out, vector._backends.object_.VectorObject4D)
            assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
            assert isinstance(
                out.longitudinal, vector._backends.object_.LongitudinalObjectZ
            )
            assert hasattr(out, "temporal")
            assert out.x == pytest.approx(0.37483425404335763)
            assert out.y == pytest.approx(0.5383405688588193)
            assert out.z == pytest.approx(0.5828282027463345)


def test_lorentz_numpy():
    axis = vector._backends.numpy_.VectorNumpy4D(
        [(0.1, 0.2, 0.3, 99)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    vec = vector._backends.numpy_.VectorNumpy4D(
        [(0.4, 0.5, 0.6, 99)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    out = vec.rotate_axis(axis, 0.25)
    assert isinstance(out, vector._backends.numpy_.VectorNumpy4D)
    assert out.dtype.names == ("x", "y", "z", "t")
    assert out[0].x == pytest.approx(0.37483425404335763)
    assert out[0].y == pytest.approx(0.5383405688588193)
    assert out[0].z == pytest.approx(0.5828282027463345)

    for t1 in (
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
        for t2 in (
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
            taxis, tvec = (
                getattr(axis, "to_" + t1)(),
                getattr(vec, "to_" + t2)(),
            )
            out = tvec.rotate_axis(taxis, 0.25)
            assert isinstance(out, vector._backends.numpy_.VectorNumpy4D)
            assert out.dtype.names in {
                ("x", "y", "z", "t"),
                ("x", "y", "z", "tau"),
            }
            assert out[0].x == pytest.approx(0.37483425404335763)
            assert out[0].y == pytest.approx(0.5383405688588193)
            assert out[0].z == pytest.approx(0.5828282027463345)
