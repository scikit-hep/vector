# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy
import pytest

import vector._backends.numpy_
import vector._backends.object_


def test_planar_object():
    v1 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectXY(3, 4)
    )
    v2 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectXY(5, 12)
    )
    out = v1.subtract(v2)
    assert out.x == pytest.approx(-2)
    assert out.y == pytest.approx(-8)

    for t1 in "xy", "rhophi":
        for t2 in "xy", "rhophi":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.subtract(transformed2)
            assert out.x == pytest.approx(-2)
            assert out.y == pytest.approx(-8)


def test_planar_numpy():
    v1 = vector._backends.numpy_.VectorNumpy2D(
        [(3, 4)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    v2 = vector._backends.numpy_.VectorNumpy2D(
        [(5, 12)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    out = v1.subtract(v2)
    assert out.x[0] == pytest.approx(-2)
    assert out.y[0] == pytest.approx(-8)

    for t1 in "xy", "rhophi":
        for t2 in "xy", "rhophi":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.subtract(transformed2)
            assert out.x[0] == pytest.approx(-2)
            assert out.y[0] == pytest.approx(-8)


def test_spatial_object():
    v1 = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(3, 4),
        vector._backends.object_.LongitudinalObjectZ(2),
    )
    v2 = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(5, 12),
        vector._backends.object_.LongitudinalObjectZ(4),
    )
    out = v1.subtract(v2)
    assert out.x == pytest.approx(-2)
    assert out.y == pytest.approx(-8)
    assert out.z == pytest.approx(-2)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.subtract(transformed2)
            assert out.x == pytest.approx(-2)
            assert out.y == pytest.approx(-8)
            assert out.z == pytest.approx(-2)


def test_spatial_numpy():
    v1 = vector._backends.numpy_.VectorNumpy3D(
        [(3, 4, 2)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    v2 = vector._backends.numpy_.VectorNumpy3D(
        [(5, 12, 4)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    out = v1.subtract(v2)
    assert out.x[0] == pytest.approx(-2)
    assert out.y[0] == pytest.approx(-8)
    assert out.z[0] == pytest.approx(-2)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.subtract(transformed2)
            assert out.x[0] == pytest.approx(-2)
            assert out.y[0] == pytest.approx(-8)
            assert out.z[0] == pytest.approx(-2)


def test_lorentz_object():
    v1 = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(3, 4),
        vector._backends.object_.LongitudinalObjectZ(2),
        vector._backends.object_.TemporalObjectT(20),
    )
    v2 = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(5, 12),
        vector._backends.object_.LongitudinalObjectZ(4),
        vector._backends.object_.TemporalObjectT(15),
    )
    out = v1.subtract(v2)
    assert out.x == pytest.approx(-2)
    assert out.y == pytest.approx(-8)
    assert out.z == pytest.approx(-2)
    assert out.t == pytest.approx(5)

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
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.subtract(transformed2)
            assert out.x == pytest.approx(-2)
            assert out.y == pytest.approx(-8)
            assert out.z == pytest.approx(-2)
            assert out.t == pytest.approx(5)


def test_lorentz_numpy():
    v1 = vector._backends.numpy_.VectorNumpy4D(
        [(3, 4, 2, 20)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    v2 = vector._backends.numpy_.VectorNumpy4D(
        [(5, 12, 4, 15)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    out = v1.subtract(v2)
    assert out.x[0] == pytest.approx(-2)
    assert out.y[0] == pytest.approx(-8)
    assert out.z[0] == pytest.approx(-2)
    assert out.t[0] == pytest.approx(5)

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
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.subtract(transformed2)
            assert out.x[0] == pytest.approx(-2)
            assert out.y[0] == pytest.approx(-8)
            assert out.z[0] == pytest.approx(-2)
            assert out.t[0] == pytest.approx(5)
