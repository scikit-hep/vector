# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy
import pytest

import vector._backends.numpy_
import vector._backends.object_


def test_spatial_object():
    v1 = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2),
        vector._backends.object_.LongitudinalObjectZ(0.3),
    )
    v2 = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(0.4, 0.5),
        vector._backends.object_.LongitudinalObjectZ(0.6),
    )
    out = v1.cross(v2)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
    assert isinstance(out.longitudinal, vector._backends.object_.LongitudinalObjectZ)
    assert (out.x, out.y, out.z) == pytest.approx((-0.03, 0.06, -0.03))

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.cross(transformed2)
            assert isinstance(out, vector._backends.object_.VectorObject3D)
            assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
            assert isinstance(
                out.longitudinal, vector._backends.object_.LongitudinalObjectZ
            )
            assert (out.x, out.y, out.z) == pytest.approx((-0.03, 0.06, -0.03))


def test_spatial_numpy():
    v1 = vector._backends.numpy_.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    v2 = vector._backends.numpy_.VectorNumpy3D(
        [(0.4, 0.5, 0.6)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    out = v1.cross(v2)
    assert isinstance(out, vector._backends.numpy_.VectorNumpy3D)
    assert out.dtype.names == ("x", "y", "z")
    assert (out[0].x, out[0].y, out[0].z) == pytest.approx((-0.03, 0.06, -0.03))

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.cross(transformed2)
            assert isinstance(out, vector._backends.numpy_.VectorNumpy3D)
            assert out.dtype.names == ("x", "y", "z")
            assert (out[0].x, out[0].y, out[0].z) == pytest.approx((-0.03, 0.06, -0.03))


def test_lorentz_object():
    v1 = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2),
        vector._backends.object_.LongitudinalObjectZ(0.3),
        vector._backends.object_.TemporalObjectT(99),
    )
    v2 = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(0.4, 0.5),
        vector._backends.object_.LongitudinalObjectZ(0.6),
        vector._backends.object_.TemporalObjectT(99),
    )
    out = v1.cross(v2)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
    assert isinstance(out.longitudinal, vector._backends.object_.LongitudinalObjectZ)
    assert (out.x, out.y, out.z) == pytest.approx((-0.03, 0.06, -0.03))

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
            out = transformed1.cross(transformed2)
            assert isinstance(out, vector._backends.object_.VectorObject3D)
            assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
            assert isinstance(
                out.longitudinal, vector._backends.object_.LongitudinalObjectZ
            )
            assert (out.x, out.y, out.z) == pytest.approx((-0.03, 0.06, -0.03))


def test_lorentz_numpy():
    v1 = vector._backends.numpy_.VectorNumpy4D(
        [(0.1, 0.2, 0.3, 99)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    v2 = vector._backends.numpy_.VectorNumpy4D(
        [(0.4, 0.5, 0.6, 99)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    out = v1.cross(v2)
    assert isinstance(out, vector._backends.numpy_.VectorNumpy3D)
    assert out.dtype.names == ("x", "y", "z")
    assert out.tolist() == pytest.approx([(-0.03, 0.06, -0.030000000000000013)])

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
            out = transformed1.cross(transformed2)
            assert isinstance(out, vector._backends.numpy_.VectorNumpy3D)
            assert out.dtype.names == ("x", "y", "z")
            assert (out[0].x, out[0].y, out[0].z) == pytest.approx((-0.03, 0.06, -0.03))
