# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy
import pytest

import vector._methods
import vector.backends.numpy
import vector.backends.object


def test_spatial_object():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.3),
    )
    out = vec.rotateY(0.25)
    assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
    assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
    assert out.x == pytest.approx(0.17111242994742137)
    assert out.y == pytest.approx(0.2)
    assert out.z == pytest.approx(0.2659333305877411)

    for t in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        out = getattr(vec, "to_" + t)().rotateY(0.25)
        assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
        assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
        assert out.x == pytest.approx(0.17111242994742137)
        assert out.y == pytest.approx(0.2)
        assert out.z == pytest.approx(0.2659333305877411)


def test_spatial_numpy():
    vec = vector.backends.numpy.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    out = vec.rotateY(0.25)
    assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
    assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
    assert out[0].x == pytest.approx(0.17111242994742137)
    assert out[0].y == pytest.approx(0.2)
    assert out[0].z == pytest.approx(0.2659333305877411)

    for t in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        out = getattr(vec, "to_" + t)().rotateY(0.25)
        assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
        assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
        assert out[0].x == pytest.approx(0.17111242994742137)
        assert out[0].y == pytest.approx(0.2)
        assert out[0].z == pytest.approx(0.2659333305877411)


def test_lorentz_object():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.3),
        temporal=vector.backends.object.TemporalObjectT(99),
    )
    out = vec.rotateY(0.25)
    assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
    assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
    assert hasattr(out, "temporal")
    assert out.x == pytest.approx(0.17111242994742137)
    assert out.y == pytest.approx(0.2)
    assert out.z == pytest.approx(0.2659333305877411)

    for t in (
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
        out = getattr(vec, "to_" + t)().rotateY(0.25)
        assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
        assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
        assert hasattr(out, "temporal")
        assert out.x == pytest.approx(0.17111242994742137)
        assert out.y == pytest.approx(0.2)
        assert out.z == pytest.approx(0.2659333305877411)


def test_lorentz_numpy():
    vec = vector.backends.numpy.VectorNumpy4D(
        [(0.1, 0.2, 0.3, 99)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    out = vec.rotateY(0.25)
    assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
    assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
    assert out[0].x == pytest.approx(0.17111242994742137)
    assert out[0].y == pytest.approx(0.2)
    assert out[0].z == pytest.approx(0.2659333305877411)

    for t in (
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
        out = getattr(vec, "to_" + t)().rotateY(0.25)
        assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
        assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
        assert out[0].x == pytest.approx(0.17111242994742137)
        assert out[0].y == pytest.approx(0.2)
        assert out[0].z == pytest.approx(0.2659333305877411)
