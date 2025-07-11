# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
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
    out = vec.rotateX(0.25)
    assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
    assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
    assert out.x == pytest.approx(0.1)
    assert out.y == pytest.approx(0.1195612965657721)
    assert out.z == pytest.approx(0.340154518364098)

    for t in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        out = getattr(vec, "to_" + t)().rotateX(0.25)
        assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
        assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
        assert out.x == pytest.approx(0.1)
        assert out.y == pytest.approx(0.1195612965657721)
        assert out.z == pytest.approx(0.340154518364098)


def test_spatial_numpy():
    vec = vector.backends.numpy.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    out = vec.rotateX(0.25)
    assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
    assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
    assert out[0].x == pytest.approx(0.1)
    assert out[0].y == pytest.approx(0.1195612965657721)
    assert out[0].z == pytest.approx(0.340154518364098)

    for t in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        out = getattr(vec, "to_" + t)().rotateX(0.25)
        assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
        assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
        assert out[0].x == pytest.approx(0.1)
        assert out[0].y == pytest.approx(0.1195612965657721)
        assert out[0].z == pytest.approx(0.340154518364098)


def test_lorentz_object():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.3),
        temporal=vector.backends.object.TemporalObjectT(99),
    )
    out = vec.rotateX(0.25)
    assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
    assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
    assert hasattr(out, "temporal")
    assert out.x == pytest.approx(0.1)
    assert out.y == pytest.approx(0.1195612965657721)
    assert out.z == pytest.approx(0.340154518364098)

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
        out = getattr(vec, "to_" + t)().rotateX(0.25)
        assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
        assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
        assert hasattr(out, "temporal")
        assert out.x == pytest.approx(0.1)
        assert out.y == pytest.approx(0.1195612965657721)
        assert out.z == pytest.approx(0.340154518364098)


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
    out = vec.rotateX(0.25)
    assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
    assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
    assert out[0].x == pytest.approx(0.1)
    assert out[0].y == pytest.approx(0.1195612965657721)
    assert out[0].z == pytest.approx(0.340154518364098)

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
        out = getattr(vec, "to_" + t)().rotateX(0.25)
        assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
        assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
        assert out[0].x == pytest.approx(0.1)
        assert out[0].y == pytest.approx(0.1195612965657721)
        assert out[0].z == pytest.approx(0.340154518364098)
