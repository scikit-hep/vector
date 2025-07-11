# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy
import pytest

import vector.backends.numpy
import vector.backends.object


def test_planar_object():
    v = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.2)
    )
    u = v.unit()
    assert type(u) is type(v)
    assert type(u.azimuthal) is type(v.azimuthal)
    assert u.rho == pytest.approx(1)

    for t1 in "xy", "rhophi":
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)
        assert type(u.azimuthal) is type(t.azimuthal)
        assert u.rho == pytest.approx(1)


def test_planar_numpy():
    v = vector.backends.numpy.VectorNumpy2D(
        [(0.1, 0.2)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    u = v.unit()
    assert type(u) is type(v)
    assert type(u.azimuthal) is type(v.azimuthal)
    assert u.rho[0] == pytest.approx(1)

    for t1 in "xy", "rhophi":
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)
        assert type(u.azimuthal) is type(t.azimuthal)
        assert u.rho[0] == pytest.approx(1)


def test_spatial_object():
    v = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.3),
    )
    u = v.unit()
    assert type(u) is type(v)
    assert type(u.azimuthal) is type(v.azimuthal)
    assert type(u.longitudinal) is type(v.longitudinal)
    assert u.mag == pytest.approx(1)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)
        assert type(u.azimuthal) is type(t.azimuthal)
        assert type(u.longitudinal) is type(t.longitudinal)
        assert u.mag == pytest.approx(1)


def test_spatial_numpy():
    v = vector.backends.numpy.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    u = v.unit()
    assert type(u) is type(v)
    assert type(u.azimuthal) is type(v.azimuthal)
    assert type(u.longitudinal) is type(v.longitudinal)
    assert u.mag[0] == pytest.approx(1)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)
        assert type(u.azimuthal) is type(t.azimuthal)
        assert type(u.longitudinal) is type(t.longitudinal)
        assert u.mag[0] == pytest.approx(1)


def test_lorentz_object():
    v = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.3),
        temporal=vector.backends.object.TemporalObjectT(0.4),
    )
    u = v.unit()
    assert type(u) is type(v)
    assert type(u.azimuthal) is type(v.azimuthal)
    assert type(u.longitudinal) is type(v.longitudinal)
    assert type(u.temporal) is type(v.temporal)
    assert u.tau == pytest.approx(1)

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
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)
        assert type(u.azimuthal) is type(t.azimuthal)
        assert type(u.longitudinal) is type(t.longitudinal)
        assert type(u.temporal) is type(t.temporal)
        assert u.tau == pytest.approx(1)


def test_lorentz_numpy():
    v = vector.backends.numpy.VectorNumpy4D(
        [(0.1, 0.2, 0.3, 0.4)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    u = v.unit()
    assert type(u) is type(v)
    assert type(u.azimuthal) is type(v.azimuthal)
    assert type(u.longitudinal) is type(v.longitudinal)
    assert type(u.temporal) is type(v.temporal)
    assert u.tau[0] == pytest.approx(1)

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
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)
        assert type(u.azimuthal) is type(t.azimuthal)
        assert type(u.longitudinal) is type(t.longitudinal)
        assert type(u.temporal) is type(t.temporal)
        assert u.tau[0] == pytest.approx(1)
