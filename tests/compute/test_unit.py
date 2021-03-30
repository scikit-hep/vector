# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy
import pytest

import vector._backends.numpy_
import vector._backends.object_


def test_planar_object():
    v = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2)
    )
    u = v.unit()
    assert type(u) is type(v)  # noqa: E721
    assert type(u.azimuthal) is type(v.azimuthal)  # noqa: E721
    assert u.rho == pytest.approx(1)

    for t1 in "xy", "rhophi":
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)  # noqa: E721
        assert type(u.azimuthal) is type(t.azimuthal)  # noqa: E721
        assert u.rho == pytest.approx(1)


def test_planar_numpy():
    v = vector._backends.numpy_.VectorNumpy2D(
        [(0.1, 0.2)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    u = v.unit()
    assert type(u) is type(v)  # noqa: E721
    assert type(u.azimuthal) is type(v.azimuthal)  # noqa: E721
    assert u.rho[0] == pytest.approx(1)

    for t1 in "xy", "rhophi":
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)  # noqa: E721
        assert type(u.azimuthal) is type(t.azimuthal)  # noqa: E721
        assert u.rho[0] == pytest.approx(1)


def test_spatial_object():
    v = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2),
        vector._backends.object_.LongitudinalObjectZ(0.3),
    )
    u = v.unit()
    assert type(u) is type(v)  # noqa: E721
    assert type(u.azimuthal) is type(v.azimuthal)  # noqa: E721
    assert type(u.longitudinal) is type(v.longitudinal)  # noqa: E721
    assert u.mag == pytest.approx(1)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)  # noqa: E721
        assert type(u.azimuthal) is type(t.azimuthal)  # noqa: E721
        assert type(u.longitudinal) is type(t.longitudinal)  # noqa: E721
        assert u.mag == pytest.approx(1)


def test_spatial_numpy():
    v = vector._backends.numpy_.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    u = v.unit()
    assert type(u) is type(v)  # noqa: E721
    assert type(u.azimuthal) is type(v.azimuthal)  # noqa: E721
    assert type(u.longitudinal) is type(v.longitudinal)  # noqa: E721
    assert u.mag[0] == pytest.approx(1)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)  # noqa: E721
        assert type(u.azimuthal) is type(t.azimuthal)  # noqa: E721
        assert type(u.longitudinal) is type(t.longitudinal)  # noqa: E721
        assert u.mag[0] == pytest.approx(1)


def test_lorentz_object():
    v = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2),
        vector._backends.object_.LongitudinalObjectZ(0.3),
        vector._backends.object_.TemporalObjectT(0.4),
    )
    u = v.unit()
    assert type(u) is type(v)  # noqa: E721
    assert type(u.azimuthal) is type(v.azimuthal)  # noqa: E721
    assert type(u.longitudinal) is type(v.longitudinal)  # noqa: E721
    assert type(u.temporal) is type(v.temporal)  # noqa: E721
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
        assert type(u) is type(t)  # noqa: E721
        assert type(u.azimuthal) is type(t.azimuthal)  # noqa: E721
        assert type(u.longitudinal) is type(t.longitudinal)  # noqa: E721
        assert type(u.temporal) is type(t.temporal)  # noqa: E721
        assert u.tau == pytest.approx(1)


def test_lorentz_numpy():
    v = vector._backends.numpy_.VectorNumpy4D(
        [(0.1, 0.2, 0.3, 0.4)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    u = v.unit()
    assert type(u) is type(v)  # noqa: E721
    assert type(u.azimuthal) is type(v.azimuthal)  # noqa: E721
    assert type(u.longitudinal) is type(v.longitudinal)  # noqa: E721
    assert type(u.temporal) is type(v.temporal)  # noqa: E721
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
        assert type(u) is type(t)  # noqa: E721
        assert type(u.azimuthal) is type(t.azimuthal)  # noqa: E721
        assert type(u.longitudinal) is type(t.longitudinal)  # noqa: E721
        assert type(u.temporal) is type(t.temporal)  # noqa: E721
        assert u.tau[0] == pytest.approx(1)
