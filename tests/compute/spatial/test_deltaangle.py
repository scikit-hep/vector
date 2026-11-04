# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import math

import numpy
import pytest

import vector
import vector.backends.numpy
import vector.backends.object


def test_spatial_object():
    v1 = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.3),
    )
    v2 = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.4, 0.5),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.6),
    )
    assert v1.deltaangle(v2) == pytest.approx(0.225726)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert transformed1.deltaangle(transformed2) == pytest.approx(0.225726)


def test_spatial_numpy():
    v1 = vector.backends.numpy.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    v2 = vector.backends.numpy.VectorNumpy3D(
        [(0.4, 0.5, 0.6)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    assert v1.deltaangle(v2)[0] == pytest.approx(0.225726)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.deltaangle(tr2)[0] == pytest.approx(0.225726)


def test_lorentz_object():
    v1 = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.3),
        temporal=vector.backends.object.TemporalObjectT(99),
    )
    v2 = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.4, 0.5),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.6),
        temporal=vector.backends.object.TemporalObjectT(99),
    )
    assert v1.deltaangle(v2) == pytest.approx(0.225726)

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
            tr1, tr2 = getattr(v1, "to_" + t1)(), getattr(v2, "to_" + t2)()
            assert tr1.deltaangle(tr2) == pytest.approx(0.225726)


def test_lorentz_numpy():
    v1 = vector.backends.numpy.VectorNumpy4D(
        [(0.1, 0.2, 0.3, 99)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    v2 = vector.backends.numpy.VectorNumpy4D(
        [(0.4, 0.5, 0.6, 99)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    assert v1.deltaangle(v2) == pytest.approx(0.225726)

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
            tr1, tr2 = getattr(v1, "to_" + t1)(), getattr(v2, "to_" + t2)()
            assert tr1.deltaangle(tr2) == pytest.approx(0.225726)


@pytest.mark.parametrize("delta", [1e-3, 1e-6, 1e-8, 1e-10, 1e-12])
def test_small_angle_precision(delta):
    v1 = vector.obj(x=1.0, y=0.0, z=0.0)
    v2 = vector.obj(x=1.0, y=delta, z=0.0)
    assert v1.deltaangle(v2) == pytest.approx(math.atan(delta), rel=1e-14, abs=0)

    a1 = vector.array({"x": [1.0], "y": [0.0], "z": [0.0]})
    a2 = vector.array({"x": [1.0], "y": [delta], "z": [0.0]})
    assert a1.deltaangle(a2) == pytest.approx([math.atan(delta)], rel=1e-14, abs=0)


def test_antiparallel_and_perpendicular():
    v = vector.obj(x=1.0, y=2.0, z=3.0)
    assert v.deltaangle(-v) == pytest.approx(math.pi)
    assert vector.obj(x=1.0, y=0.0, z=0.0).deltaangle(
        vector.obj(x=0.0, y=5.0, z=0.0)
    ) == pytest.approx(math.pi / 2)


@pytest.mark.parametrize(
    "coordinates", ["xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta"]
)
@pytest.mark.parametrize(
    "other", ["xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta"]
)
def test_same_vector_in_different_coordinates(coordinates, other):
    v = vector.obj(x=1.0, y=1.0, z=1.0)
    first = getattr(v, "to_" + coordinates)()
    second = getattr(v, "to_" + other)()
    assert first.deltaangle(second) == pytest.approx(0.0, abs=1e-14)
