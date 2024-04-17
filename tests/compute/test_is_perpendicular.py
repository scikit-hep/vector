# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy

import vector.backends.numpy
import vector.backends.object


def test_planar_object():
    v1 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0, 1)
    )
    v2 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 0)
    )
    v3 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.5)
    )
    assert not v1.is_perpendicular(v1)
    assert not v2.is_perpendicular(v2)
    assert not v3.is_perpendicular(v3)
    assert v1.is_perpendicular(v2)
    assert not v1.is_perpendicular(v3)
    assert not v2.is_perpendicular(v3)

    for t1 in "xy", "rhophi":
        for t2 in "xy", "rhophi":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_perpendicular(tr2)


def test_planar_numpy():
    v1 = vector.backends.numpy.VectorNumpy2D(
        [(0, 1)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    v2 = vector.backends.numpy.VectorNumpy2D(
        [(1, 0)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    v3 = vector.backends.numpy.VectorNumpy2D(
        [(0.1, 0.5)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    assert not v1.is_perpendicular(v1)
    assert not v2.is_perpendicular(v2)
    assert not v3.is_perpendicular(v3)
    assert v1.is_perpendicular(v2)
    assert not v1.is_perpendicular(v3)
    assert not v2.is_perpendicular(v3)

    for t1 in "xy", "rhophi":
        for t2 in "xy", "rhophi":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_perpendicular(tr2)


def test_spatial_object():
    v1 = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0, 1),
        longitudinal=vector.backends.object.LongitudinalObjectZ(1),
    )
    v2 = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 0),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0),
    )
    v3 = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.5),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.9),
    )
    assert not v1.is_perpendicular(v1)
    assert not v2.is_perpendicular(v2)
    assert not v3.is_perpendicular(v3)
    assert v1.is_perpendicular(v2)
    assert not v1.is_perpendicular(v3)
    assert not v2.is_perpendicular(v3)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_perpendicular(tr2)


def test_spatial_numpy():
    v1 = vector.backends.numpy.VectorNumpy3D(
        [(0, 1, 1)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    v2 = vector.backends.numpy.VectorNumpy3D(
        [(1, 0, 0)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    v3 = vector.backends.numpy.VectorNumpy3D(
        [(0.1, 0.5, 0.9)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    assert not v1.is_perpendicular(v1)
    assert not v2.is_perpendicular(v2)
    assert not v3.is_perpendicular(v3)
    assert v1.is_perpendicular(v2)
    assert not v1.is_perpendicular(v3)
    assert not v2.is_perpendicular(v3)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_perpendicular(tr2)
