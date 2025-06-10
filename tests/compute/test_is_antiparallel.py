# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy

import vector.backends.numpy
import vector.backends.object


def test_planar_object():
    v1 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.2)
    )
    v2 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(-0.3, -0.6)
    )
    v3 = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.3, 0.6)
    )
    assert v1.is_antiparallel(-v1)
    assert v2.is_antiparallel(-v2)
    assert v3.is_antiparallel(-v3)
    assert v1.is_antiparallel(v2)
    assert v2.is_antiparallel(v3)
    assert not v1.is_antiparallel(v3)

    for t1 in "xy", "rhophi":
        for t2 in "xy", "rhophi":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_antiparallel(tr2)


def test_planar_numpy():
    v1 = vector.backends.numpy.VectorNumpy2D(
        [(0.1, 0.2)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    v2 = vector.backends.numpy.VectorNumpy2D(
        [(-0.3, -0.6)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    v3 = vector.backends.numpy.VectorNumpy2D(
        [(0.3, 0.6)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    assert v1.is_antiparallel(-v1)
    assert v2.is_antiparallel(-v2)
    assert v3.is_antiparallel(-v3)
    assert v1.is_antiparallel(v2)
    assert v2.is_antiparallel(v3)
    assert not v1.is_antiparallel(v3)

    for t1 in "xy", "rhophi":
        for t2 in "xy", "rhophi":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_antiparallel(tr2)


def test_spatial_object():
    v1 = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.3),
    )
    v2 = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(-0.3, -0.6),
        longitudinal=vector.backends.object.LongitudinalObjectZ(-0.9),
    )
    v3 = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.3, 0.6),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.9),
    )
    assert v1.is_antiparallel(-v1)
    assert v2.is_antiparallel(-v2)
    assert v3.is_antiparallel(-v3)
    assert v1.is_antiparallel(v2)
    assert v2.is_antiparallel(v3)
    assert not v1.is_antiparallel(v3)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_antiparallel(tr2)


def test_spatial_numpy():
    v1 = vector.backends.numpy.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    v2 = vector.backends.numpy.VectorNumpy3D(
        [(-0.3, -0.6, -0.9)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    v3 = vector.backends.numpy.VectorNumpy3D(
        [(0.3, 0.6, 0.9)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    assert v1.is_antiparallel(-v1)
    assert v2.is_antiparallel(-v2)
    assert v3.is_antiparallel(-v3)
    assert v1.is_antiparallel(v2)
    assert v2.is_antiparallel(v3)
    assert not v1.is_antiparallel(v3)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_antiparallel(tr2)
