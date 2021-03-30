# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

import vector._backends.numpy_
import vector._backends.object_


def test_planar_object():
    v1 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2)
    )
    v2 = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2)
    )
    assert not v1.not_equal(v2)

    for t1 in "xy", "rhophi":
        for t2 in "xy", "rhophi":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            # can't test for exact equality due to round-off: see test_isclose
            transformed1.not_equal(transformed2)


def test_planar_numpy():
    v1 = vector._backends.numpy_.VectorNumpy2D(
        [(0.1, 0.2)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    v2 = vector._backends.numpy_.VectorNumpy2D(
        [(0.1, 0.2)],
        dtype=[("x", numpy.float64), ("y", numpy.float64)],
    )
    assert not v1.not_equal(v2).all()

    for t1 in "xy", "rhophi":
        for t2 in "xy", "rhophi":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            # can't test for exact equality due to round-off: see test_isclose
            tr1.not_equal(tr2)


def test_spatial_object():
    v1 = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2),
        vector._backends.object_.LongitudinalObjectZ(0.3),
    )
    v2 = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2),
        vector._backends.object_.LongitudinalObjectZ(0.3),
    )
    assert not v1.not_equal(v2)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            # can't test for exact equality due to round-off: see test_isclose
            transformed1.not_equal(transformed2)


def test_spatial_numpy():
    v1 = vector._backends.numpy_.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    v2 = vector._backends.numpy_.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    assert not v1.not_equal(v2).all()

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            # can't test for exact equality due to round-off: see test_isclose
            tr1.not_equal(tr2)


def test_lorentz_object():
    v1 = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2),
        vector._backends.object_.LongitudinalObjectZ(0.3),
        vector._backends.object_.TemporalObjectT(0.4),
    )
    v2 = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(0.1, 0.2),
        vector._backends.object_.LongitudinalObjectZ(0.3),
        vector._backends.object_.TemporalObjectT(0.4),
    )
    assert not v1.not_equal(v2)

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
            # can't test for exact equality due to round-off: see test_isclose
            tr1.not_equal(tr2)


def test_lorentz_numpy():
    v1 = vector._backends.numpy_.VectorNumpy4D(
        [(0.1, 0.2, 0.3, 0.4)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    v2 = vector._backends.numpy_.VectorNumpy4D(
        [(0.1, 0.2, 0.3, 0.4)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("t", numpy.float64),
        ],
    )
    assert not v1.not_equal(v2).all()

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
            # can't test for exact equality due to round-off: see test_isclose
            tr1.not_equal(tr2)
