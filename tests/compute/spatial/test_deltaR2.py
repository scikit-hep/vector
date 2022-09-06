# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy
import pytest

import vector.backends.numpy
import vector.backends.object


def test_spatial_object():
    v1 = vector.backends.object.VectorObject3D(
        vector.backends.object.AzimuthalObjectXY(0.1, 0.2),
        vector.backends.object.LongitudinalObjectZ(0.3),
    )
    v2 = vector.backends.object.VectorObject3D(
        vector.backends.object.AzimuthalObjectXY(0.4, 0.5),
        vector.backends.object.LongitudinalObjectZ(0.6),
    )
    assert v1.deltaR2(v2) == pytest.approx(0.116083865330319)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert transformed1.deltaR2(transformed2) == pytest.approx(
                0.116083865330319
            )


def test_spatial_numpy():
    v1 = vector.backends.numpy.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    v2 = vector.backends.numpy.VectorNumpy3D(
        [(0.4, 0.5, 0.6)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    assert v1.deltaR2(v2)[0] == pytest.approx(0.116083865330319)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.deltaR2(tr2)[0] == pytest.approx(0.116083865330319)


def test_lorentz_object():
    v1 = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(0.1, 0.2),
        vector.backends.object.LongitudinalObjectZ(0.3),
        vector.backends.object.TemporalObjectT(99),
    )
    v2 = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(0.4, 0.5),
        vector.backends.object.LongitudinalObjectZ(0.6),
        vector.backends.object.TemporalObjectT(99),
    )
    assert v1.deltaR2(v2) == pytest.approx(0.116083865330319)

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
            assert tr1.deltaR2(tr2) == pytest.approx(0.116083865330319)


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
    assert v1.deltaR2(v2) == pytest.approx(0.116083865330319)

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
            assert tr1.deltaR2(tr2) == pytest.approx(0.116083865330319)
