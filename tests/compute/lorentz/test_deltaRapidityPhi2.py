# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy
import pytest

import vector.backends.numpy
import vector.backends.object


def test_lorentz_object():
    v1 = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(1.0, 1.0),
        vector.backends.object.LongitudinalObjectZ(1.0),
        vector.backends.object.TemporalObjectTau(1.0),
    )
    v2 = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(-1.0, -1.0),
        vector.backends.object.LongitudinalObjectZ(-1.0),
        vector.backends.object.TemporalObjectTau(1.0),
    )
    expected_result = (
        # phi
        numpy.pi**2
        # rapidity
        + ((0.5 * numpy.log(3 / 1) - 0.5 * numpy.log(1 / 3)) ** 2)
    )
    assert v1.deltaRapidityPhi2(v2) == pytest.approx(expected_result)

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
            assert tr1.deltaRapidityPhi2(tr2) == pytest.approx(expected_result)


def test_lorentz_numpy():
    v1 = vector.backends.numpy.VectorNumpy4D(
        [(1.0, 1.0, 1.0, 1.0)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("tau", numpy.float64),
        ],
    )
    v2 = vector.backends.numpy.VectorNumpy4D(
        [(-1.0, -1.0, -1.0, 1.0)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("tau", numpy.float64),
        ],
    )
    expected_result = (
        # phi
        numpy.pi**2
        # rapidity
        + ((0.5 * numpy.log(3 / 1) - 0.5 * numpy.log(1 / 3)) ** 2)
    )
    assert v1.deltaRapidityPhi2(v2) == pytest.approx(expected_result)

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
            assert numpy.allclose(tr1.deltaRapidityPhi2(tr2), expected_result)
