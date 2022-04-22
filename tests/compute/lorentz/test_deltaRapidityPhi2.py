# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy
import pytest

import vector._backends.numpy_
import vector._backends.object_


def test_lorentz_object():
    v1 = vector._backends.object_.MomentumObject4D(
        vector._backends.object_.AzimuthalObjectXY(1.0, 1.0),
        vector._backends.object_.LongitudinalObjectZ(1.0),
        vector._backends.object_.TemporalObjectTau(1.0),
    )
    v2 = vector._backends.object_.MomentumObject4D(
        vector._backends.object_.AzimuthalObjectXY(-1.0, -1.0),
        vector._backends.object_.LongitudinalObjectZ(-1.0),
        vector._backends.object_.TemporalObjectTau(1.0),
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
    v1 = vector._backends.numpy_.VectorNumpy4D(
        [(1.0, 1.0, 1.0, 1.0)],
        dtype=[
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("z", numpy.float64),
            ("tau", numpy.float64),
        ],
    )
    v2 = vector._backends.numpy_.VectorNumpy4D(
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
