# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector.backends.numpy
import vector.backends.object


def test():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(3),
        temporal=vector.backends.object.TemporalObjectT(4),
    )
    p4 = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(5, 6),
        longitudinal=vector.backends.object.LongitudinalObjectZ(7),
        temporal=vector.backends.object.TemporalObjectT(15),
    )
    out = vec.boost_p4(p4)
    assert out.x == pytest.approx(3.5537720741941676)
    assert out.y == pytest.approx(5.0645264890330015)
    assert out.z == pytest.approx(6.575280903871835)
    assert out.t == pytest.approx(9.138547120755076)

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
            tvec, tp4 = getattr(vec, "to_" + t1)(), getattr(p4, "to_" + t2)()
            out = tvec.boost_p4(tp4)
            assert out.x == pytest.approx(3.5537720741941676)
            assert out.y == pytest.approx(5.0645264890330015)
            assert out.z == pytest.approx(6.575280903871835)
            assert out.t == pytest.approx(9.138547120755076)
