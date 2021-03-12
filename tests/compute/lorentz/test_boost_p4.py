# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

import vector.backends.numpy_
import vector.backends.object_
import vector.geometry


def test():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectXY(1, 2),
        vector.backends.object_.LongitudinalObjectZ(3),
        vector.backends.object_.TemporalObjectT(4),
    )
    p4 = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectXY(5, 6),
        vector.backends.object_.LongitudinalObjectZ(7),
        vector.backends.object_.TemporalObjectT(15),
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
            print(t2)

            tvec, tp4 = getattr(vec, "to_" + t1)(), getattr(p4, "to_" + t2)()
            out = tvec.boost_p4(tp4)
            assert out.x == pytest.approx(3.5537720741941676)
            assert out.y == pytest.approx(5.0645264890330015)
            assert out.z == pytest.approx(6.575280903871835)
            assert out.t == pytest.approx(9.138547120755076)