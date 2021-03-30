# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

import vector._backends.numpy_
import vector._backends.object_


def test():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(3, 2),
        vector._backends.object_.LongitudinalObjectZ(1),
        vector._backends.object_.TemporalObjectT(4),
    )
    out = vec.boostX(gamma=-3)
    assert out.x == pytest.approx(-2.313708498984761)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(1)
    assert out.t == pytest.approx(3.5147186257614287)

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
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.boostX(gamma=-3)
        assert out.x == pytest.approx(-2.313708498984761)
        assert out.y == pytest.approx(2)
        assert out.z == pytest.approx(1)
        assert out.t == pytest.approx(3.5147186257614287)
