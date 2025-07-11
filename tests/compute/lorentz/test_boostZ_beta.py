# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
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
    out = vec.boostZ(beta=-0.9428090415820634)
    assert out.x == pytest.approx(1)
    assert out.y == pytest.approx(2)
    assert out.z == pytest.approx(-2.313708498984761)
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
        out = tvec.boostZ(beta=-0.9428090415820634)
        assert out.x == pytest.approx(1)
        assert out.y == pytest.approx(2)
        assert out.z == pytest.approx(-2.313708498984761)
        assert out.t == pytest.approx(3.5147186257614287)
