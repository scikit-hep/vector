# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z, t = sympy.symbols("x y z t", real=True)
values = {x: 3, y: 2, z: 1, t: 10}


def test():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    out = vec.boostX(gamma=3)
    assert out.x == 2 * sympy.sqrt(2) * t + 3 * x
    assert out.y == y
    assert out.z == z
    assert out.t == 3 * t + 2 * sympy.sqrt(2) * x

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
        out = tvec.boostX(gamma=3)
        assert out.x.subs(values).evalf() == pytest.approx(37.2842712474619)
        assert out.y.subs(values).evalf() == pytest.approx(2)
        assert out.z.subs(values).evalf() == pytest.approx(1)
        assert out.t.subs(values).evalf() == pytest.approx(38.4852813742386)
