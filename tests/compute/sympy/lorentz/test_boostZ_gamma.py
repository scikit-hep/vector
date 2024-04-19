# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z, t = sympy.symbols("x y z t", real=True)
values = {x: 1, y: 2, z: 3, t: 4}


def test():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    out = vec.boostZ(gamma=3)
    assert out.x == x
    assert out.y == y
    assert out.z.subs(values).evalf() == pytest.approx(20.3137084989848)
    assert out.t.subs(values).evalf() == pytest.approx(20.4852813742386)

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
        out = tvec.boostZ(gamma=3)
        assert out.x.subs(values).evalf() == pytest.approx(1)
        assert out.y.subs(values).evalf() == pytest.approx(2)
        assert out.z.subs(values).evalf() == pytest.approx(20.3137084989848)
        assert out.t.subs(values).evalf() == pytest.approx(20.4852813742386)
