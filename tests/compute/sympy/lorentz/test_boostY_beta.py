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
values = {x: 2, y: 3, z: 1, t: 4}


def test():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    out = vec.boostY(beta=-0.9428090415820634)
    assert out.x == x
    # cannot equate the sympy expressions because of floating point errors
    assert out.y.subs(values).evalf() == pytest.approx(-2.313708498984761)
    assert out.z == z
    # cannot equate the sympy expressions because of floating point errors
    assert out.t.subs(values).evalf() == pytest.approx(3.5147186257614287)

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
        out = tvec.boostY(beta=-0.9428090415820634)
        assert (
            out.x.subs(values).evalf() == pytest.approx(2)
            if not isinstance(out.x, (float, int))
            else out.x == pytest.approx(2)
        )
        assert (
            out.y.subs(values).evalf() == pytest.approx(-2.313708498984761)
            if not isinstance(out.y, (float, int))
            else out.y == pytest.approx(-2.313708498984761)
        )
        assert (
            out.z.subs(values).evalf() == pytest.approx(1)
            if not isinstance(out.z, (float, int))
            else out.z == pytest.approx(1)
        )
        assert (
            out.t.subs(values).evalf() == pytest.approx(3.5147186257614287)
            if not isinstance(out.t, (float, int))
            else out.t == pytest.approx(3.5147186257614287)
        )
