# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z, t, px, py, pz, M = sympy.symbols("x y z t px py pz M", real=True)
values = {x: 1, y: 2, z: 3, t: 4, px: 5, py: 6, pz: 7, M: 15}


def test():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    p4 = vector.MomentumSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(px, py),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(pz),
        temporal=vector.backends.sympy.TemporalSympyT(M),
    )
    out = vec.boost_p4(p4)
    assert out.x.simplify() == (
        M**2 * x
        + M * px * t
        + M * x * sympy.sqrt(M**2 - px**2 - py**2 - pz**2)
        + px * py * y
        + px * pz * z
        + px * t * sympy.sqrt(M**2 - px**2 - py**2 - pz**2)
        - py**2 * x
        - pz**2 * x
    ) / (M**2 + M * sympy.sqrt(M**2 - px**2 - py**2 - pz**2) - px**2 - py**2 - pz**2)
    assert out.y.simplify() == (
        M**2 * y
        + M * py * t
        + M * y * sympy.sqrt(M**2 - px**2 - py**2 - pz**2)
        - px**2 * y
        + px * py * x
        + py * pz * z
        + py * t * sympy.sqrt(M**2 - px**2 - py**2 - pz**2)
        - pz**2 * y
    ) / (M**2 + M * sympy.sqrt(M**2 - px**2 - py**2 - pz**2) - px**2 - py**2 - pz**2)
    assert out.z.simplify() == (
        M**2 * z
        + M * pz * t
        + M * z * sympy.sqrt(M**2 - px**2 - py**2 - pz**2)
        - px**2 * z
        + px * pz * x
        - py**2 * z
        + py * pz * y
        + pz * t * sympy.sqrt(M**2 - px**2 - py**2 - pz**2)
    ) / (M**2 + M * sympy.sqrt(M**2 - px**2 - py**2 - pz**2) - px**2 - py**2 - pz**2)
    assert out.t.simplify() == (M * t + px * x + py * y + pz * z) / sympy.sqrt(
        M**2 - px**2 - py**2 - pz**2
    )

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
            assert out.x.subs(values).evalf() == pytest.approx(3.5537720741941676)
            assert out.y.subs(values).evalf() == pytest.approx(5.0645264890330015)
            assert out.z.subs(values).evalf() == pytest.approx(6.575280903871835)
            assert out.t.subs(values).evalf() == pytest.approx(9.138547120755076)
