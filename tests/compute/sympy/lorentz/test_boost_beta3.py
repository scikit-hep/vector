# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z, t, px, py, pz = sympy.symbols("x y z t px py pz", real=True)
values = {x: 1, y: 2, z: 3, t: 4, px: 5 / 15, py: 6 / 15, pz: 7 / 15}


def test():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    beta = vector.MomentumSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(px, py),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(pz),
    )
    out = vec.boost_beta3(beta)
    assert out.x.simplify() == (
        -px * py * y
        - px * pz * z
        - px * t * sympy.sqrt(-(px**2) - py**2 - pz**2 + 1)
        - px * t
        + py**2 * x
        + pz**2 * x
        - x * sympy.sqrt(-(px**2) - py**2 - pz**2 + 1)
        - x
    ) / (px**2 + py**2 + pz**2 - sympy.sqrt(-(px**2) - py**2 - pz**2 + 1) - 1)
    assert out.y.simplify() == (
        px**2 * y
        - px * py * x
        - py * pz * z
        - py * t * sympy.sqrt(-(px**2) - py**2 - pz**2 + 1)
        - py * t
        + pz**2 * y
        - y * sympy.sqrt(-(px**2) - py**2 - pz**2 + 1)
        - y
    ) / (px**2 + py**2 + pz**2 - sympy.sqrt(-(px**2) - py**2 - pz**2 + 1) - 1)
    assert out.z.simplify() == (
        px**2 * z
        - px * pz * x
        + py**2 * z
        - py * pz * y
        - pz * t * sympy.sqrt(-(px**2) - py**2 - pz**2 + 1)
        - pz * t
        - z * sympy.sqrt(-(px**2) - py**2 - pz**2 + 1)
        - z
    ) / (px**2 + py**2 + pz**2 - sympy.sqrt(-(px**2) - py**2 - pz**2 + 1) - 1)
    assert out.t.simplify() == (px * x + py * y + pz * z + t) / sympy.sqrt(
        -(px**2) - py**2 - pz**2 + 1
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
            "xyz",
            "xytheta",
            "xyeta",
            "rhophiz",
            "rhophitheta",
            "rhophieta",
        ):
            tvec, tbeta = getattr(vec, "to_" + t1)(), getattr(beta, "to_" + t2)()
            out = tvec.boost_beta3(tbeta)
            assert out.x.subs(values).evalf() == pytest.approx(3.5537720741941676)
            assert out.y.subs(values).evalf() == pytest.approx(5.0645264890330015)
            assert out.z.subs(values).evalf() == pytest.approx(6.575280903871835)
            assert out.t.subs(values).evalf() == pytest.approx(9.138547120755076)
