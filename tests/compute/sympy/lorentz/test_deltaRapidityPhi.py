# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z, t, nx, ny, nz, nt = sympy.symbols("x y z t nx ny nz nt", real=True)
values = {x: 1.0, y: 1.0, z: 1.0, t: 1.0, nx: -1.0, ny: -1.0, nz: -1.0, nt: 1.0}


def test_lorentz_object():
    v1 = vector.MomentumSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyTau(t),
    )
    v2 = vector.MomentumSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(nx, ny),
        vector.backends.sympy.LongitudinalSympyZ(nz),
        vector.backends.sympy.TemporalSympyTau(nt),
    )
    assert v1.deltaRapidityPhi(v2).simplify() == sympy.sqrt(
        0.25
        * (
            sympy.log(
                (-nz - sympy.sqrt(nt**2 + nx**2 + ny**2 + nz**2))
                / (nz - sympy.sqrt(nt**2 + nx**2 + ny**2 + nz**2))
            )
            - sympy.log(
                (-z - sympy.sqrt(t**2 + x**2 + y**2 + z**2))
                / (z - sympy.sqrt(t**2 + x**2 + y**2 + z**2))
            )
        )
        ** 2
        + (
            sympy.Mod(-sympy.atan2(ny, nx) + sympy.atan2(y, x) + sympy.pi, 2 * sympy.pi)
            - sympy.pi
        )
        ** 2
    )

    expected_result = sympy.sqrt(
        # phi
        sympy.pi**2
        # rapidity
        + ((0.5 * sympy.log(3 / 1) - 0.5 * sympy.log(1 / 3)) ** 2)
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
            tr1, tr2 = getattr(v1, "to_" + t1)(), getattr(v2, "to_" + t2)()
            assert tr1.deltaRapidityPhi(tr2).subs(values).evalf() == pytest.approx(
                expected_result.evalf()
            )
