# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z, time = sympy.symbols("x y z time")
values = {x: 1, y: 2, z: 3, time: 10}


def test_planar_sympy():
    v = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y))
    u = v.unit()
    assert type(u) is type(v)
    assert type(u.azimuthal) is type(v.azimuthal)
    assert u.rho.simplify() == 1

    for t1 in "xy", "rhophi":
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)
        assert type(u.azimuthal) is type(t.azimuthal)
        assert (
            u.rho == 1 if isinstance(u.rho, int) else u.rho.subs(values).evalf() == 1.0
        )


def test_spatial_sympy():
    v = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    u = v.unit()
    assert type(u) is type(v)
    assert type(u.azimuthal) is type(v.azimuthal)
    assert type(u.longitudinal) is type(v.longitudinal)
    assert u.mag.simplify() == 1

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)
        assert type(u.azimuthal) is type(t.azimuthal)
        assert type(u.longitudinal) is type(t.longitudinal)
        assert u.mag.subs(values).evalf() == 1.0


def test_lorentz_sympy():
    v = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(time),
    )
    u = v.unit()
    assert type(u) is type(v)
    assert type(u.azimuthal) is type(v.azimuthal)
    assert type(u.longitudinal) is type(v.longitudinal)
    assert type(u.temporal) is type(v.temporal)
    assert u.tau.simplify() == sympy.sqrt(
        sympy.Abs(
            (-(time**2) + x**2 + y**2 + z**2)
            / sympy.Abs(-(time**2) + x**2 + y**2 + z**2)
        )
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
        t = getattr(v, "to_" + t1)()
        u = t.unit()
        assert type(u) is type(t)
        assert type(u.azimuthal) is type(t.azimuthal)
        assert type(u.longitudinal) is type(t.longitudinal)
        assert type(u.temporal) is type(t.temporal)
        assert (
            u.tau == 1 if isinstance(u.tau, int) else u.tau.subs(values).evalf() == 1.0
        )
