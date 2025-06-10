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
values = {x: 3, y: 4, z: 2, t: 40, nx: 5, ny: 12, nz: 4, nt: 20}


def test_planar_sympy():
    v1 = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y))
    v2 = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny))
    out = v1.subtract(v2)
    assert out.x == x - nx
    assert out.y == y - ny

    for t1 in "xy", "rhophi":
        for t2 in "xy", "rhophi":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.subtract(transformed2)
            assert out.x.subs(values).evalf() == pytest.approx(-2)
            assert out.y.subs(values).evalf() == pytest.approx(-8)


def test_spatial_sympy():
    v1 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    v2 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(nz),
    )
    out = v1.subtract(v2)
    assert out.x == x - nx
    assert out.y == y - ny
    assert out.z == z - nz

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.subtract(transformed2)
            assert out.x.subs(values).evalf() == pytest.approx(-2)
            assert out.y.subs(values).evalf() == pytest.approx(-8)
            assert out.z.subs(values).evalf() == pytest.approx(-2)


def test_lorentz_sympy():
    v1 = vector.backends.sympy.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    v2 = vector.backends.sympy.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(nz),
        temporal=vector.backends.sympy.TemporalSympyT(nt),
    )
    out = v1.subtract(v2)
    assert out.x == x - nx
    assert out.y == y - ny
    assert out.z == z - nz
    assert out.t == t - nt

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
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.subtract(transformed2)
            assert out.x.subs(values).evalf() == pytest.approx(-2)
            assert out.y.subs(values).evalf() == pytest.approx(-8)
            assert out.z.subs(values).evalf() == pytest.approx(-2)
            assert out.t.subs(values).evalf() == pytest.approx(20)
