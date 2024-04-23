# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z, nx, ny, nz, t = sympy.symbols("x y z nx ny nz t", real=True, positive=True)
values = {x: 0.1, y: 0.2, z: 0.3, nx: 0.4, ny: 0.5, nz: 0.6}


def test_spatial_sympy():
    v1 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    v2 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(nz),
    )
    assert v1.deltaangle(v2) == sympy.acos(
        (nx * x + ny * y + nz * z)
        / (sympy.sqrt(nx**2 + ny**2 + nz**2) * sympy.sqrt(x**2 + y**2 + z**2))
    )

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert transformed1.deltaangle(transformed2).subs(values) == pytest.approx(
                0.225726
            )


def test_lorentz_sympy():
    v1 = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    v2 = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(nz),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    assert v1.deltaangle(v2) == sympy.acos(
        (nx * x + ny * y + nz * z)
        / (sympy.sqrt(nx**2 + ny**2 + nz**2) * sympy.sqrt(x**2 + y**2 + z**2))
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
            assert tr1.deltaangle(tr2).subs(values) == pytest.approx(0.225726)
