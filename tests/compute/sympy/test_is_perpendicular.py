# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, nx, ny, mx, my, z, nz, mz = sympy.symbols("x y nx ny mx my z nz mz", real=True)
values = {x: 0, y: 1, nx: 1, ny: 0, mx: 0.1, my: 0.5, z: 1, nz: 0, mz: 0.9}


def test_planar_sympy():
    v1 = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(0, 1))
    v2 = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(1, 0))
    v3 = vector.VectorSympy2D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(0.1, 0.5)
    )
    assert not v1.is_perpendicular(v1).subs(values)
    assert not v2.is_perpendicular(v2).subs(values)
    assert not v3.is_perpendicular(v3).subs(values)
    assert v1.is_perpendicular(v2).subs(values)
    assert not v1.is_perpendicular(v3).subs(values)
    assert not v2.is_perpendicular(v3).subs(values)

    for t1 in "xy", "rhophi":
        for t2 in "xy", "rhophi":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_perpendicular(tr2).subs(values)


def test_spatial_sympy():
    v1 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(0, 1),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(1),
    )
    v2 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(1, 0),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(0),
    )
    v3 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(0.1, 0.5),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(0.9),
    )
    assert not v1.is_perpendicular(v1).subs(values)
    assert not v2.is_perpendicular(v2).subs(values)
    assert not v3.is_perpendicular(v3).subs(values)
    assert v1.is_perpendicular(v2).subs(values)
    assert not v1.is_perpendicular(v3).subs(values)
    assert not v2.is_perpendicular(v3).subs(values)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_perpendicular(tr2).subs(values)
