# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, nx, ny, mx, my, z, nz, mz = sympy.symbols("x y nx ny mx my z nz mz", real=True)
values = {
    x: 0.1,
    y: 0.2,
    nx: -0.3,
    ny: -0.6,
    mx: 0.3,
    my: 0.6,
    z: 0.3,
    nz: -0.9,
    mz: 0.9,
}


def test_planar_sympy():
    v1 = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y))
    v2 = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny))
    v3 = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(mx, my))
    assert v1.is_antiparallel(-v1).subs(values)
    assert v2.is_antiparallel(-v2).subs(values)
    assert v3.is_antiparallel(-v3).subs(values)
    assert v1.is_antiparallel(v2).subs(values)
    assert v2.is_antiparallel(v3).subs(values)
    assert not v1.is_antiparallel(v3).subs(values)

    for t1 in "xy", "rhophi":
        for t2 in "xy", "rhophi":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_antiparallel(tr2).subs(values)


def test_spatial_sympy():
    v1 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    v2 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(nz),
    )
    v3 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(mx, my),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(mz),
    )
    assert v1.is_antiparallel(-v1).subs(values)
    assert v2.is_antiparallel(-v2).subs(values)
    assert v3.is_antiparallel(-v3).subs(values)
    assert v1.is_antiparallel(v2).subs(values)
    assert v2.is_antiparallel(v3).subs(values)
    assert not v1.is_antiparallel(v3).subs(values)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            tr1, tr2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            assert tr1.is_antiparallel(tr2).subs(values)
