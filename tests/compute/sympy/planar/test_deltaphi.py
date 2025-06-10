# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import math

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, nx, ny = sympy.symbols("x y nx ny")
values = {x: 1, y: 2, nx: 3, ny: 4}


def test_xy_xy():
    v1 = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y))
    v2 = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny))
    assert (
        v1.deltaphi(v2)
        == sympy.Mod(-sympy.atan2(ny, nx) + sympy.atan2(y, x) + sympy.pi, 2 * sympy.pi)
        - sympy.pi
    )
    assert v1.deltaphi(v2).subs(values).evalf() == pytest.approx(
        math.atan2(2, 1) - math.atan2(4, 3)
    )


def test_xy_rhophi():
    v1 = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y))
    v2 = vector.VectorSympy2D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(nx, ny)
    )
    assert (
        v1.deltaphi(v2)
        == sympy.Mod(-ny + sympy.atan2(y, x) + sympy.pi, 2 * sympy.pi) - sympy.pi
    )
    assert v1.deltaphi(v2).subs(values).evalf() == pytest.approx(math.atan2(2, 1) - 4)


def test_rhophi_xy():
    v1 = vector.VectorSympy2D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(x, y)
    )
    v2 = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny))
    assert (
        v1.deltaphi(v2)
        == sympy.Mod(y - sympy.atan2(ny, nx) + sympy.pi, 2 * sympy.pi) - sympy.pi
    )
    assert v1.deltaphi(v2).subs(values).evalf() == pytest.approx(2 - math.atan2(4, 3))


def test_rhophi_rhophi():
    v1 = vector.VectorSympy2D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(x, y)
    )
    v2 = vector.VectorSympy2D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(nx, ny)
    )
    assert v1.deltaphi(v2) == sympy.Mod(-ny + y + sympy.pi, 2 * sympy.pi) - sympy.pi
    assert v1.deltaphi(v2).subs(values).evalf() == pytest.approx(2 - 4)
