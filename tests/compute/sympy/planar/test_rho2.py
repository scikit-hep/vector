# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y = sympy.symbols("x y")
values = {x: 3, y: 4}


def test_xy():
    vec = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y))
    assert vec.rho2 == x**2 + y**2
    assert vec.rho2.subs(values) == pytest.approx(25)


def test_rhophi():
    vec = vector.VectorSympy2D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(
            sympy.sqrt(x**2 + y**2), sympy.atan2(y, x)
        )
    )
    assert vec.rho2 == x**2 + y**2
    assert vec.rho2.subs(values) == pytest.approx(25)
