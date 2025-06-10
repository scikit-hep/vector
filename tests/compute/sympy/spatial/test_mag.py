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

x, y, rho, phi, z = sympy.symbols("x y rho phi z", real=True, positive=True)
values = {x: 3, y: 4, rho: 5, phi: 0, z: 10}


def test_xy_z():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    assert vec.mag == sympy.sqrt(x**2 + y**2 + z**2)
    assert vec.mag.subs(values).evalf() == pytest.approx(math.sqrt(125))


def test_xy_theta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
    )
    assert vec.mag.simplify() == sympy.sqrt(x**2 + y**2 + z**2)
    assert vec.mag.subs(values).evalf() == pytest.approx(math.sqrt(125))


def test_xy_eta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
    )
    assert vec.mag.simplify() == 1.0 * sympy.sqrt(x**2 + y**2) * sympy.sqrt(
        z**2 / (x**2 + y**2) + 1
    )
    assert vec.mag.subs(values).evalf() == pytest.approx(math.sqrt(125))


def test_rhophi_z():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    assert vec.mag == sympy.sqrt(rho**2 + z**2)
    assert vec.mag.subs(values).evalf() == pytest.approx(math.sqrt(125))


def test_rhophi_theta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(rho**2 + z**2))
        ),
    )
    assert vec.mag.simplify() == sympy.sqrt(rho**2 + z**2)
    assert vec.mag.subs(values).evalf() == pytest.approx(math.sqrt(125))


def test_rhophi_eta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
    )
    assert vec.mag.simplify() == 1.0 * rho * sympy.sqrt(1 + z**2 / rho**2)
    assert vec.mag.subs(values).evalf() == pytest.approx(math.sqrt(125))
