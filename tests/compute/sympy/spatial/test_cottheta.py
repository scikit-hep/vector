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

x, y, rho, phi, theta = sympy.symbols("x y rho phi theta", real=True)
values = {x: 3, y: 4, rho: 5, phi: 0, theta: 0.4636476090008061}


def test_xy_z():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(
            sympy.sqrt(x**2 + y**2) / sympy.tan(theta)
        ),
    )
    assert vec.cottheta == 1 / sympy.tan(theta)
    assert vec.cottheta.subs(values).evalf() == pytest.approx(
        1 / math.tan(0.4636476090008061)
    )


def test_xy_theta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(theta),
    )
    assert vec.cottheta == 1 / sympy.tan(theta)
    assert vec.cottheta.subs(values).evalf() == pytest.approx(
        1 / math.tan(0.4636476090008061)
    )


def test_xy_eta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(
            -sympy.log(sympy.tan(0.5 * theta))
        ),
    )
    assert vec.cottheta == 1 / sympy.tan(2.0 * sympy.atan(sympy.tan(0.5 * theta)))
    assert vec.cottheta.subs(values).evalf() == pytest.approx(
        1 / math.tan(0.4636476090008061)
    )


def test_rhophi_z():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(rho / sympy.tan(theta)),
    )
    assert vec.cottheta == 1 / sympy.tan(theta)
    assert vec.cottheta.subs(values).evalf() == pytest.approx(
        1 / math.tan(0.4636476090008061)
    )


def test_rhophi_theta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(theta),
    )
    assert vec.cottheta == 1 / sympy.tan(theta)
    assert vec.cottheta.subs(values).evalf() == pytest.approx(
        1 / math.tan(0.4636476090008061)
    )


def test_rhophi_eta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(
            -sympy.log(sympy.tan(0.5 * theta))
        ),
    )
    assert vec.cottheta == 1 / sympy.tan(2.0 * sympy.atan(sympy.tan(0.5 * theta)))
    assert vec.cottheta.subs(values).evalf() == pytest.approx(
        1 / math.tan(0.4636476090008061)
    )
