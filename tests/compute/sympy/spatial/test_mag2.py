# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, rho, phi, z = sympy.symbols("x y rho phi z", real=True)
values = {x: 3, y: 4, rho: 5, phi: 0, z: 10}


def test_xy_z():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    assert vec.mag2 == x**2 + y**2 + z**2
    assert vec.mag2.subs(values).evalf() == pytest.approx(125)


def test_xy_theta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
    )
    assert vec.mag2.simplify() == x**2 + y**2 + z**2
    assert vec.mag2.subs(values).evalf() == pytest.approx(125)


def test_xy_eta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
    )
    # TODO: why won't sympy equate the expressions without double
    # simplifying?
    assert (
        sympy.simplify(
            vec.mag2.simplify() - 0.25 * (x**2 + y**2) * (4 * z**2 / (x**2 + y**2) + 4)
        )
        == 0
    )
    assert vec.mag2.subs(values).evalf() == pytest.approx(125)


def test_rhophi_z():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    assert vec.mag2 == rho**2 + z**2
    assert vec.mag2.subs(values).evalf() == pytest.approx(125)


def test_rhophi_theta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(rho**2 + z**2))
        ),
    )
    assert vec.mag2.simplify() == rho**2 + z**2
    assert vec.mag2.subs(values).evalf() == pytest.approx(125)


def test_rhophi_eta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
    )
    assert vec.mag2.simplify() == 0.25 * rho**2 * (4 + 4 * z**2 / rho**2)
    assert vec.mag2.subs(values).evalf() == pytest.approx(125)
