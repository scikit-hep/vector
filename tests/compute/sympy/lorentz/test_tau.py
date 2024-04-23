# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, rho, phi, z, t = sympy.symbols("x y rho phi z t", real=True)
values = {x: 3, y: 4, rho: 5, phi: 0, z: 10, t: 20}


def test_xy_z_t():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.tau == sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)


def test_xy_z_tau():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert vec.tau.simplify() == sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)


def test_xy_theta_t():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.tau.simplify() == sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)


def test_xy_theta_tau():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert vec.tau.simplify() == sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)


def test_xy_eta_t():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        vector.backends.sympy.TemporalSympyT(t),
    )
    # TODO: the expression blows up on simplifying?
    assert vec.tau == sympy.sqrt(
        sympy.Abs(
            t**2
            - 0.25
            * (1 + sympy.exp(-2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))) ** 2
            * (x**2 + y**2)
            * sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
        )
    )
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)


def test_xy_eta_tau():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert vec.tau.simplify() == sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)


def test_rhophi_z_t():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.tau.simplify() == sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)


def test_rhophi_z_tau():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
        ),
    )
    assert vec.tau.simplify() == sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)


def test_rhophi_theta_t():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(rho**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.tau.simplify() == sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)


def test_rhophi_theta_tau():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(rho**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
        ),
    )
    assert vec.tau.simplify() == sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)


def test_rhophi_eta_t():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
        vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.tau.simplify() == sympy.sqrt(sympy.Abs(1.0 * rho**2 - t**2 + 1.0 * z**2))
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)


def test_rhophi_eta_tau():
    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
        ),
    )
    assert vec.tau.simplify() == sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
    assert vec.tau.subs(values).evalf() == pytest.approx(16.583123951777)
