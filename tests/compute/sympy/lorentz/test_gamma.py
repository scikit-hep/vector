# Copyright (c) t19-t24, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

x, y, rho, phi, z, t = sympy.symbols("x y rho phi z t", real=True)


def test_xy_z_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.gamma == t / sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))


def test_xy_z_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert vec.gamma.simplify() == sympy.sqrt(
        x**2 + y**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    ) / sympy.Abs(sympy.sqrt(-(t**2) + x**2 + y**2 + z**2))


def test_xy_theta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.gamma.simplify() == t / sympy.Abs(sympy.sqrt(t**2 - x**2 - y**2 - z**2))


def test_xy_theta_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert vec.gamma.simplify() == sympy.sqrt(
        x**2 + y**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    ) / sympy.Abs(sympy.sqrt(-(t**2) + x**2 + y**2 + z**2))


def test_xy_eta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    # TODO: the expression blows up on simplifying?
    assert vec.gamma == t / sympy.sqrt(
        sympy.Abs(
            t**2
            - 0.25
            * (1 + sympy.exp(-2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))) ** 2
            * (x**2 + y**2)
            * sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
        )
    )


def test_xy_eta_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
        ),
    )
    # TODO: the expression blows up on simplifying?
    assert vec.gamma == sympy.sqrt(
        0.25
        * (1 + sympy.exp(-2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))) ** 2
        * (x**2 + y**2)
        * sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
        + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    ) / sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))


def test_rhophi_z_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.gamma.simplify() == t / sympy.Abs(sympy.sqrt(rho**2 - t**2 + z**2))


def test_rhophi_z_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
        ),
    )
    assert vec.gamma.simplify() == sympy.sqrt(
        rho**2 + z**2 + sympy.Abs(rho**2 - t**2 + z**2)
    ) / sympy.Abs(sympy.sqrt(rho**2 - t**2 + z**2))


def test_rhophi_theta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(rho**2 + z**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.gamma.simplify() == t / sympy.Abs(sympy.sqrt(-(rho**2) + t**2 - z**2))


def test_rhophi_theta_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(rho**2 + z**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
        ),
    )
    assert vec.gamma.simplify() == sympy.sqrt(
        rho**2 + z**2 + sympy.Abs(rho**2 - t**2 + z**2)
    ) / sympy.Abs(sympy.sqrt(rho**2 - t**2 + z**2))


def test_rhophi_eta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    # TODO: the expression blows up on simplifying?
    assert vec.gamma == t / sympy.sqrt(
        sympy.Abs(
            0.25
            * rho**2
            * (1 + sympy.exp(-2 * sympy.asinh(z / rho))) ** 2
            * sympy.exp(2 * sympy.asinh(z / rho))
            - t**2
        )
    )


def test_rhophi_eta_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
        ),
    )
    # TODO: the expression blows up on simplifying?
    assert vec.gamma == sympy.sqrt(
        0.25
        * rho**2
        * (1 + sympy.exp(-2 * sympy.asinh(z / rho))) ** 2
        * sympy.exp(2 * sympy.asinh(z / rho))
        + sympy.Abs(rho**2 - t**2 + z**2)
    ) / sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
