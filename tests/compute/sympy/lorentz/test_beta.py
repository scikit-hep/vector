# Copyright (c) t19-t24, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, rho, phi, z, t = sympy.symbols("x y rho phi z t", real=True, positive=True)
values = {x: 3, y: 4, rho: 5, phi: 0, z: 10, t: 20}


def test_xy_z_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.beta == sympy.sqrt(x**2 + y**2 + z**2) / t
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)


def test_xy_z_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert vec.beta.simplify() == sympy.sqrt(x**2 + y**2 + z**2) / sympy.sqrt(
        x**2 + y**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)


def test_xy_theta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.beta.simplify() == sympy.sqrt(x**2 + y**2 + z**2) / t
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)


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
    assert vec.beta.simplify() == sympy.sqrt(
        x**4 + 2 * x**2 * y**2 + x**2 * z**2 + y**4 + y**2 * z**2
    ) / (
        sympy.sqrt(x**2 + y**2)
        * sympy.sqrt(x**2 + y**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
    )
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)


def test_xy_eta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    assert (
        vec.beta.simplify()
        == 1.0 * sympy.sqrt(x**2 + y**2) * sympy.sqrt(z**2 / (x**2 + y**2) + 1) / t
    )
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)


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
    # TODO: why won't sympy equate the expressions without double
    # simplifying?
    assert (
        sympy.simplify(
            vec.beta.simplify()
            - 1.0
            * sympy.sqrt(x**2 + y**2)
            * sympy.sqrt(z**2 / (x**2 + y**2) + 1)
            / sympy.sqrt(
                (
                    0.25
                    * (x**2 + y**2)
                    * (sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2))) + 1) ** 2
                    + sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
                    * sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
                )
                * sympy.exp(-2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
            )
        )
        == 0
    )
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)


def test_rhophi_z_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.beta == sympy.sqrt(rho**2 + z**2) / t
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)


def test_rhophi_z_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
        ),
    )
    assert vec.beta.simplify() == sympy.sqrt(rho**2 + z**2) / sympy.sqrt(
        rho**2 + z**2 + sympy.Abs(rho**2 - t**2 + z**2)
    )
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)


def test_rhophi_theta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(rho**2 + z**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.beta.simplify() == sympy.sqrt(rho**2 + z**2) / t
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)


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
    assert vec.beta.simplify() == sympy.sqrt(rho**2 + z**2) / sympy.sqrt(
        rho**2 + z**2 + sympy.Abs(rho**2 - t**2 + z**2)
    )
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)


def test_rhophi_eta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    assert vec.beta.simplify() == 1.0 * rho * sympy.sqrt(1 + z**2 / rho**2) / t
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)


def test_rhophi_eta_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - t**2 + z**2))
        ),
    )
    assert vec.beta.simplify() == 1.0 * rho * sympy.sqrt(
        1 + z**2 / rho**2
    ) / sympy.sqrt(
        (
            0.25 * rho**2 * (sympy.exp(2 * sympy.asinh(z / rho)) + 1) ** 2
            + sympy.exp(2 * sympy.asinh(z / rho)) * sympy.Abs(rho**2 - t**2 + z**2)
        )
        * sympy.exp(-2 * sympy.asinh(z / rho))
    )
    assert vec.beta.subs(values).evalf() == pytest.approx(0.5590169943749475)
