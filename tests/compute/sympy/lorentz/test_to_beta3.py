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
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == x / t
    assert out.y == y / t
    assert out.z == z / t
    assert out.x.subs(values).evalf() == pytest.approx(3 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(4 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)


def test_xy_z_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
        ),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == x / sympy.sqrt(
        x**2 + y**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.y == y / sympy.sqrt(
        x**2 + y**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.z == z / sympy.sqrt(
        x**2 + y**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.x.subs(values).evalf() == pytest.approx(3 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(4 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)


def test_xy_theta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == x / t
    assert out.y == y / t
    assert out.z.simplify() == z / sympy.Abs(t)
    assert out.x.subs(values).evalf() == pytest.approx(3 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(4 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)


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
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x.simplify() == x / sympy.sqrt(
        x**2 + y**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.y.simplify() == y / sympy.sqrt(
        x**2 + y**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.z.simplify() == z * sympy.sqrt(x**2 + y**2) / sympy.sqrt(
        x**2 * (x**2 + y**2 + z**2)
        + x**2 * sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
        + y**2 * (x**2 + y**2 + z**2)
        + y**2 * sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.x.subs(values).evalf() == pytest.approx(3 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(4 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)


def test_xy_eta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == x / t
    assert out.y == y / t
    assert out.z == z * sympy.sqrt(x**2 / t**2 + y**2 / t**2) / sympy.sqrt(x**2 + y**2)
    assert out.x.subs(values).evalf() == pytest.approx(3 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(4 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)


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
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == x / sympy.sqrt(
        0.25
        * (1 + sympy.exp(-2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))) ** 2
        * (x**2 + y**2)
        * sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
        + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.y == y / sympy.sqrt(
        0.25
        * (1 + sympy.exp(-2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))) ** 2
        * (x**2 + y**2)
        * sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
        + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    # TODO: why won't sympy equate the expressions without double
    # simplifying?
    assert (
        sympy.simplify(
            out.z
            - z
            * sympy.sqrt(
                sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
                / (
                    0.25
                    * (x**2 + y**2)
                    * (sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2))) + 1) ** 2
                    + sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
                    * sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
                )
            )
        )
        == 0
    )
    assert out.x.subs(values).evalf() == pytest.approx(3 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(4 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)


def test_rhophi_z_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == rho * sympy.cos(phi) / t
    assert out.y == rho * sympy.sin(phi) / t
    assert out.z == z / t
    assert out.x.subs(values).evalf() == pytest.approx(5 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(0 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)


def test_rhophi_z_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
        ),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == rho * sympy.cos(phi) / sympy.sqrt(
        rho**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.y == rho * sympy.sin(phi) / sympy.sqrt(
        rho**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.z == z / sympy.sqrt(
        rho**2 + z**2 + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.x.subs(values).evalf() == pytest.approx(5 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(0 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)


def test_rhophi_theta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == rho * sympy.cos(phi) / t
    assert out.y == rho * sympy.sin(phi) / t
    assert out.z.simplify() == rho * z / (t * sympy.sqrt(x**2 + y**2))
    assert out.x.subs(values).evalf() == pytest.approx(5 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(0 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)


def test_rhophi_theta_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
        ),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == rho * sympy.cos(phi) / sympy.sqrt(
        rho**2 / (-(z**2) / (x**2 + y**2 + z**2) + 1)
        + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.y == rho * sympy.sin(phi) / sympy.sqrt(
        rho**2 / (-(z**2) / (x**2 + y**2 + z**2) + 1)
        + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.z.simplify() == rho * z / sympy.sqrt(
        rho**2 * (x**2 + y**2 + z**2)
        + (x**2 + y**2) * sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.x.subs(values).evalf() == pytest.approx(5 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(0 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)


def test_rhophi_eta_t():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == rho * sympy.cos(phi) / t
    assert out.y == rho * sympy.sin(phi) / t
    assert out.z == rho * z / (t * sympy.sqrt(x**2 + y**2))
    assert out.x.subs(values).evalf() == pytest.approx(5 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(0 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)


def test_rhophi_eta_tau():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        temporal=vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))
        ),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.sympy.VectorSympy3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == rho * sympy.cos(phi) / sympy.sqrt(
        0.25
        * rho**2
        * (1 + sympy.exp(-2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))) ** 2
        * sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
        + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.y == rho * sympy.sin(phi) / sympy.sqrt(
        0.25
        * rho**2
        * (1 + sympy.exp(-2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))) ** 2
        * sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
        + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
    )
    assert out.z == rho * z / (
        sympy.sqrt(x**2 + y**2)
        * sympy.sqrt(
            0.25
            * rho**2
            * (1 + sympy.exp(-2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))) ** 2
            * sympy.exp(2 * sympy.asinh(z / sympy.sqrt(x**2 + y**2)))
            + sympy.Abs(-(t**2) + x**2 + y**2 + z**2)
        )
    )
    assert out.x.subs(values).evalf() == pytest.approx(5 / 20)
    assert out.y.subs(values).evalf() == pytest.approx(0 / 20)
    assert out.z.subs(values).evalf() == pytest.approx(10 / 20)
