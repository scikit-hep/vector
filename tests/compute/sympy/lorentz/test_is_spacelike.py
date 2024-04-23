# Copyright (c) 019-024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, rho, phi, z, space_t, light_t, time_t = sympy.symbols(
    "x y rho phi z space_t light_t time_t", real=True
)
values = {x: 1, y: 0, rho: 1, phi: 0, z: 0, space_t: 0, light_t: 1, time_t: 2}


def test_xy_z_t():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyXY(x, y),
    #     vector.backends.sympy.LongitudinalSympyZ(z),
    #     vector.backends.sympy.TemporalSympyT(space_t),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyT(light_t),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyT(time_t),
    )
    assert not vec.is_spacelike().subs(values)


def test_xy_z_tau():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyXY(x, y),
    #     vector.backends.sympy.LongitudinalSympyZ(z),
    #     vector.backends.sympy.TemporalSympyTau(sympy.sqrt(sympy.Abs(-(space_t**2) + x**2 + y**2 + z**2))),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(light_t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(time_t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)


def test_xy_theta_t():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyXY(x, y),
    #     vector.backends.sympy.LongitudinalSympyTheta(sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))),
    #     vector.backends.sympy.TemporalSympyT(space_t),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyT(light_t),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyT(time_t),
    )
    assert not vec.is_spacelike().subs(values)


def test_xy_theta_tau():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyXY(x, y),
    #     vector.backends.sympy.LongitudinalSympyTheta(sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))),
    #     vector.backends.sympy.TemporalSympyTau(sympy.sqrt(sympy.Abs(-(space_t**2) + x**2 + y**2 + z**2))),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(light_t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(time_t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)


def test_xy_eta_t():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyXY(x, y),
    #     vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / sympy.sqrt(x**2 + y**2))),
    #     vector.backends.sympy.TemporalSympyT(space_t),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        vector.backends.sympy.TemporalSympyT(light_t),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        vector.backends.sympy.TemporalSympyT(time_t),
    )
    assert not vec.is_spacelike().subs(values)


def test_xy_eta_tau():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyXY(x, y),
    #     vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / sympy.sqrt(x**2 + y**2))),
    #     vector.backends.sympy.TemporalSympyTau(sympy.sqrt(sympy.Abs(-(space_t**2) + x**2 + y**2 + z**2))),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(light_t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyXY(x, y),
        vector.backends.sympy.LongitudinalSympyEta(
            sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        ),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(-(time_t**2) + x**2 + y**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)


def test_rhophi_z_t():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
    #     vector.backends.sympy.LongitudinalSympyZ(z),
    #     vector.backends.sympy.TemporalSympyT(space_t),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyT(light_t),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyT(time_t),
    )
    assert not vec.is_spacelike().subs(values)


def test_rhophi_z_tau():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
    #     vector.backends.sympy.LongitudinalSympyZ(z),
    #     vector.backends.sympy.TemporalSympyTau(sympy.sqrt(sympy.Abs(rho**2 - space_t**2 + z**2))),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - light_t**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyZ(z),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - time_t**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)


def test_rhophi_theta_t():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
    #     vector.backends.sympy.LongitudinalSympyTheta(sympy.acos(z / sympy.sqrt(rho**2 + z**2))),
    #     vector.backends.sympy.TemporalSympyT(space_t),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(rho**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyT(light_t),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(rho**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyT(time_t),
    )
    assert not vec.is_spacelike().subs(values)


def test_rhophi_theta_tau():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
    #     vector.backends.sympy.LongitudinalSympyTheta(sympy.acos(z / sympy.sqrt(rho**2 + z**2))),
    #     vector.backends.sympy.TemporalSympyTau(sympy.sqrt(sympy.Abs(rho**2 - space_t**2 + z**2))),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(rho**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - light_t**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyTheta(
            sympy.acos(z / sympy.sqrt(rho**2 + z**2))
        ),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - time_t**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)


def test_rhophi_eta_t():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
    #     vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
    #     vector.backends.sympy.TemporalSympyT(space_t),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
        vector.backends.sympy.TemporalSympyT(light_t),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
        vector.backends.sympy.TemporalSympyT(time_t),
    )
    assert not vec.is_spacelike().subs(values)


def test_rhophi_eta_tau():
    # the following test fails, but it represent the t**2 < mag**2 case
    # so it should be okay for it to fail
    # vec = vector.VectorSympy4D(
    #     vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
    #     vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
    #     vector.backends.sympy.TemporalSympyTau(sympy.sqrt(sympy.Abs(rho**2 - space_t**2 + z**2))),
    # )
    # assert vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - light_t**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)

    vec = vector.VectorSympy4D(
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        vector.backends.sympy.LongitudinalSympyEta(sympy.asinh(z / rho)),
        vector.backends.sympy.TemporalSympyTau(
            sympy.sqrt(sympy.Abs(rho**2 - time_t**2 + z**2))
        ),
    )
    assert not vec.is_spacelike().subs(values)
