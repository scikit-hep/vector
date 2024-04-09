# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

x, y, rho, phi, eta = sympy.symbols("x y rho phi eta", real=True, positive=True)


def test_xy_z():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(
            sympy.sqrt(x**2 + y**2) * sympy.sinh(eta)
        ),
    )
    assert vec.eta.simplify() == sympy.asinh(sympy.sinh(eta))


def test_xy_theta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            2.0 * sympy.atan(sympy.exp(-eta))
        ),
    )
    assert vec.eta.simplify() == -sympy.log(
        sympy.tan(1.0 * sympy.atan(sympy.exp(-eta)))
    )


def test_xy_eta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(eta),
    )
    assert vec.eta == eta


def test_rhophi_z():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(rho * sympy.sinh(eta)),
    )
    assert vec.eta.simplify() == sympy.asinh(sympy.sinh(eta))


def test_rhophi_theta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyTheta(
            2.0 * sympy.atan(sympy.exp(-eta))
        ),
    )
    assert vec.eta.simplify() == -sympy.log(
        sympy.tan(1.0 * sympy.atan(sympy.exp(-eta)))
    )


def test_rhophi_eta():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi),
        longitudinal=vector.backends.sympy.LongitudinalSympyEta(eta),
    )
    assert vec.eta == eta
