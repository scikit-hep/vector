# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Spatial.scale(self, factor)
"""

from __future__ import annotations

import typing

import numpy

from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    _aztype,
    _flavor_of,
    _from_signature,
    _ltype,
)


def rectify(lib, phi):
    return (phi + lib.pi) % (2 * lib.pi) - lib.pi


def xy_z(lib, factor, x, y, z):
    return (x * factor, y * factor, z * factor)


def xy_theta(lib, factor, x, y, theta):
    sign = lib.sign(factor)
    flip_if_negative = lib.absolute(theta + (0.5 * (sign - 1) * lib.pi))
    return (x * factor, y * factor, flip_if_negative)


def xy_eta(lib, factor, x, y, eta):
    return (x * factor, y * factor, eta * lib.sign(factor))


def rhophi_z(lib, factor, rho, phi, z):
    absfactor = lib.absolute(factor)
    sign = lib.sign(factor)
    turn_if_negative = -0.5 * (sign - 1) * lib.pi
    return (rho * absfactor, rectify(lib, phi + turn_if_negative), z * factor)


def rhophi_theta(lib, factor, rho, phi, theta):
    absfactor = lib.absolute(factor)
    sign = lib.sign(factor)
    turn_if_negative = -0.5 * (sign - 1) * lib.pi
    flip_if_negative = lib.absolute(theta + (0.5 * (sign - 1) * lib.pi))
    return (rho * absfactor, rectify(lib, phi + turn_if_negative), flip_if_negative)


def rhophi_eta(lib, factor, rho, phi, eta):
    absfactor = lib.absolute(factor)
    sign = lib.sign(factor)
    turn_if_negative = -0.5 * (sign - 1) * lib.pi
    return (rho * absfactor, rectify(lib, phi + turn_if_negative), eta * sign)


dispatch_map = {
    (AzimuthalXY, LongitudinalZ): (xy_z, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalTheta): (xy_theta, AzimuthalXY, LongitudinalTheta),
    (AzimuthalXY, LongitudinalEta): (xy_eta, AzimuthalXY, LongitudinalEta),
    (AzimuthalRhoPhi, LongitudinalZ): (rhophi_z, AzimuthalRhoPhi, LongitudinalZ),
    (AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_theta,
        AzimuthalRhoPhi,
        LongitudinalTheta,
    ),
    (AzimuthalRhoPhi, LongitudinalEta): (rhophi_eta, AzimuthalRhoPhi, LongitudinalEta),
}


def dispatch(factor: typing.Any, v: typing.Any) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v),
            _ltype(v),
        ),
    )
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v),
            function(v.lib, factor, *v.azimuthal.elements, *v.longitudinal.elements),
            returns,
            1,
        )
