# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Spatial.unit(self)
"""

from __future__ import annotations

import typing
from math import inf

import numpy

from vector._compute.spatial import mag
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


def xy_z(lib, x, y, z):
    norm = mag.xy_z(lib, x, y, z)
    return (
        lib.nan_to_num(x / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(y / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(z / norm, nan=0, posinf=inf, neginf=-inf),
    )


def xy_theta(lib, x, y, theta):
    norm = mag.xy_theta(lib, x, y, theta)
    return (
        lib.nan_to_num(x / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(y / norm, nan=0, posinf=inf, neginf=-inf),
        theta,
    )


def xy_eta(lib, x, y, eta):
    norm = mag.xy_eta(lib, x, y, eta)
    return (
        lib.nan_to_num(x / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(y / norm, nan=0, posinf=inf, neginf=-inf),
        eta,
    )


def rhophi_z(lib, rho, phi, z):
    norm = mag.rhophi_z(lib, rho, phi, z)
    return (
        lib.nan_to_num(rho / norm, nan=0, posinf=inf, neginf=-inf),
        phi,
        lib.nan_to_num(z / norm, nan=0, posinf=inf, neginf=-inf),
    )


def rhophi_theta(lib, rho, phi, theta):
    norm = mag.rhophi_theta(lib, rho, phi, theta)
    return (lib.nan_to_num(rho / norm, nan=0, posinf=inf, neginf=-inf), phi, theta)


def rhophi_eta(lib, rho, phi, eta):
    norm = mag.rhophi_eta(lib, rho, phi, eta)
    return (lib.nan_to_num(rho / norm, nan=0, posinf=inf, neginf=-inf), phi, eta)


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


def dispatch(v: typing.Any) -> typing.Any:
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
            v._wrap_dispatched_function(function)(
                v.lib, *v.azimuthal.elements, *v.longitudinal.elements
            ),
            returns,
            1,
        )
